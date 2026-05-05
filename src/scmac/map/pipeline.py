# -------------------------------------------------------------------------
# High-level scMAC-MAP pipeline utilities.
#
# This module provides stable command-line building blocks for:
#   1. reproducing paired scRNA/TCR-seq cross-validation evaluation;
#   2. predicting clone-level functional probabilities from bulk TCR data;
#   3. aggregating clone-level probabilities into repertoire composition.
#
# The network architecture is imported from the original manuscript code.
# -------------------------------------------------------------------------

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.spatial.distance import braycurtis, jensenshannon
from scipy.stats import pearsonr
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader, Dataset

_MAP_DIR = Path(__file__).resolve().parent
if str(_MAP_DIR) not in sys.path:
    sys.path.insert(0, str(_MAP_DIR))

import utils  # noqa: E402
from network import scMACMap  # noqa: E402


@dataclass
class MapResources:
    """Runtime resources required by scMAC-MAP."""

    vocab: dict
    aa_features: dict
    latent_dim: int
    filters_n: int
    batch_size: int
    use_v: bool
    use_j: bool
    seed: int
    model_paths: list[Path]


class TCRMapDataset(Dataset):
    """Dataset for scMAC-MAP inference or paired-data evaluation."""

    def __init__(
        self,
        df: pd.DataFrame,
        aa_features: dict,
        max_len: int = 24,
        latent_dim: int = 64,
        include_labels: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.aa_features = aa_features
        self.max_len = max_len
        self.latent_dim = latent_dim
        self.include_labels = include_labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = str(row["CDR3"])
        padded = (seq + "X" * self.max_len)[: self.max_len]
        tcr_matrix = [self.aa_features.get(aa.upper(), [0.0] * 15) for aa in padded]
        cdr3_tensor = torch.FloatTensor(tcr_matrix).transpose(0, 1)

        v_idx = torch.tensor(int(row.get("V_idx", 0)), dtype=torch.long)
        j_idx = torch.tensor(int(row.get("J_idx", 0)), dtype=torch.long)
        latent = torch.zeros(self.latent_dim, dtype=torch.float32)
        label = torch.tensor(int(row.get("Label_idx", 0)), dtype=torch.long)

        if self.include_labels:
            return cdr3_tensor, v_idx, j_idx, latent, label
        return cdr3_tensor, v_idx, j_idx


def read_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def infer_latent_dim(reference_npz: str | Path | None = None, default: int = 64) -> int:
    """Infer latent dimension from a reference NPZ file when available."""
    if reference_npz is None:
        return default
    npz_path = Path(reference_npz)
    if not npz_path.exists():
        return default
    data = np.load(npz_path, allow_pickle=True)
    if "matrix" in data:
        return int(data["matrix"].shape[1])
    if "embeddings" in data:
        return int(data["embeddings"].shape[1])
    return default


def discover_model_paths(model_dir: str | Path, model_glob: str = "fold_*_map.pth") -> list[Path]:
    model_dir = Path(model_dir)
    paths = sorted(model_dir.glob(model_glob))
    if not paths:
        final_model = model_dir / "final_model_map.pth"
        if final_model.exists():
            paths = [final_model]
    if not paths:
        raise FileNotFoundError(f"No scMAC-MAP model files found in {model_dir}.")
    return paths


def load_resources(
    model_dir: str | Path,
    aaindex_path: str | Path,
    reference_npz: str | Path | None = None,
    latent_dim: int | None = None,
    model_glob: str = "fold_*_map.pth",
) -> MapResources:
    """Load vocabulary, AAindex features and model paths."""
    model_dir = Path(model_dir)
    vocab = read_json(model_dir / "vocab_map.json")
    config_path = model_dir / "config_map.json"
    config = read_json(config_path) if config_path.exists() else {}
    resolved_latent_dim = latent_dim or infer_latent_dim(reference_npz, default=64)
    return MapResources(
        vocab=vocab,
        aa_features=utils.get_features(str(aaindex_path)),
        latent_dim=resolved_latent_dim,
        filters_n=int(config.get("FILTERS_N", 1)),
        batch_size=int(config.get("BATCH_SIZE", 256)),
        use_v=bool(config.get("USE_V", True)),
        use_j=bool(config.get("USE_J", True)),
        seed=int(config.get("SEED", 2026)),
        model_paths=discover_model_paths(model_dir, model_glob=model_glob),
    )


def class_names_from_vocab(vocab: dict) -> list[str]:
    inv_label = {int(v): k for k, v in vocab["label_dict"].items()}
    return [inv_label[i] for i in sorted(inv_label)]


def add_vocab_indices(df: pd.DataFrame, vocab: dict, require_labels: bool = False) -> pd.DataFrame:
    """Map function labels and V/J genes to the integer vocabulary used by the model."""
    out = df.copy()
    out["CDR3"] = out["CDR3"].astype(str)
    out["V_idx"] = out["V"].map(vocab["v_dict"]).fillna(0).astype(int)
    out["J_idx"] = out["J"].map(vocab["j_dict"]).fillna(0).astype(int)
    if require_labels:
        out["Label_idx"] = out["Function"].map(vocab["label_dict"]).fillna(0).astype(int)
    elif "Function" in out.columns:
        out["Label_idx"] = out["Function"].map(vocab["label_dict"]).fillna(0).astype(int)
    else:
        out["Label_idx"] = 0
    return out


def load_paired_data(csv_file: str | Path, reference_npz: str | Path, vocab: dict) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Load paired TCR labels and single-cell reference embeddings for CV evaluation."""
    df = pd.read_csv(csv_file).dropna(subset=["CDR3"])
    npz = np.load(reference_npz, mmap_mode="r", allow_pickle=True)
    matrix = npz["matrix"]
    barcodes = npz["barcodes"]
    bc_to_idx = {(bc.decode() if isinstance(bc, bytes) else bc): idx for idx, bc in enumerate(barcodes)}
    df = df[df["Barcodes"].isin(bc_to_idx.keys())].reset_index(drop=True)
    df = add_vocab_indices(df, vocab, require_labels=True)
    return df, matrix, bc_to_idx


def predict_dataframe(
    df: pd.DataFrame,
    resources: MapResources,
    device: str = "auto",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict known-class probabilities, latent vectors and uncertainties for TCR rows."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    loader = DataLoader(
        TCRMapDataset(df, resources.aa_features, latent_dim=resources.latent_dim, include_labels=True),
        batch_size=resources.batch_size,
        shuffle=False,
    )

    all_known_probs = []
    all_latents = []
    all_uncertainties = []

    for model_path in resources.model_paths:
        model = scMACMap(
            v_vocab_size=resources.vocab["v_vocab_size"],
            j_vocab_size=resources.vocab["j_vocab_size"],
            latent_dim=resources.latent_dim,
            filters_num=resources.filters_n,
            class_num=resources.vocab["label_vocab_size"],
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        model_probs = []
        model_latents = []
        model_uncertainties = []
        with torch.no_grad():
            for b_cdr3, b_v, b_j, _, _ in loader:
                b_cdr3 = b_cdr3.to(device)
                b_v = b_v.to(device)
                b_j = b_j.to(device)
                outputs = model(b_cdr3, b_v, b_j)
                known_probs = F.softmax(outputs["logits"][:, 1:], dim=1).cpu().numpy()
                uncertainty = torch.exp(outputs["logvar"]).mean(dim=1).cpu().numpy()
                model_probs.append(known_probs)
                model_latents.append(outputs["z_tcr"].cpu().numpy())
                model_uncertainties.append(uncertainty)

        all_known_probs.append(np.vstack(model_probs))
        all_latents.append(np.vstack(model_latents))
        all_uncertainties.append(np.concatenate(model_uncertainties))

    return (
        np.mean(np.stack(all_known_probs, axis=0), axis=0),
        np.mean(np.stack(all_latents, axis=0), axis=0),
        np.mean(np.stack(all_uncertainties, axis=0), axis=0),
    )


def aggregate_composition(probabilities: np.ndarray, abundance: np.ndarray) -> pd.Series:
    """Aggregate clone-level probabilities into abundance-weighted repertoire composition."""
    abundance = np.asarray(abundance, dtype=float)
    if abundance.sum() <= 0:
        abundance = np.ones_like(abundance)
    weights = abundance / abundance.sum()
    composition = (probabilities * weights[:, None]).sum(axis=0)
    if composition.sum() > 0:
        composition = composition / composition.sum()
    return pd.Series(composition)


def evaluate_fold_composition_hard(y_true, y_pred, inv_dict, all_indices) -> dict:
    """Compute hard-composition metrics used in the manuscript evaluation."""
    total = len(y_true)
    rows = []
    for label_idx in all_indices:
        actual_p = (y_true == label_idx).sum() / total * 100
        pred_p = (y_pred == label_idx).sum() / total * 100
        rows.append(
            {
                "Cell_Type": inv_dict.get(label_idx, f"Class_{label_idx}"),
                "Actual_%": actual_p,
                "Predicted_%": pred_p,
                "Abs_Err": abs(actual_p - pred_p),
            }
        )

    comp_df = pd.DataFrame(rows)
    actual = comp_df["Actual_%"].values / 100.0
    pred = comp_df["Predicted_%"].values / 100.0
    r_val, _ = pearsonr(actual, pred) if len(actual) > 1 else (0, 0)
    r2_val = r2_score(actual, pred)
    ccc_val = (2 * np.cov(actual, pred, bias=True)[0][1]) / (
        np.var(actual) + np.var(pred) + (np.mean(actual) - np.mean(pred)) ** 2 + 1e-10
    )
    jsd_val = jensenshannon(actual + 1e-10, pred + 1e-10)
    bc_dist = braycurtis(actual, pred)
    macro_smape = np.mean(200 * np.abs(actual - pred) / (actual + pred + 1e-8))
    slope = LinearRegression().fit(actual.reshape(-1, 1), pred.reshape(-1, 1)).coef_[0][0]
    mae_val = comp_df["Abs_Err"].mean() / 100.0
    return {
        "r": r_val,
        "r2": r2_val,
        "ccc": ccc_val,
        "smape": macro_smape,
        "jsd": jsd_val,
        "bc": bc_dist,
        "mae": mae_val,
        "slope": slope,
    }


def run_cv_evaluation(
    csv_file: str | Path,
    reference_npz: str | Path,
    model_dir: str | Path,
    aaindex_path: str | Path,
    output_dir: str | Path,
    device: str = "auto",
    model_glob: str = "fold_*_map.pth",
) -> pd.DataFrame:
    """Run the paired-data 5-fold CV evaluation using pretrained fold models."""
    resources = load_resources(
        model_dir=model_dir,
        aaindex_path=aaindex_path,
        reference_npz=reference_npz,
        model_glob=model_glob,
    )
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    utils.seed_torch(resources.seed)

    df, _, _ = load_paired_data(csv_file, reference_npz, resources.vocab)
    splits = utils.get_mapping_splits(df, n_splits=5, is_unknown=False, seed=resources.seed)
    inv_label = {int(v): k for k, v in resources.vocab["label_dict"].items()}
    all_indices = sorted(df["Label_idx"].unique().tolist())

    overall_true = []
    overall_pred = []
    overall_barcodes = []
    fold_metrics = []

    original_model_paths = resources.model_paths
    for fold, (_, test_idx) in enumerate(splits, start=1):
        fold_path = Path(model_dir) / f"fold_{fold}_map.pth"
        if not fold_path.exists():
            continue
        resources.model_paths = [fold_path]
        test_df = df.iloc[test_idx].copy()
        probs, _, _ = predict_dataframe(test_df, resources, device=device)
        preds = np.argmax(probs, axis=1) + 1
        y_true = test_df["Label_idx"].values

        present_gt = np.unique(y_true).tolist()
        comp = evaluate_fold_composition_hard(y_true, preds, inv_label, all_indices)
        metrics = {
            "Fold": fold,
            "Accuracy": accuracy_score(y_true, preds),
            "Macro_F1": f1_score(y_true, preds, labels=present_gt, average="macro", zero_division=0),
            "Weighted_F1": f1_score(y_true, preds, average="weighted", zero_division=0),
            "Macro_P": precision_score(y_true, preds, labels=present_gt, average="macro", zero_division=0),
            "Macro_R": recall_score(y_true, preds, labels=present_gt, average="macro", zero_division=0),
            "sMAPE": comp["smape"],
            "R2": comp["r2"],
            "Pearson_r": comp["r"],
            "CCC": comp["ccc"],
            "Bray_Curtis": comp["bc"],
            "Slope": comp["slope"],
            "JSD": comp["jsd"],
            "MAE": comp["mae"],
        }
        fold_metrics.append(metrics)
        overall_true.extend(y_true)
        overall_pred.extend(preds)
        overall_barcodes.extend(test_df["Barcodes"].values)

    resources.model_paths = original_model_paths
    metrics_df = pd.DataFrame(fold_metrics)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    numeric_cols = [c for c in metrics_df.columns if c != "Fold"]
    means = metrics_df[numeric_cols].mean()
    stds = metrics_df[numeric_cols].std(ddof=1)
    overall_true_arr = np.array(overall_true)
    overall_pred_arr = np.array(overall_pred)
    global_gt = np.unique(overall_true_arr).tolist()
    formatted_df = metrics_df.copy()
    for col in numeric_cols:
        formatted_df[col] = formatted_df[col].map(lambda x: f"{x:.4f}")

    overall_row = {
        "Fold": "Overall_Pooled",
        "Accuracy": f"{accuracy_score(overall_true_arr, overall_pred_arr):.4f}",
        "Macro_F1": f"{f1_score(overall_true_arr, overall_pred_arr, labels=global_gt, average='macro', zero_division=0):.4f}",
        "Weighted_F1": f"{f1_score(overall_true_arr, overall_pred_arr, average='weighted', zero_division=0):.4f}",
        "Macro_P": f"{precision_score(overall_true_arr, overall_pred_arr, labels=global_gt, average='macro', zero_division=0):.4f}",
        "Macro_R": f"{recall_score(overall_true_arr, overall_pred_arr, labels=global_gt, average='macro', zero_division=0):.4f}",
    }
    for col in ["sMAPE", "R2", "Pearson_r", "CCC", "Bray_Curtis", "Slope", "JSD", "MAE"]:
        overall_row[col] = f"{means[col]:.4f} +/- {stds[col]:.4f}"

    full_df = pd.concat([formatted_df, pd.DataFrame([overall_row])], ignore_index=True)
    metrics_df.to_csv(output_dir / "eval_metrics_folds_numeric_map.csv", index=False)
    full_df.to_csv(output_dir / "eval_metrics_full_map.csv", index=False)

    prediction_df = pd.DataFrame(
        {
            "Barcodes": overall_barcodes,
            "True_Label_Idx": overall_true,
            "Pred_Label_Idx": overall_pred,
        }
    )
    prediction_df["True_Function"] = prediction_df["True_Label_Idx"].map(inv_label)
    prediction_df["Pred_Function"] = prediction_df["Pred_Label_Idx"].map(inv_label)
    prediction_df["Is_Correct"] = prediction_df["True_Label_Idx"] == prediction_df["Pred_Label_Idx"]
    prediction_df.to_csv(output_dir / "cv_prediction_results_map.csv", index=False)

    cm = confusion_matrix(overall_true_arr, overall_pred_arr, labels=all_indices)
    cm_df = pd.DataFrame(
        cm,
        index=[inv_label[i] + " (True)" for i in all_indices],
        columns=[inv_label[i] + " (Pred)" for i in all_indices],
    )
    cm_df.to_csv(output_dir / "confusion_matrix_overall_map.csv")

    return full_df


def compare_with_expected(observed: pd.DataFrame, expected_csv: str | Path, tolerance: float = 1e-4) -> dict:
    """Compare observed CV metrics with a previously exported reference metrics file."""
    expected = pd.read_csv(expected_csv)
    fold_rows = observed[observed["Fold"] != "Overall_Pooled"].copy()
    expected_fold_rows = expected[expected["Fold"] != "Overall_Pooled"].copy()
    comparable_cols = [
        "Accuracy",
        "Macro_F1",
        "Weighted_F1",
        "Macro_P",
        "Macro_R",
        "sMAPE",
        "R2",
        "Pearson_r",
        "CCC",
        "Bray_Curtis",
        "Slope",
        "JSD",
        "MAE",
    ]
    max_abs_diff = 0.0
    failures = []
    for col in comparable_cols:
        obs = fold_rows[col].astype(float).values
        exp = expected_fold_rows[col].astype(float).values
        diff = np.max(np.abs(obs - exp))
        max_abs_diff = max(max_abs_diff, float(diff))
        if diff > tolerance:
            failures.append({"metric": col, "max_abs_diff": float(diff)})
    return {
        "passed": len(failures) == 0,
        "tolerance": tolerance,
        "max_abs_diff": max_abs_diff,
        "failures": failures,
    }


def validate_columns(df: pd.DataFrame, required: Iterable[str], file_label: str):
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{file_label} is missing required columns: {missing}")
