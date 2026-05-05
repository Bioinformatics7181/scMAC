#!/usr/bin/env python
"""Predict functional context for a bulk TCR repertoire using scMAC-MAP."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scmac.map.pipeline import (  # noqa: E402
    add_vocab_indices,
    aggregate_composition,
    class_names_from_vocab,
    load_resources,
    predict_dataframe,
    validate_columns,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run scMAC-MAP inference on a bulk TCR repertoire."
    )
    parser.add_argument("--input-csv", required=True, help="Bulk TCR repertoire CSV.")
    parser.add_argument("--model-dir", default=str(REPO_ROOT / "models" / "map"))
    parser.add_argument("--aaindex", default=str(REPO_ROOT / "resources" / "AAidx_PCA.txt"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "bulk_prediction"))
    parser.add_argument("--cdr3-col", default="CDR3")
    parser.add_argument("--v-col", default="V")
    parser.add_argument("--j-col", default="J")
    parser.add_argument("--clone-id-col", default=None)
    parser.add_argument(
        "--abundance-col",
        default=None,
        help="Count or frequency column. If omitted, every input row receives abundance 1.",
    )
    parser.add_argument(
        "--model-glob",
        default="fold_*_map.pth",
        help="Model file pattern. The default ensembles all fold models.",
    )
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    validate_columns(df, [args.cdr3_col, args.v_col, args.j_col], args.input_csv)

    work_df = df.rename(
        columns={
            args.cdr3_col: "CDR3",
            args.v_col: "V",
            args.j_col: "J",
        }
    ).copy()

    if args.clone_id_col and args.clone_id_col in df.columns:
        clone_ids = df[args.clone_id_col].astype(str).values
    else:
        clone_ids = np.array([f"clone_{i + 1}" for i in range(len(df))])

    if args.abundance_col and args.abundance_col in df.columns:
        abundance = pd.to_numeric(df[args.abundance_col], errors="coerce").fillna(0).values
    else:
        abundance = np.ones(len(df), dtype=float)

    resources = load_resources(
        model_dir=args.model_dir,
        aaindex_path=args.aaindex,
        latent_dim=args.latent_dim,
        model_glob=args.model_glob,
    )
    work_df = add_vocab_indices(work_df, resources.vocab, require_labels=False)
    probs, latents, uncertainties = predict_dataframe(work_df, resources, device=args.device)

    class_names = class_names_from_vocab(resources.vocab)
    pred_idx = np.argmax(probs, axis=1)
    pred_labels = [class_names[i] for i in pred_idx]

    clone_out = pd.DataFrame(
        {
            "Clone_ID": clone_ids,
            "CDR3": work_df["CDR3"].values,
            "V": work_df["V"].values,
            "J": work_df["J"].values,
            "Abundance": abundance,
            "Predicted_Function": pred_labels,
            "Uncertainty": uncertainties,
        }
    )
    for i, name in enumerate(class_names):
        clone_out[f"Prob_{name}"] = probs[:, i]

    clone_out.to_csv(output_dir / "clone_function_probabilities.csv", index=False)
    np.savez(
        output_dir / "clone_latent_embeddings.npz",
        clone_ids=clone_ids,
        embeddings=latents.astype(np.float32),
    )

    composition = aggregate_composition(probs, abundance)
    composition.index = class_names
    composition_df = composition.rename("Abundance_weighted_probability").reset_index()
    composition_df = composition_df.rename(columns={"index": "Function"})
    composition_df.to_csv(output_dir / "repertoire_functional_composition.csv", index=False)

    print(f"Saved clone-level probabilities to {output_dir / 'clone_function_probabilities.csv'}")
    print(f"Saved repertoire-level composition to {output_dir / 'repertoire_functional_composition.csv'}")


if __name__ == "__main__":
    main()
