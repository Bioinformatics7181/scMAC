# -------------------------------------------------------------------------
# Utility functions for scMAC-MAP.
#
# These functions preserve the data encoding and split logic used by the
# manuscript experiments while keeping the public code path small and explicit.
# -------------------------------------------------------------------------

import os
import random

import numpy as np
import torch
from sklearn.model_selection import KFold, StratifiedGroupKFold


def read_tsv(filename, inf_ind, skip_1st=False, file_encoding="utf8"):
    """Read selected columns from a tab-separated text file."""
    extract_inf = []
    with open(filename, "r", encoding=file_encoding) as tsv_f:
        if skip_1st:
            tsv_f.readline()
        line = tsv_f.readline()
        while line:
            line = line[:-1]
            line_list = line.split("\t")
            if len(line_list) <= max(inf_ind):
                line = tsv_f.readline()
                continue
            extract_inf.append([line_list[ind] for ind in inf_ind])
            line = tsv_f.readline()
    return extract_inf


def get_features(filename, f_num=15):
    """Load AAindex-PCA amino-acid features used by the TCR encoder."""
    f_list = read_tsv(filename, list(range(16)), True)
    f_dict = {}
    left_num = 0
    right_num = 0
    if f_num > 15:
        left_num = (f_num - 15) // 2
        right_num = f_num - 15 - left_num

    for row in f_list:
        f_dict[row[0]] = [0] * left_num
        f_dict[row[0]] += [float(x) for x in row[1:]]
        f_dict[row[0]] += [0] * right_num

    xs_map = np.zeros((len(f_dict), len(f_dict["A"])), dtype=np.float64)
    for i, aa in enumerate(sorted(f_dict.keys())):
        xs_map[i, :] = np.array(f_dict[aa], dtype=np.float64)

    std = np.std(xs_map, axis=0)
    std[std == 0] = 1.0
    for aa in sorted(f_dict.keys()):
        f_dict[aa] = list(np.array(f_dict[aa], dtype=np.float64) / std)

    f_dict["X"] = [0] * f_num
    return f_dict


def generate_input_for_training(sps, sp_lbs, feature_dict, ins_num=100, feature_num=15, max_len=24):
    """Generate the TCRD training tensor format from grouped TCR sequences."""
    xs, ys = [], []
    for i, sp in enumerate(sps):
        xs.append([[[0] * feature_num] * max_len] * ins_num)
        ys.append(sp_lbs[i])
        for j, tcr in enumerate(sp):
            tcr_seq = str(tcr[0])
            tcr_seq = (tcr_seq + "X" * max_len)[:max_len]
            xs[i][j] = [feature_dict[aa.upper()] for aa in tcr_seq]
    xs = np.array(xs)
    xs = xs.swapaxes(2, 3)
    ys = np.array(ys)
    return xs, ys


def generate_input_for_prediction(sp, feature_dict, ins_num=10000, feature_num=15, max_len=24):
    """Generate the TCRD prediction tensor format from grouped TCR sequences."""
    xs = [[[[0] * feature_num] * max_len] * ins_num]
    for i, tcr in enumerate(sp):
        tcr_seq = str(tcr[0])
        tcr_seq = (tcr_seq + "X" * max_len)[:max_len]
        xs[0][i] = [feature_dict[aa.upper()] for aa in tcr_seq]
    xs = np.array(xs)
    xs = xs.swapaxes(2, 3)
    return xs


def seed_torch(seed=2026):
    """Set Python, NumPy and PyTorch seeds for reproducible inference/evaluation."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_mapping_splits(df, n_splits=5, is_unknown=False, seed=2026):
    """Reproduce the fold split protocol used in the scMAC-MAP experiments."""
    df = df.reset_index(drop=True)
    if is_unknown:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(splitter.split(df, y=df["Label_idx"], groups=df["CDR3"]))
        for i, (train_idx, test_idx) in enumerate(splits):
            train_seqs = set(df.iloc[train_idx]["CDR3"])
            test_seqs = set(df.iloc[test_idx]["CDR3"])
            overlap = train_seqs & test_seqs
            if overlap:
                raise ValueError(f"Fold {i + 1} has {len(overlap)} overlapping CDR3 sequences.")
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(splitter.split(df))
    return splits
