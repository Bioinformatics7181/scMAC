#!/usr/bin/env python
"""Train scMAC-MAP or a supported TCR baseline from paired scRNA/TCR-seq data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MAP_SRC_DIR = REPO_ROOT / "src" / "scmac" / "map"
if str(MAP_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(MAP_SRC_DIR))

import mapping_unified  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Train the scMAC-MAP mapping module.")
    parser.add_argument("--model-type", default="map", choices=["map", "cgcl", "tcrd", "mist"])
    parser.add_argument(
        "--paired-csv",
        default=str(REPO_ROOT / "data" / "examples" / "map" / "paired_tcr_labels.csv"),
    )
    parser.add_argument(
        "--reference-npz",
        default=str(REPO_ROOT / "data" / "examples" / "map" / "scmac_llm_reference.npz"),
    )
    parser.add_argument("--aaindex", default=str(REPO_ROOT / "resources" / "AAidx_PCA.txt"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "map_training"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use-v", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-j", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--training-mode", action="store_true")
    parser.add_argument("--unknown-split", action="store_true")
    parser.add_argument("--disable-rejection", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.epochs is not None:
        mapping_unified.EPOCHS = args.epochs
    if args.batch_size is not None:
        mapping_unified.BATCH_SIZE = args.batch_size
    if args.learning_rate is not None:
        mapping_unified.LR = args.learning_rate
    if args.seed is not None:
        mapping_unified.SEED = args.seed
    if args.use_v is not None:
        mapping_unified.USE_V = args.use_v
    if args.use_j is not None:
        mapping_unified.USE_J = args.use_j
    mapping_unified.IS_TRAINING_MODE = args.training_mode
    mapping_unified.IS_UNKNOWN = args.unknown_split
    if args.disable_rejection:
        mapping_unified.ENABLE_REJECTION = False

    mapping_unified.run_unified_cv(
        model_type=args.model_type,
        csv_file=args.paired_csv,
        npz_file=args.reference_npz,
        output_dir=args.output_dir,
        aaindex_file=args.aaindex,
    )


if __name__ == "__main__":
    main()
