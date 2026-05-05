#!/usr/bin/env python
"""Convert scMAC-LLM cell embeddings into the NPZ format expected by scMAC-MAP."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a scMAC-LLM embedding NPZ to a scMAC-MAP reference NPZ."
    )
    parser.add_argument("--input-npz", required=True, help="NPZ with embeddings/cell_names or matrix/barcodes.")
    parser.add_argument("--output-npz", required=True, help="Output NPZ with matrix and barcodes keys.")
    return parser.parse_args()


def main():
    args = parse_args()
    data = np.load(args.input_npz, allow_pickle=True)
    if "matrix" in data and "barcodes" in data:
        matrix = data["matrix"]
        barcodes = data["barcodes"]
    elif "embeddings" in data and "cell_names" in data:
        matrix = data["embeddings"]
        barcodes = data["cell_names"]
    else:
        raise ValueError(
            "Input NPZ must contain either matrix/barcodes or embeddings/cell_names."
        )

    output_path = Path(args.output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, matrix=matrix.astype(np.float32), barcodes=barcodes)
    print(f"Saved scMAC-MAP reference NPZ to {output_path}")


if __name__ == "__main__":
    main()
