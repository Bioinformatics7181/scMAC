#!/usr/bin/env python
"""Create a bulk-like TCR repertoire example from paired single-cell TCR labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collapse paired single-cell TCR labels into a bulk-like clonotype table."
    )
    parser.add_argument(
        "--paired-csv",
        default=str(REPO_ROOT / "data" / "examples" / "map" / "paired_tcr_labels.csv"),
    )
    parser.add_argument(
        "--output-csv",
        default=str(REPO_ROOT / "data" / "examples" / "map" / "bulk_tcr_example.csv"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.paired_csv)
    required = ["CDR3", "V", "J"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Input file is missing required columns: {missing}")

    out = (
        df.groupby(["CDR3", "V", "J"], dropna=False)
        .size()
        .reset_index(name="Count")
        .sort_values("Count", ascending=False)
        .reset_index(drop=True)
    )
    out.insert(0, "Clone_ID", [f"clone_{i + 1}" for i in range(len(out))])
    out["Frequency"] = out["Count"] / out["Count"].sum()

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Saved bulk-like example to {output_path}")


if __name__ == "__main__":
    main()
