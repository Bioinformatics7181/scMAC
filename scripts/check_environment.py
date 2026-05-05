#!/usr/bin/env python
"""Check whether the current Python environment can run the scMAC pipelines."""

from __future__ import annotations

import importlib.util


REQUIRED = [
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "torch",
]

OPTIONAL_SINGLE_CELL = [
    "scanpy",
    "anndata",
    "gseapy",
]


def check_modules(modules):
    return {name: importlib.util.find_spec(name) is not None for name in modules}


def main():
    required = check_modules(REQUIRED)
    optional = check_modules(OPTIONAL_SINGLE_CELL)
    print("Required dependencies:")
    for name, ok in required.items():
        print(f"  {name}: {'OK' if ok else 'MISSING'}")
    print("\nSingle-cell workflow dependencies:")
    for name, ok in optional.items():
        print(f"  {name}: {'OK' if ok else 'MISSING'}")

    missing = [name for name, ok in required.items() if not ok]
    if missing:
        raise SystemExit(
            "Missing required dependencies: "
            + ", ".join(missing)
            + "\nInstall them with: pip install -r requirements.txt"
        )


if __name__ == "__main__":
    main()
