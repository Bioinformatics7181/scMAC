#!/usr/bin/env python
"""Train scMAC-LLM and export single-cell functional reference embeddings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import scanpy as sc

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scmac.llm import scMACLLMTrainer  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train scMAC-LLM from an AnnData file and optional gene semantic embeddings."
    )
    parser.add_argument("--h5ad", required=True, help="Input single-cell AnnData file.")
    parser.add_argument("--gene-embedding", default=None, help="Pickle file of gene semantic embeddings.")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "llm_reference"))
    parser.add_argument("--data-name", default="dataset")
    parser.add_argument("--n-genes", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embed-weight", type=float, default=1.0)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--save-reconstruction", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    raw = sc.read_h5ad(args.h5ad)
    adata = scMACLLMTrainer.preprocess_h5ad(
        raw,
        normalize=args.normalize,
        n_genes=args.n_genes,
    )
    hvg_order = adata.var["dispersions_norm"].sort_values(ascending=False).index
    adata = adata[:, hvg_order]

    embedding = None
    if args.gene_embedding:
        embedding, message = scMACLLMTrainer.load_gene_embedding(adata, args.gene_embedding)
        print(message)

    trainer = scMACLLMTrainer(
        adata=adata,
        embedding=embedding,
        batch_size=args.batch_size,
        train_epoch=args.epochs,
        embed_weight=args.embed_weight,
        device=device,
    )
    trainer.train()
    results = trainer.evaluate_all_outputs()

    model_name = "scMACLLM" if results["gene_embed"] is not None else "scMAC"
    torch.save(trainer.AE.state_dict(), output_dir / f"{args.data_name}_{model_name}.pt")
    np.savez(
        output_dir / f"{args.data_name}_{model_name}_cell_embed.npz",
        embeddings=results["cell_embed"].astype(np.float32),
        cell_names=results["cell_names"],
    )
    if results["gene_embed"] is not None:
        np.savez(
            output_dir / f"{args.data_name}_{model_name}_gene_embed.npz",
            embeddings=results["gene_embed"].astype(np.float32),
            gene_names=results["gene_names"],
        )
    np.savez(
        output_dir / f"{args.data_name}_{model_name}_conv_attn.npz",
        attention=results["conv_attn"].astype(np.float32),
        cell_names=results["cell_names"],
    )
    pd.Series(trainer.get_train_loss_curve(), name="loss").to_csv(
        output_dir / f"{args.data_name}_{model_name}_loss.csv",
        index=False,
    )
    if args.save_reconstruction:
        pd.DataFrame(
            results["recon_expr"].astype(np.float32),
            index=results["cell_names"],
            columns=results["gene_names"],
        ).to_csv(output_dir / f"{args.data_name}_{model_name}_recon_expr.csv")

    print(f"Saved scMAC-LLM outputs to {output_dir}")


if __name__ == "__main__":
    main()
