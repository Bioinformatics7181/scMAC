# scMAC

scMAC is a two-module framework for functional context completion of bulk T cell
receptor (TCR) repertoires.

- **scMAC-LLM** builds a denoised single-cell functional reference from
  scRNA-seq expression profiles and offline LLM-derived gene semantic
  embeddings.
- **scMAC-MAP** maps TCR beta chain (TCRbeta) CDR3 sequences and V/J gene usage
  into the learned functional space and returns clone-level functional
  probabilities. Repertoire-level functional composition is computed by
  abundance-weighted aggregation of clone-level probabilities.

This repository is the minimal runnable release for the scMAC manuscript. It
contains the model code, an example processed dataset, a pretrained scMAC-MAP
model and scripts for direct bulk inference and retraining.

## Repository Layout

```text
data/examples/map/          Processed paired and bulk-like example inputs
models/map/                 Pretrained scMAC-MAP model and runtime vocabulary
resources/AAidx_PCA.txt     Amino-acid feature table used by the TCR encoder
scripts/                    Command-line entry points
src/scmac/llm/              scMAC-LLM model and training wrapper
src/scmac/map/              scMAC-MAP model and pipeline utilities
```

## Installation

Create a Python environment with Python 3.10 or later.

```bash
pip install -r requirements.txt
pip install -e .
```

Check the active environment:

```bash
python scripts/check_environment.py
```

## Input Formats

### Bulk TCR repertoire input

The direct inference script expects a CSV file with:

```text
CDR3,V,J
```

Optional columns:

```text
Clone_ID,Count,Frequency
```

If no abundance column is supplied, each row is assigned abundance 1.

### Paired scRNA/TCR-seq input for MAP retraining

The paired label CSV should contain:

```text
Barcodes,CDR3,V,J,Function
```

The scMAC-LLM reference NPZ should contain:

```text
matrix      # cells x latent_dim single-cell functional embeddings
barcodes    # cell barcodes matching the paired label CSV
```

## Workflow 1: Direct Bulk TCR Inference

Use this workflow when processed bulk TCR records and a trained scMAC-MAP model
are available.

```bash
python scripts/predict_bulk.py \
  --input-csv data/examples/map/bulk_tcr_example.csv \
  --model-dir models/map \
  --model-glob final_model_map.pth \
  --aaindex resources/AAidx_PCA.txt \
  --abundance-col Count \
  --output-dir outputs/bulk_prediction
```

Outputs:

```text
outputs/bulk_prediction/clone_function_probabilities.csv
outputs/bulk_prediction/clone_latent_embeddings.npz
outputs/bulk_prediction/repertoire_functional_composition.csv
```

`clone_function_probabilities.csv` is the primary functional context completion
output. The repertoire composition file is a downstream abundance-weighted
summary.

## Workflow 2: Create A Bulk-Like Example

The bundled paired example can be collapsed into a bulk-like TCR input:

```bash
python scripts/make_bulk_example.py \
  --paired-csv data/examples/map/paired_tcr_labels.csv \
  --output-csv data/examples/map/bulk_tcr_example.csv
```

## Workflow 3: Retrain scMAC-LLM Reference Embeddings

Train scMAC-LLM from an AnnData file and offline gene semantic embeddings:

```bash
python scripts/train_llm_reference.py \
  --h5ad path/to/single_cell_data.h5ad \
  --gene-embedding path/to/gene_semantic_embeddings.pkl \
  --data-name CRC \
  --output-dir outputs/llm_reference \
  --n-genes 5000 \
  --epochs 20
```

Convert the exported cell embeddings to the MAP reference format:

```bash
python scripts/prepare_reference_npz.py \
  --input-npz outputs/llm_reference/CRC_scMACLLM_cell_embed.npz \
  --output-npz outputs/llm_reference/CRC_scmac_map_reference.npz
```

## Workflow 4: Retrain scMAC-MAP

Train the TCR-to-function mapper from paired labels and scMAC-LLM reference
embeddings:

```bash
python scripts/train_map.py \
  --model-type map \
  --paired-csv data/examples/map/paired_tcr_labels.csv \
  --reference-npz data/examples/map/scmac_llm_reference.npz \
  --aaindex resources/AAidx_PCA.txt \
  --output-dir outputs/map_training
```

The training script uses the same five-fold cross-validation engine as the
manuscript experiments. The output directory contains fold models,
configuration files, vocabulary files and prediction summaries. For public
deployment, either use the fold model selected by your validation criterion or
retrain a final model using your preferred training split.

## Bundled Pretrained Model

The bundled model is:

```text
models/map/final_model_map.pth
```

It was selected from the original five-fold scMAC-MAP training run because the
corresponding fold had the strongest macro F1-score among the five validation
folds. The original manuscript-level composition and clone-level metrics were
computed from all five cross-validation folds, not from this single deployment
model.

See `models/map/model_metadata.json` for details.

## Notes

- Unknown V/J genes in new bulk repertoires are mapped to index 0.
- The functional label space contains seven T cell states:
  `CD4_Helper`, `CD8_Effector`, `T_Exhausted`, `T_Innate_like`,
  `T_Naive_Memory`, `T_Proliferating` and `T_Regulatory`.
- The repository does not include the full benchmark scripts for external
  baselines. It focuses on the runnable scMAC workflows needed for inference
  and retraining.
