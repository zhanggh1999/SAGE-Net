# JARVIS MBJ Band Gap Fixed-Split Data

This directory contains the minimal MBJ band gap data needed to reproduce the
SAGE-CartNet multimodal fixed-split experiments in this repository.

## Files

- `description.csv`
  - Per-sample metadata and target values used by the local CartNet loader.
- `text_embeddings.npy`
  - Text embeddings indexed by `description.csv` row `Id`.
- `dft_3d_mbj_bandgap_densegnn_split.npz`
  - Fixed train/validation/test row-index split.
- `dft_3d_mbj_bandgap.json`
  - Legacy split-compatible metadata file.
- `cif/`
  - CIF structures referenced by `description.csv`.

## Usage

From the repository root, train the default MBJ multimodal CartNet fixed-split
run with:

```bash
conda run --no-capture-output -n SAGE-CartNet \
  bash scripts/train_mbj_multimodal_fixed.sh
```

The training script defaults to `300` epochs, `description.csv`,
`text_embeddings.npy`, this dataset directory, and output directory
`results/mbj`.

Export per-sample test predictions with:

```bash
conda run -n SAGE-CartNet python scripts/export_jarvis_test_predictions.py \
  --run-dir results/mbj \
  --dataset mbj \
  --dataset-path datasets/jarvis/mbj
```

Plot exported predictions with:

```bash
conda run -n SAGE-CartNet python scripts/plot_jarvis_test_predictions.py \
  --predictions results/mbj/test/test_predictions.csv \
  --dataset mbj
```

Generated caches are written under `datasets/jarvis/mbj/cartnet_cache/` and are
ignored by Git.
