# SAGE-Net: SAGE-CartNet Reviewer Release

## Reviewer Release Notice

This repository is a partial release for manuscript review. It contains the
subset of code, dataset files, and trained model weights needed to run an
end-to-end prediction workflow and obtain results, using the `mbj_bandgap`
dataset as the example.

The overall framework is **SAGE-Net**. The files currently released in this
repository correspond to the **SAGE-CartNet** component, a multimodal
CartNet-based model for JARVIS MBJ band gap prediction.

After the manuscript is accepted, we will publicly release the complete
SAGE-CartNet, SAGE-CGCNN, SAGE-ALIGNN, and SAGE-DenseGNN-Lite codebases, full
datasets, and trained model weights.

This README is intentionally limited to the SAGE-CartNet MBJ fixed-split
reproduction workflow and avoids unrelated legacy experiments.

## Quick Start

The commands below assume the Conda environment is named `SAGE-CartNet`.

Inspect the bundled test metrics:

```bash
cat results/mbj/test/stats.json
```

Regenerate per-sample test predictions from the bundled checkpoint:

```bash
conda run -n SAGE-CartNet python scripts/export_jarvis_test_predictions.py \
  --run-dir results/mbj \
  --dataset mbj \
  --dataset-path datasets/jarvis/mbj
```

Regenerate the prediction plots:

```bash
conda run -n SAGE-CartNet python scripts/plot_jarvis_test_predictions.py \
  --predictions results/mbj/test/test_predictions.csv \
  --dataset mbj
```

Retrain the default MBJ multimodal run:

```bash
conda run --no-capture-output -n SAGE-CartNet \
  bash scripts/train_mbj_multimodal_fixed.sh
```

The training script defaults to `RUN_DIR=results/mbj`. To keep the bundled
result directory unchanged during a fresh run, override the output directory:

```bash
RUN_DIR=results/mbj_retrain conda run --no-capture-output -n SAGE-CartNet \
  bash scripts/train_mbj_multimodal_fixed.sh
```

## Repository Layout

| Path | Purpose |
|---|---|
| `main.py` | Training and evaluation entry point with multimodal arguments. |
| `models/cartnet.py` | CartNet backbone plus text projection, middle fusion, late fusion, and contrastive loss modules. |
| `models/master.py` | Model factory wiring command-line multimodal options into `CartNet`. |
| `dataset/slme_dataset.py` | Fixed-split JARVIS-style dataset loader with optional text embeddings. |
| `loader/loader.py` | Loader configuration for fixed local datasets. |
| `scripts/train_mbj_multimodal_fixed.sh` | Default MBJ multimodal fixed-split training script. |
| `scripts/export_jarvis_test_predictions.py` | Checkpoint inference and per-sample test CSV export. |
| `scripts/plot_jarvis_test_predictions.py` | Predicted-vs-actual plot generation. |
| `datasets/jarvis/mbj` | Bundled MBJ dataset, fixed split, CIF files, and text embeddings. |
| `results/mbj` | Bundled checkpoint, metrics, predictions, and plots. |

More detail about fixed-split prediction export and plotting is available in
[`docs/multimodal_fixed_jarvis_prediction.md`](docs/multimodal_fixed_jarvis_prediction.md).

## Environment

The experiments were run in the local environment `SAGE-CartNet`.

If needed, create an equivalent Conda environment from the provided dependency
files. `environment_2.yml` is the recommended Torch 2.4.0 variant, while
`environment.yml` keeps the Torch 1.13.1 variant:

```bash
conda env create -f environment_2.yml
conda activate SAGE-CartNet
```

This minimal environment is intended for the SAGE-CartNet MBJ release path.
Legacy ADP, CSD, and Monte Carlo utilities may require additional packages.

## Dataset

The MBJ reproduction dataset is stored in:

```text
datasets/jarvis/mbj
```

Dataset summary:

| Item | Value |
|---|---:|
| Total samples | `18162` |
| Train samples | `14530` |
| Validation samples | `1817` |
| Test samples | `1814` |
| Text embedding shape | `(18162, 768)` |
| Target | MBJ band gap |

Important files:

- `description.csv`
  - Per-sample metadata and target values.
- `text_embeddings.npy`
  - Precomputed text embeddings aligned with `description.csv` by row `Id`.
- `dft_3d_mbj_bandgap_densegnn_split.npz`
  - Fixed train/validation/test split.
- `dft_3d_mbj_bandgap.json`
  - Legacy split-compatible metadata file.
- `cif/`
  - CIF structures referenced by the dataset.

Generated graph caches are written to `datasets/jarvis/mbj/cartnet_cache/` and
are intentionally ignored by Git.

## Training

Run the default fixed-split multimodal training command:

```bash
conda run --no-capture-output -n SAGE-CartNet \
  bash scripts/train_mbj_multimodal_fixed.sh
```

Default settings:

| Setting | Value |
|---|---:|
| Dataset | `mbj` |
| Dataset path | `datasets/jarvis/mbj` |
| Epochs | `300` |
| Batch size | `128` |
| Learning rate | `0.001` |
| Output directory | `results/mbj` |
| Description file | `description.csv` |
| Text embeddings | `text_embeddings.npy` |
| Text projection dim | `128` |
| Late fusion | gated |
| Late fusion output dim | `128` |
| Middle fusion | residual |
| Middle fusion layer | `2` |
| Middle fusion hidden dim | `256` |
| Text sample dropout | `0.20` |
| Contrastive weight | `0.03` |
| Contrastive temperature | `0.10` |
| Contrastive projection dim | `128` |

The script sets `WANDB_MODE=offline` unless the variable is already defined.

## Bundled Result

The bundled result is stored in:

```text
results/mbj
```

Included artifacts:

| File | Description |
|---|---|
| `results/mbj/ckpt/best.ckpt` | Best validation checkpoint. |
| `results/mbj/logging.log` | Training log. |
| `results/mbj/train/stats.json` | Recorded train metrics. |
| `results/mbj/val/stats.json` | Recorded validation metrics. |
| `results/mbj/test/stats.json` | Test metrics for the exported checkpoint. |
| `results/mbj/test/test_predictions.csv` | Per-sample test targets, predictions, and errors. |
| `results/mbj/test/mbj_with_text.png` | Plot with title, axis labels, and tick labels. |
| `results/mbj/test/mbj_with_text.pdf` | PDF version of the labeled plot. |
| `results/mbj/test/mbj_no_text.png` | Plot without title, axis labels, or tick labels. |
| `results/mbj/test/mbj_no_text.pdf` | PDF version of the unlabeled plot. |

Bundled test metrics:

| Metric | Value |
|---|---:|
| MAE | `0.2335542745` |
| MSE | `0.3006642526` |
| R2 | `0.9406090975` |
| Spearman r | `0.9284445599` |

## Export Predictions

To regenerate `results/mbj/test/test_predictions.csv` from
`results/mbj/ckpt/best.ckpt`:

```bash
conda run -n SAGE-CartNet python scripts/export_jarvis_test_predictions.py \
  --run-dir results/mbj \
  --dataset mbj \
  --dataset-path datasets/jarvis/mbj
```

The exported CSV contains:

- `test_order`
- `csv_id`
- `jid`
- `file_name`
- `composition`
- `target`
- `prediction`
- `abs_error`
- `squared_error`

## Plot Predictions

To regenerate the predicted-vs-actual plots:

```bash
conda run -n SAGE-CartNet python scripts/plot_jarvis_test_predictions.py \
  --predictions results/mbj/test/test_predictions.csv \
  --dataset mbj
```

The plotting script writes both PNG and PDF files to
`results/mbj/test`. The labeled plot keeps English text and numeric tick labels;
the unlabeled plot removes text labels while preserving the plotted points,
ticks, and diagonal reference line.

## Reproducibility Notes

- The fixed split is stored in
  `datasets/jarvis/mbj/dft_3d_mbj_bandgap_densegnn_split.npz`.
- The bundled checkpoint is below the GitHub 100 MB single-file limit.
- Small numerical differences can occur across GPU models, CUDA versions, and
  PyTorch builds.

## License

See [`LICENSE`](LICENSE).
