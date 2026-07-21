# CartNet Multimodal Fixed-Split Prediction

This repository now includes a small set of scripts for running CartNet
multimodal inference on fixed JARVIS-style splits and for plotting the
resulting predicted-vs-actual scatter figures.

## What is included

- `main.py`
  - Adds multimodal CartNet arguments:
    - `--run_dir`
    - `--use_text`
    - `--description_file`
    - `--text_embedding_file`
    - `--use_late_fusion`
    - `--late_fusion_type`
    - `--use_middle_fusion`
    - `--middle_fusion_type`
    - `--middle_fusion_layers`
    - `--contrastive_weight`
  - These options are forwarded into the model and dataset loader.
- `dataset/slme_dataset.py`
  - Loads fixed split files from local CSV/NPZ/JARVIS layouts.
  - Supports text embeddings through a description CSV plus a `.npy` embedding file.
- `loader/loader.py`
  - Passes the fixed split and text configuration into the dataset object.
- `scripts/export_jarvis_test_predictions.py`
  - Restores a checkpoint.
  - Runs test-only inference.
  - Writes per-sample predictions to CSV.
- `scripts/plot_jarvis_test_predictions.py`
  - Reads an exported CSV.
  - Produces `with_text` and `no_text` scatter plots.
  - Matches the style of `kgcnn.utils.plots.plot_predict_true`.

## Expected dataset layout

For a fixed JARVIS-style dataset directory, the script expects:

```text
dataset_root/
  description.csv
  text_embeddings.npy
  cif/
  dft_3d_<task>_densegnn_split.npz
```

For no-stopword / no-local variants, use the matching files:

```text
dataset_root/
  description_noSWnoLOCAL.csv
  text_embeddings_noSWnoLOCAL.npy
```

## Example: export predictions

```bash
conda run -n SAGE-CartNet python scripts/export_jarvis_test_predictions.py \
  --run-dir /public/home/ghzhang/CartNet-main/results/cartnet_lite_multimodal_gate_late_cl003_textdrop020_avg_seed123_300ep/123 \
  --dataset avg \
  --dataset-path /public/home/ghzhang/SAGE-DenseGNN/DenseGNN-main/datasets/jarvis/avg
```

## Example: train MBJ multimodal fixed split

The MBJ band gap reproduction data is included under `datasets/jarvis/mbj`.
To reproduce the fixed-split multimodal run:

```bash
conda run --no-capture-output -n SAGE-CartNet \
  bash scripts/train_mbj_multimodal_fixed.sh
```

This writes the run to:

```text
results/mbj
```

For noSW / noLOCAL runs:

```bash
conda run -n SAGE-CartNet python scripts/export_jarvis_test_predictions.py \
  --run-dir /public/home/ghzhang/CartNet-main/results/cartnet_lite_multimodal_gate_late_cl003_textdrop020_noSWnoLOCAL_shear_seed123_500ep/123 \
  --dataset shear \
  --dataset-path /public/home/ghzhang/SAGE-DenseGNN/DenseGNN-main/datasets/jarvis/shear \
  --description_file description_noSWnoLOCAL.csv \
  --text_embedding_file text_embeddings_noSWnoLOCAL.npy
```

The export script writes:

- `test/test_predictions.csv`
- `test/stats.json` is read back for comparison

The CSV contains:

- `test_order`
- `csv_id`
- `jid`
- `file_name`
- `composition`
- `target`
- `prediction`
- `abs_error`
- `squared_error`

## Example: plot predictions

```bash
conda run -n SAGE-CartNet python scripts/plot_jarvis_test_predictions.py \
  --predictions /public/home/ghzhang/CartNet-main/results/cartnet_lite_multimodal_gate_late_cl003_textdrop020_avg_seed123_300ep/123/test/test_predictions.csv \
  --dataset avg
```

This produces:

- `avg_with_text.png`
- `avg_with_text.pdf`
- `avg_no_text.png`
- `avg_no_text.pdf`

## Plot style

The plotting script intentionally follows the structure of
`kgcnn.utils.plots.plot_predict_true`:

- `Predicted` on the x-axis
- `Actual` on the y-axis
- a red diagonal `y = x`
- a `with_text` version with title and axis labels
- a `no_text` version without labels or numeric tick labels

The MAE legend has been removed from the plotting script.

## Currently organized fixed-split JARVIS runs

The repository already contains exported checkpoints and results for:

- `avg`
- `bulk_new`
- `max`
- `mbj`
- `mepsz`
- `seebeck`
- `shear`
- `optb88`
- `slme`
- `spillage`

These runs can be re-exported and re-plotted without touching the training code.
