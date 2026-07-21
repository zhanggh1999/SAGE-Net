# MBJ Multimodal Fixed-Split Result Bundle

This directory contains the generated SAGE-CartNet multimodal fixed-split MBJ
band gap result bundle.

## Files

- `ckpt/best.ckpt`
  - Best validation checkpoint from the run.
- `logging.log`
  - Training log.
- `train/stats.json`
  - Final recorded train metrics.
- `val/stats.json`
  - Final recorded validation metrics.
- `test/stats.json`
  - Test metrics for the exported checkpoint.
- `test/test_predictions.csv`
  - Per-sample test predictions and targets.
- `test/mbj_with_text.png`
- `test/mbj_with_text.pdf`
- `test/mbj_no_text.png`
- `test/mbj_no_text.pdf`
  - Predicted-vs-actual plots exported from `test/test_predictions.csv`.

The matching dataset is stored in `datasets/jarvis/mbj`.
