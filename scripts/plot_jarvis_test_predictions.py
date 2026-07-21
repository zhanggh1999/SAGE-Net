#!/usr/bin/env python3
"""Plot predicted-vs-actual scatter figures from exported CartNet predictions."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_prediction_csv(path: Path, prediction_column: str, target_column: str):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    y_predict = np.array([float(row[prediction_column]) for row in rows], dtype=float)
    y_true = np.array([float(row[target_column]) for row in rows], dtype=float)
    return y_predict, y_true


def diagonal_values(y_true: np.ndarray, step: float):
    valid_true = y_true[~np.isnan(y_true)]
    if valid_true.size == 0:
        raise ValueError("No finite true values found for plotting.")
    lo = float(np.amin(valid_true))
    hi = float(np.amax(valid_true))
    if lo == hi:
        pad = max(abs(lo) * 0.05, step)
        lo -= pad
        hi += pad
    values = np.arange(lo, hi, step)
    if values.size < 2:
        values = np.array([lo, hi], dtype=float)
    return values


def plot_predict_true(
    y_predict: np.ndarray,
    y_true: np.ndarray,
    dataset_name: str,
    output_dir: Path,
    suffix: str,
    model_name: str,
    with_text: bool,
    diagonal_step: float,
):
    """Match kgcnn.utils.plots.plot_predict_true style, with optional no-text output."""
    if len(y_predict.shape) == 1:
        y_predict = np.expand_dims(y_predict, axis=-1)
    if len(y_true.shape) == 1:
        y_true = np.expand_dims(y_true, axis=-1)
    num_targets = y_true.shape[1]

    fig = plt.figure(figsize=[6.4, 4.8], dpi=300.0)
    for i in range(num_targets):
        plt.scatter(y_predict[:, i], y_true[:, i])

    diagonal = diagonal_values(y_true, diagonal_step)
    plt.plot(diagonal, diagonal, color="red")

    if with_text:
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Prediction of " + model_name + " for " + dataset_name)
    else:
        ax = plt.gca()
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
        ax.tick_params(labelbottom=False, labelleft=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        plt.savefig(output_dir / f"{dataset_name}_{suffix}.{ext}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True, help="CSV from export_jarvis_test_predictions.py")
    parser.add_argument("--dataset", required=True, help="Dataset name used in output filenames")
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to the CSV parent directory")
    parser.add_argument("--model-name", default="CartNet")
    parser.add_argument("--prediction-column", default="prediction")
    parser.add_argument("--target-column", default="target")
    parser.add_argument("--versions", choices=["both", "with_text", "no_text"], default="both")
    parser.add_argument("--diagonal-step", type=float, default=0.05)
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir is not None else args.predictions.parent
    y_predict, y_true = read_prediction_csv(args.predictions, args.prediction_column, args.target_column)

    if args.versions in ("both", "with_text"):
        plot_predict_true(
            y_predict,
            y_true,
            args.dataset,
            output_dir,
            "with_text",
            args.model_name,
            with_text=True,
            diagonal_step=args.diagonal_step,
        )
    if args.versions in ("both", "no_text"):
        plot_predict_true(
            y_predict,
            y_true,
            args.dataset,
            output_dir,
            "no_text",
            args.model_name,
            with_text=False,
            diagonal_step=args.diagonal_step,
        )

    print(f"wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
