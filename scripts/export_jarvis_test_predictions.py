#!/usr/bin/env python3
"""Export per-sample test predictions from a fixed-split JARVIS CartNet run."""
import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch_geometric.graphgym.config import cfg, set_cfg
from torch_geometric.loader import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from loader.loader import create_loader
from models.master import create_model


LOCAL_JARVIS_SPLIT_FILES = {
    "avg": "dft_3d_avg_hole_mass_densegnn_split.npz",
    "slme": "dft_3d_slme_densegnn_split.npz",
    "bulk_new": "dft_3d_bulk_modulus_kv_densegnn_split.npz",
    "max": "dft_3d_max_efg_densegnn_split.npz",
    "mbj": "dft_3d_mbj_bandgap_densegnn_split.npz",
    "mepsz": "dft_3d_mepsz_densegnn_split.npz",
    "optb88": "dft_3d_optb88vdw_bandgap_densegnn_split.npz",
    "seebeck": "dft_3d_n_Seebeck_densegnn_split.npz",
    "shear": "dft_3d_shear_modulus_gv_densegnn_split.npz",
    "spillage": "dft_3d_spillage_densegnn_split.npz",
}

LOCAL_JARVIS_LEGACY_SPLIT_FILES = {
    "avg": "dft_3d_avg_hole_mass.json",
    "slme": "dft_3d_slme.json",
    "bulk_new": "dft_3d_bulk_modulus_kv.json",
    "max": "dft_3d_max_efg.json",
    "mbj": "dft_3d_mbj_bandgap.json",
    "mepsz": "dft_3d_mepsz.json",
    "optb88": "dft_3d_optb88vdw_bandgap.json",
    "seebeck": "dft_3d_n_Seebeck.json",
    "shear": "dft_3d_shear_modulus_gv.json",
    "spillage": "dft_3d_spillage.json",
}


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in ("yes", "true", "t", "1", "y"):
        return True
    if value in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {value}.")


def read_last_json(path: Path) -> dict:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"No JSON records found in {path}")
    return json.loads(lines[-1])


def load_description_rows(dataset_path: Path, description_file: str) -> dict[int, dict[str, str]]:
    rows = {}
    with (dataset_path / description_file).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[int(row["Id"])] = row
    return rows


def configure(args) -> None:
    set_cfg(cfg)
    cfg.name = args.run_dir.parent.name
    cfg.run_dir = str(args.run_dir)
    cfg.dataset.task_type = "regression"
    cfg.dataset.name = args.dataset
    cfg.dataset_path = str(args.dataset_path)
    cfg.batch = args.batch
    cfg.workers = args.workers
    cfg.device = args.device
    cfg.model = "CartNet"
    cfg.max_neighbours = args.max_neighbours
    cfg.radius = args.radius
    cfg.num_layers = args.num_layers
    cfg.dim_in = args.dim_in
    cfg.dim_rbf = args.dim_rbf
    cfg.invariant = False
    cfg.use_temp = False
    cfg.envelope = True
    cfg.use_H = True
    cfg.use_atom_types = True

    cfg.use_text = args.use_text
    cfg.description_file = args.description_file
    cfg.text_embedding_file = args.text_embedding_file
    cfg.use_late_fusion = args.use_late_fusion
    cfg.text_embedding_dim = args.text_embedding_dim
    cfg.text_projection_dim = args.text_projection_dim
    cfg.late_fusion_type = args.late_fusion_type
    cfg.late_fusion_output_dim = args.late_fusion_output_dim
    cfg.fusion_dropout = args.fusion_dropout
    cfg.use_middle_fusion = args.use_middle_fusion
    cfg.middle_fusion_type = args.middle_fusion_type
    cfg.middle_fusion_layers = args.middle_fusion_layers
    cfg.middle_fusion_hidden_dim = args.middle_fusion_hidden_dim
    cfg.middle_fusion_num_heads = args.middle_fusion_num_heads
    cfg.middle_fusion_dropout = args.middle_fusion_dropout
    cfg.middle_fusion_use_gate_norm = args.middle_fusion_use_gate_norm
    cfg.middle_fusion_use_learnable_scale = args.middle_fusion_use_learnable_scale
    cfg.middle_fusion_initial_scale = args.middle_fusion_initial_scale
    cfg.middle_fusion_gate_bias = args.middle_fusion_gate_bias
    cfg.middle_fusion_correction_scale = args.middle_fusion_correction_scale
    cfg.text_sample_dropout = args.text_sample_dropout
    cfg.contrastive_weight = args.contrastive_weight
    cfg.contrastive_temperature = args.contrastive_temperature
    cfg.contrastive_projection_dim = args.contrastive_projection_dim


def create_test_loader(args):
    if args.dataset not in LOCAL_JARVIS_SPLIT_FILES:
        return create_loader()[-1]

    from dataset.slme_dataset import SLMEDataset, load_slme_splits

    split_file = LOCAL_JARVIS_SPLIT_FILES[args.dataset]
    if not (args.dataset_path / split_file).exists():
        split_file = LOCAL_JARVIS_LEGACY_SPLIT_FILES[args.dataset]
    splits = load_slme_splits(
        str(args.dataset_path),
        split_file,
        description_file=args.description_file,
    )
    dataset_test = SLMEDataset(
        root=str(args.dataset_path / "cartnet_cache"),
        samples=splits["test"],
        split_name="test",
        dataset_name=args.dataset,
        radius=args.radius,
        max_neigh=args.max_neighbours,
        source_root=str(args.dataset_path),
        split_file=split_file,
        use_text=args.use_text,
        description_file=args.description_file,
        text_embedding_file=args.text_embedding_file,
    )
    return DataLoader(
        dataset_test,
        batch_size=args.batch,
        persistent_workers=False,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="avg")
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--radius", type=float, default=5.0)
    parser.add_argument("--max_neighbours", type=int, default=-1)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dim_in", type=int, default=256)
    parser.add_argument("--dim_rbf", type=int, default=64)

    parser.add_argument("--use_text", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--description_file", default="description.csv")
    parser.add_argument("--text_embedding_file", default="text_embeddings.npy")
    parser.add_argument("--use_late_fusion", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--text_embedding_dim", type=int, default=768)
    parser.add_argument("--text_projection_dim", type=int, default=128)
    parser.add_argument("--late_fusion_type", default="gated", choices=["concat", "gated"])
    parser.add_argument("--late_fusion_output_dim", type=int, default=128)
    parser.add_argument("--fusion_dropout", type=float, default=0.1)
    parser.add_argument("--use_middle_fusion", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--middle_fusion_type", default="residual", choices=["residual", "conditioned_update"])
    parser.add_argument("--middle_fusion_layers", default="2")
    parser.add_argument("--middle_fusion_hidden_dim", type=int, default=256)
    parser.add_argument("--middle_fusion_num_heads", type=int, default=2)
    parser.add_argument("--middle_fusion_dropout", type=float, default=0.1)
    parser.add_argument("--middle_fusion_use_gate_norm", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--middle_fusion_use_learnable_scale", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--middle_fusion_initial_scale", type=float, default=1.0)
    parser.add_argument("--middle_fusion_gate_bias", type=float, default=-3.0)
    parser.add_argument("--middle_fusion_correction_scale", type=float, default=0.1)
    parser.add_argument("--text_sample_dropout", type=float, default=0.20)
    parser.add_argument("--contrastive_weight", type=float, default=0.03)
    parser.add_argument("--contrastive_temperature", type=float, default=0.10)
    parser.add_argument("--contrastive_projection_dim", type=int, default=128)
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if args.checkpoint is None:
        args.checkpoint = args.run_dir / "ckpt" / "best.ckpt"
    if args.output is None:
        args.output = args.run_dir / "test" / "test_predictions.csv"

    configure(args)

    torch.set_num_threads(8)
    test_loader = create_test_loader(args)
    test_samples = list(test_loader.dataset.samples)
    description_rows = load_description_rows(args.dataset_path, args.description_file)

    model = create_model()
    checkpoint = torch.load(args.checkpoint, map_location=torch.device(args.device), weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(args.device)
    model.eval()

    rows = []
    offset = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(args.device)
            pred, true = model(batch)
            pred_cpu = pred.detach().cpu().view(-1)
            true_cpu = true.detach().cpu().view(-1)
            batch_size = int(pred_cpu.numel())
            samples = test_samples[offset : offset + batch_size]
            if len(samples) != batch_size:
                raise RuntimeError(
                    f"Sample alignment failed at offset {offset}: "
                    f"got {len(samples)} samples for batch size {batch_size}."
                )

            for local_idx, (sample, pred_value, true_value) in enumerate(zip(samples, pred_cpu, true_cpu)):
                csv_id = int(sample["csv_id"])
                desc_row = description_rows.get(csv_id, {})
                pred_float = float(pred_value.item())
                true_float = float(true_value.item())
                abs_error = abs(pred_float - true_float)
                rows.append(
                    {
                        "test_order": offset + local_idx,
                        "csv_id": csv_id,
                        "jid": sample.get("jid", ""),
                        "file_name": desc_row.get("File_Name", Path(sample.get("cif_path", "")).name),
                        "composition": desc_row.get("Composition", ""),
                        "target": true_float,
                        "prediction": pred_float,
                        "abs_error": abs_error,
                        "squared_error": (pred_float - true_float) ** 2,
                    }
                )
            offset += batch_size

    if offset != len(test_samples):
        raise RuntimeError(f"Exported {offset} predictions but test split has {len(test_samples)} samples.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        fieldnames = [
            "test_order",
            "csv_id",
            "jid",
            "file_name",
            "composition",
            "target",
            "prediction",
            "abs_error",
            "squared_error",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    mae = sum(row["abs_error"] for row in rows) / len(rows)
    mse = sum(row["squared_error"] for row in rows) / len(rows)
    stats_path = args.run_dir / "test" / "stats.json"
    stats = read_last_json(stats_path) if stats_path.exists() else {}
    print(f"wrote {args.output}")
    print(f"n={len(rows)}")
    print(f"per_sample_MAE={mae}")
    print(f"per_sample_MSE={mse}")
    if stats:
        print(f"stats_MAE={stats.get('MAE')}")
        print(f"stats_MSE={stats.get('MSE')}")


if __name__ == "__main__":
    main()
