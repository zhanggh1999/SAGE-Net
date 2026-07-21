import csv
import json
import logging
import os
import os.path as osp

import numpy as np
import torch
from jarvis.core.atoms import Atoms
from jarvis.core.specie import get_node_attributes
from torch_geometric.data import Batch, Data, InMemoryDataset
from tqdm.auto import tqdm

from dataset.utils import radius_graph_pbc


def _row_to_sample(row, cif_dir):
    cif_path = osp.join(cif_dir, f"{row['Id']}.cif")
    if not osp.exists(cif_path) and row.get("File_Name"):
        cif_path = osp.join(cif_dir, row["File_Name"])
    if not osp.exists(cif_path):
        return None

    try:
        target = float(row["prop"])
    except (TypeError, ValueError):
        return None

    return {
        "jid": row.get("File_Name", str(row["Id"])).replace(".cif", ""),
        "csv_id": int(row["Id"]),
        "cif_path": cif_path,
        "target": target,
    }


def load_slme_splits(
    source_root,
    split_file="dft_3d_slme_densegnn_split.npz",
    description_file="description.csv",
):
    """Load split metadata from the local SAGE-DenseGNN JARVIS/CIF layout.

    The DenseGNN lite experiments use fixed ``*_densegnn_split.npz`` files with
    row indices. Older CartNet scripts used JSON files keyed by JARVIS IDs. This
    loader supports both so existing runs remain usable while new lite-style runs
    share the DenseGNN fixed splits.
    """
    description_path = osp.join(source_root, description_file)
    split_path = osp.join(source_root, split_file)
    cif_dir = osp.join(source_root, "cif")

    with open(description_path, newline="") as f:
        rows = list(csv.DictReader(f))

    skipped_missing = 0
    skipped_duplicate = 0
    splits = {}

    if split_file.endswith(".npz"):
        split_data = np.load(split_path)
        for split_name in ["train", "val", "test"]:
            samples = []
            for row_idx in split_data[split_name]:
                idx = int(row_idx)
                if idx < 0 or idx >= len(rows):
                    skipped_missing += 1
                    continue
                sample = _row_to_sample(rows[idx], cif_dir)
                if sample is None:
                    skipped_missing += 1
                    continue
                samples.append(sample)
            splits[split_name] = samples
    else:
        with open(split_path) as f:
            split_data = json.load(f)

        rows_by_jid = {
            row["File_Name"].replace(".cif", ""): row
            for row in rows
            if row.get("File_Name")
        }

        seen = set()
        for split_name in ["train", "val", "test"]:
            samples = []
            for jid in split_data.get(split_name, {}):
                if jid in seen:
                    skipped_duplicate += 1
                    continue
                seen.add(jid)

                row = rows_by_jid.get(jid)
                if row is None:
                    skipped_missing += 1
                    continue

                sample = _row_to_sample(row, cif_dir)
                if sample is None:
                    skipped_missing += 1
                    continue
                samples.append(sample)
            splits[split_name] = samples

    if skipped_missing:
        logging.warning("Skipped %s entries missing CSV/CIF/target data.", skipped_missing)
    if skipped_duplicate:
        logging.warning("Skipped %s duplicate entries across splits.", skipped_duplicate)

    return splits

class SLMEDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        samples,
        split_name,
        dataset_name="slme",
        radius=5.0,
        max_neigh=-1,
        source_root=None,
        split_file=None,
        use_text=False,
        text_embedding_file="text_embeddings.npy",
        description_file="description.csv",
        transform=None,
        pre_transform=None,
    ):
        self.samples = samples
        self.split_name = split_name
        self.dataset_name = dataset_name
        self.radius = radius
        self.max_neigh = max_neigh if max_neigh > 0 else None
        self.source_root = source_root if source_root is not None else osp.dirname(osp.abspath(root))
        self.split_file = split_file
        self.use_text = use_text
        self.text_embedding_file = text_embedding_file
        self.description_file = description_file
        self.text_embeddings = None
        if self.use_text:
            text_embedding_path = osp.join(self.source_root, self.text_embedding_file)
            if not osp.exists(text_embedding_path):
                raise FileNotFoundError(f"Text embedding file not found: {text_embedding_path}")
            self.text_embeddings = np.load(text_embedding_path, mmap_mode="r")
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        radius = str(self.radius).replace(".", "p")
        max_neigh = "all" if self.max_neigh is None else str(self.max_neigh)
        text_suffix = "_text" if self.use_text else ""
        split_suffix = ""
        if self.split_file:
            split_suffix = "_" + osp.splitext(osp.basename(self.split_file))[0]
        description_suffix = ""
        if self.description_file and self.description_file != "description.csv":
            description_suffix = "_" + osp.splitext(osp.basename(self.description_file))[0]
        embedding_suffix = ""
        if self.use_text and self.text_embedding_file != "text_embeddings.npy":
            embedding_suffix = "_" + osp.splitext(osp.basename(self.text_embedding_file))[0]
        return f"{self.dataset_name}_{self.split_name}{split_suffix}{description_suffix}{text_suffix}{embedding_suffix}_r{radius}_n{max_neigh}.pt"

    def download(self):
        pass

    def process(self):
        data_list = []
        for sample in tqdm(self.samples, total=len(self.samples), desc=f"{self.dataset_name} {self.split_name}"):
            atoms = Atoms.from_cif(sample["cif_path"], use_cif2cell=False)
            atomic_numbers = torch.tensor(
                [get_node_attributes(s, atom_features="atomic_number") for s in atoms.elements],
                dtype=torch.long,
            ).squeeze(-1)

            data = Data(x=atomic_numbers, y=torch.tensor(sample["target"], dtype=torch.float32))
            data.pos = torch.tensor(atoms.cart_coords, dtype=torch.float32)
            data.cell = torch.tensor(atoms.lattice_mat, dtype=torch.float32).unsqueeze(0)
            data.pbc = torch.tensor([[True, True, True]])
            data.natoms = torch.tensor([data.x.shape[0]])
            if self.use_text:
                csv_id = sample["csv_id"]
                if csv_id >= self.text_embeddings.shape[0]:
                    raise IndexError(
                        f"CSV id {csv_id} is out of bounds for text embeddings "
                        f"with shape {self.text_embeddings.shape}."
                    )
                text_embedding = np.asarray(self.text_embeddings[csv_id], dtype=np.float32)
                data.text_embedding = torch.from_numpy(text_embedding.copy()).unsqueeze(0)

            batch = Batch.from_data_list([data])
            edge_index, _, _, cart_vector = radius_graph_pbc(batch, self.radius, self.max_neigh)
            data.cart_dist = torch.norm(cart_vector, p=2, dim=-1)
            data.cart_dir = torch.nn.functional.normalize(cart_vector, p=2, dim=-1)
            data.edge_index = edge_index

            delattr(data, "pbc")
            data_list.append(data)

        data, slices = self.collate(data_list)
        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save((data, slices), self.processed_paths[0])
