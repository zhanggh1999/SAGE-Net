# Copyright Universitat Politècnica de Catalunya 2024 https://imatge.upc.edu
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

import torch
from torch_geometric.graphgym.config import cfg
from torch_geometric.loader import DataLoader
import random
import os.path as osp
import numpy as np

def create_loader():
    """
    Create data loader object

    Returns: List of PyTorch data loaders

    """
    if cfg.dataset.name == "ADP":
        from dataset.datasetADP import DatasetADP
        from dataset.utils import compute_knn

        refcodes = ["dataset/csv/train_files.csv", "dataset/csv/val_files.csv", "dataset/csv/test_files.csv"]
        if cfg.model in ["icomformer", "ecomformer"]:
            assert cfg.max_neighbours is not None, "max_neighbours are needed for e/iComformer"
            cfg.dataset_path = compute_knn(cfg.max_neighbours, cfg.radius, cfg.dataset_path, refcodes)

        optimize_cell = True if cfg.model == "icomformer" else False
        dataset_train, dataset_val, dataset_test = (DatasetADP(root=osp.join(cfg.dataset_path, "data/"), file_names=refcodes[0], hydrogens=cfg.use_H, standarize_temp = cfg.standarize_temp, augment=cfg.augment, optimize_cell=optimize_cell),
                                                    DatasetADP(root=osp.join(cfg.dataset_path, "data/"), file_names=refcodes[1], hydrogens=cfg.use_H, standarize_temp = cfg.standarize_temp, optimize_cell=optimize_cell),
                                                    DatasetADP(root=osp.join(cfg.dataset_path, "data/"), file_names=refcodes[2], hydrogens=cfg.use_H, standarize_temp = cfg.standarize_temp, optimize_cell=optimize_cell) 
                                                )
    elif cfg.dataset.name == "jarvis" or cfg.dataset.name=="megnet":
        from jarvis.db.figshare import data as jdata
        from dataset.figshare_dataset import Figshare_Dataset
        import math
        import pandas as pd


        if cfg.dataset.name == "jarvis":
            cfg.dataset.name = "dft_3d_2021"

        seed = 123 #PotNet uses seed=123 for the comparative table
        target = cfg.figshare_target
        if cfg.figshare_target in ["shear modulus", "bulk modulus"] and cfg.dataset.name == "megnet":
            import pickle as pk
            target = cfg.figshare_target
            if cfg.figshare_target == "bulk modulus":
                try:
                    data_train = pk.load(open("./dataset/megnet/bulk_megnet_train.pkl", "rb"))
                    data_val = pk.load(open("./dataset/megnet/bulk_megnet_val.pkl", "rb"))
                    data_test = pk.load(open("./dataset/megnet/bulk_megnet_test.pkl", "rb"))
                except:
                    raise Exception("Bulk modulus dataset not found, please download it from https://figshare.com/projects/Bulk_and_shear_datasets/165430")
            elif cfg.figshare_target == "shear modulus":
                try:
                    data_train = pk.load(open("./dataset/megnet/shear_megnet_train.pkl", "rb"))
                    data_val = pk.load(open("./dataset/megnet/shear_megnet_val.pkl", "rb"))
                    data_test = pk.load(open("./dataset/megnet/shear_megnet_test.pkl", "rb"))
                except:
                    raise Exception("Shear modulus dataset not found, please download it from https://figshare.com/projects/Bulk_and_shear_datasets/165430")
            
            targets_train = []
            dat_train = []
            targets_val = []
            dat_val = []
            targets_test = []
            dat_test = []
            for split, datalist, targets in zip([data_train, data_val, data_test], 
                                            [dat_train, dat_val, dat_test],
                                            [targets_train, targets_val, targets_test]):
                for i in split:
                    if (
                        i[target] is not None
                        and i[target] != "na"
                        and not math.isnan(i[target])
                    ):
                        datalist.append(i)
                        targets.append(i[target])
            
        else:
            data = jdata(cfg.dataset.name)
            dat = []
            all_targets = []
            for i in data:
                if isinstance(i[target], list):
                    all_targets.append(torch.tensor(i[target]))
                    dat.append(i)

                elif (
                        i[target] is not None
                        and i[target] != "na"
                        and not math.isnan(i[target])
                ):
                    dat.append(i)
                    all_targets.append(i[target])
            
            ids_train, ids_val, ids_test = create_train_val_test(dat, seed=seed) 
            dat_train = [dat[i] for i in ids_train]
            dat_val = [dat[i] for i in ids_val]
            dat_test = [dat[i] for i in ids_test]
            targets_train = [all_targets[i] for i in ids_train]
            targets_val = [all_targets[i] for i in ids_val]
            targets_test = [all_targets[i] for i in ids_test]
        
        radius = cfg.radius
        prefix = cfg.dataset.name+"_"+str(radius)+"_"+str(cfg.max_neighbours)+"_"+target+"_"+str(seed)
        dataset_train = Figshare_Dataset(root=cfg.dataset_path, data=dat_train, targets=targets_train, radius=radius, max_neigh=cfg.max_neighbours, name=prefix+"_train")
        dataset_val = Figshare_Dataset(root=cfg.dataset_path, data=dat_val, targets=targets_val, radius=radius, max_neigh=cfg.max_neighbours, name=prefix+"_val")
        dataset_test = Figshare_Dataset(root=cfg.dataset_path, data=dat_test, targets=targets_test, radius=radius, max_neigh=cfg.max_neighbours, name=prefix+"_test")
    elif cfg.dataset.name in [
        "avg",
        "bulk_new",
        "max",
        "mbj",
        "mepsz",
        "optb88",
        "seebeck",
        "shear",
        "slme",
        "spillage",
    ]:
        from dataset.slme_dataset import SLMEDataset, load_slme_splits

        densegnn_split_files = {
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
        legacy_json_split_files = {
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
        split_file = densegnn_split_files[cfg.dataset.name]
        if not osp.exists(osp.join(cfg.dataset_path, split_file)):
            split_file = legacy_json_split_files[cfg.dataset.name]
        description_file = getattr(cfg, "description_file", "description.csv")
        text_embedding_file = getattr(cfg, "text_embedding_file", "text_embeddings.npy")
        splits = load_slme_splits(
            cfg.dataset_path,
            split_file,
            description_file=description_file,
        )
        cache_root = osp.join(cfg.dataset_path, "cartnet_cache")
        use_text = getattr(cfg, "use_text", False)
        dataset_kwargs = {
            "root": cache_root,
            "dataset_name": cfg.dataset.name,
            "radius": cfg.radius,
            "max_neigh": cfg.max_neighbours,
            "source_root": cfg.dataset_path,
            "split_file": split_file,
            "use_text": use_text,
            "description_file": description_file,
            "text_embedding_file": text_embedding_file,
        }
        dataset_train = SLMEDataset(samples=splits["train"], split_name="train", **dataset_kwargs)
        dataset_val = SLMEDataset(samples=splits["val"], split_name="val", **dataset_kwargs)
        dataset_test = SLMEDataset(samples=splits["test"], split_name="test", **dataset_kwargs)
    else:
        raise Exception("Dataset not implemented")
    
    loaders = [
        DataLoader(dataset_train, batch_size=cfg.batch, persistent_workers=cfg.workers > 0,
                                  shuffle=True, num_workers=cfg.workers,
                                  pin_memory=True),
        DataLoader(dataset_val, batch_size=cfg.batch, persistent_workers=cfg.workers > 0,
                                    shuffle=False, num_workers=cfg.workers,
                                    pin_memory=True),
        DataLoader(dataset_test, batch_size=1 if cfg.dataset.name == "ADP" else cfg.batch, persistent_workers=False,
                                    shuffle=False, num_workers=cfg.workers,
                                    pin_memory=True)
    ]
    
    return loaders



def create_train_val_test(data, val_ratio=0.1, test_ratio=0.1, seed=123):
    ids = list(np.arange(len(data)))
    n = len(data)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    n_train = n - n_val - n_test
    random.seed(seed)
    random.shuffle(ids)
    ids_train = ids[:n_train]
    ids_val = ids[-(n_val + n_test): -n_test]
    ids_test = ids[-n_test:]
    return ids_train, ids_val, ids_test
