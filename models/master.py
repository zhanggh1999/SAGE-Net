# Copyright Universitat Politècnica de Catalunya 2024 https://imatge.upc.edu
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

import torch
from torch_geometric.graphgym.config import cfg


def create_model():
    """
    Creates and returns a model based on the configuration specified in `cfg`.

    Returns:
    model: An instance of the specified model class, moved to the CUDA device.
    Raises:
    Exception: If the specified model in `cfg.model` is not implemented.
    Notes:
    - If `cfg.model` is "CartNet", it imports and initializes a `CartNet` model with parameters from `cfg`.
    - If `cfg.model` is "ecomformer", it imports and initializes an `eComformer` model, ensuring the dataset is "ADP".
    - If `cfg.model` is "icomformer", it imports and initializes an `iComformer` model, ensuring the dataset is "ADP".
    """
   
    if cfg.model == "CartNet":
        from models.cartnet import CartNet
        device = getattr(cfg, "device", "cuda:0")
        model = CartNet(dim_in=cfg.dim_in,
                        dim_rbf=cfg.dim_rbf, 
                        num_layers=cfg.num_layers, 
                        invariant=cfg.invariant, 
                        temperature=cfg.use_temp, 
                        use_envelope=cfg.envelope,
                        atom_types=cfg.use_atom_types,
                        cholesky=True if cfg.dataset.name == "ADP" else False,
                        use_text=getattr(cfg, "use_text", False),
                        use_late_fusion=getattr(cfg, "use_late_fusion", True),
                        text_embedding_dim=getattr(cfg, "text_embedding_dim", 768),
                        text_projection_dim=getattr(cfg, "text_projection_dim", 64),
                        late_fusion_type=getattr(cfg, "late_fusion_type", "gated"),
                        late_fusion_output_dim=getattr(cfg, "late_fusion_output_dim", 64),
                        fusion_dropout=getattr(cfg, "fusion_dropout", 0.1),
                        use_middle_fusion=getattr(cfg, "use_middle_fusion", False),
                        middle_fusion_type=getattr(cfg, "middle_fusion_type", "residual"),
                        middle_fusion_layers=getattr(cfg, "middle_fusion_layers", "2"),
                        middle_fusion_hidden_dim=getattr(cfg, "middle_fusion_hidden_dim", 128),
                        middle_fusion_num_heads=getattr(cfg, "middle_fusion_num_heads", 2),
                        middle_fusion_dropout=getattr(cfg, "middle_fusion_dropout", 0.1),
                        middle_fusion_use_gate_norm=getattr(cfg, "middle_fusion_use_gate_norm", False),
                        middle_fusion_use_learnable_scale=getattr(cfg, "middle_fusion_use_learnable_scale", False),
                        middle_fusion_initial_scale=getattr(cfg, "middle_fusion_initial_scale", 1.0),
                        middle_fusion_gate_bias=getattr(cfg, "middle_fusion_gate_bias", -3.0),
                        middle_fusion_correction_scale=getattr(cfg, "middle_fusion_correction_scale", 0.1),
                        text_sample_dropout=getattr(cfg, "text_sample_dropout", 0.0),
                        contrastive_weight=getattr(cfg, "contrastive_weight", 0.0),
                        contrastive_temperature=getattr(cfg, "contrastive_temperature", 0.1),
                        contrastive_projection_dim=getattr(cfg, "contrastive_projection_dim", 128)
                    ).to(device)
    
    elif cfg.model == "ecomformer":
        from models.comformer import eComformer
        assert cfg.dataset.name == "ADP", "eComformer only for ADP dataset"
        model = eComformer(dim_in=cfg.dim_in).to(getattr(cfg, "device", "cuda:0"))

    elif cfg.model == "icomformer":
        from models.comformer import iComformer
        assert cfg.dataset.name == "ADP", "iComformer only for ADP dataset"
        model = iComformer(dim_in=cfg.dim_in).to(getattr(cfg, "device", "cuda:0"))
    else:
        raise Exception("Model not implemented")
    return model
