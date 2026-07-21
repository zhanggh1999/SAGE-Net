# Copyright Universitat Politècnica de Catalunya 2024 https://imatge.upc.edu
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

import torch
import torch.nn.functional as F
import logging
import argparse
import pickle
from tqdm import tqdm
from logger.logger import create_logger
from loader.loader import create_loader
from models.master import create_model
from train.train import train
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric import seed_everything
from torch_geometric.graphgym.config import cfg, set_cfg
from torch_geometric.graphgym.logger import set_printing


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "1", "y"):
        return True
    if value in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {value}.")


def load_matching_pretrained_weights(model, checkpoint_path):
    ckpt = torch.load(
        checkpoint_path,
        weights_only=False,
        map_location=torch.device(getattr(cfg, "device", "cuda:0")),
    )
    pretrained_state = ckpt.get("model_state", ckpt)
    model_state = model.state_dict()

    matched_state = {}
    skipped = []
    for key, value in pretrained_state.items():
        if key in model_state and model_state[key].shape == value.shape:
            matched_state[key] = value
        else:
            skipped.append(key)

    model_state.update(matched_state)
    model.load_state_dict(model_state)

    missing = [key for key in model_state if key not in matched_state]
    logging.info(
        "Loaded %s tensors from pretrained checkpoint %s; skipped %s tensors; "
        "%s model tensors remain newly initialized.",
        len(matched_state),
        checkpoint_path,
        len(skipped),
        len(missing),
    )
    if skipped:
        logging.info("Skipped pretrained tensors: %s", skipped[:20])


def create_optimizer(model):
    graph_lr = getattr(cfg, "graph_lr", None)
    fusion_lr = getattr(cfg, "fusion_lr", None)
    if graph_lr is None and fusion_lr is None:
        cfg.max_lrs = cfg.lr
        return torch.optim.Adam(model.parameters(), lr=cfg.lr)

    graph_lr = cfg.lr if graph_lr is None else graph_lr
    fusion_lr = cfg.lr if fusion_lr is None else fusion_lr
    fusion_keywords = ("text_projection", "middle_fusion_modules", "fusion_module", "graph_projection", "contrastive_")

    graph_params = []
    fusion_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(keyword in name for keyword in fusion_keywords):
            fusion_params.append(param)
        else:
            graph_params.append(param)

    param_groups = []
    max_lrs = []
    if graph_params:
        param_groups.append({"params": graph_params, "lr": graph_lr, "name": "graph"})
        max_lrs.append(graph_lr)
    if fusion_params:
        param_groups.append({"params": fusion_params, "lr": fusion_lr, "name": "fusion"})
        max_lrs.append(fusion_lr)

    cfg.max_lrs = max_lrs
    logging.info(
        "Using parameter groups: graph_lr=%s (%s params), fusion_lr=%s (%s params).",
        graph_lr,
        sum(param.numel() for param in graph_params),
        fusion_lr,
        sum(param.numel() for param in fusion_params),
    )
    return torch.optim.Adam(param_groups)


def inference(model, loader):
    """
    Run inference using the trained model and data loader, compute metrics, and save the results.
    Args:
        model (torch.nn.Module): The trained model to be evaluated.
        loader (torch.utils.data.DataLoader): DataLoader for the dataset to perform inference on.
    This function sets the model to evaluation mode and disables gradient calculations.
    It iterates over the data loader, collects predictions and ground truths, and computes metrics such as IoU,
    Mean Absolute Error (MAE), and similarity index for each batch. The metrics are logged, and all inference outputs
    are saved to a pickle file specified by `cfg.inference_output`.
    """
    from train.metrics import compute_3D_IoU, get_similarity_index
    model.eval()
    
    with torch.no_grad():
        inference_output = {"pred": [], "true": [], "temp": [], "cell": [], "refcode": [], "pos": [], "atoms": [], "iou": [], "mae": [], "similarity_index": []}
        for iter, batch in tqdm(enumerate(loader), total=len(loader), ncols=50):
            batch.to(cfg.device)
            inference_output["cell"].append(batch.cell.detach().to("cpu"))
            inference_output["atoms"].append(batch.x[batch.non_H_mask].detach().to("cpu"))
            inference_output["pos"].append(batch.pos[batch.non_H_mask].detach().to("cpu"))
            inference_output["refcode"].append(batch.refcode[0])
            inference_output["temp"].append(batch.temperature_og.detach().to("cpu")[0])
            _pred, _true = model(batch)
            inference_output["pred"].append(_pred.detach().to("cpu"))
            inference_output["true"].append(_true.detach().to("cpu"))
            inference_output["iou"].append(compute_3D_IoU(_pred, _true).detach().to("cpu"))
            inference_output["mae"].append(F.l1_loss(_pred,_true, reduce="none").detach().to("cpu"))
            inference_output["similarity_index"].append(get_similarity_index(_pred, _true).detach().to("cpu"))
        
        
        iou = torch.cat(inference_output["iou"], dim=0)
        mae = torch.cat(inference_output["mae"], dim=0)
        similarity_index = torch.cat(inference_output["similarity_index"], dim=0)
        
        logging.info(f"Mean IoU: {iou.mean().item()} +/- {iou.std().item()}")
        logging.info(f"Mean MAE: {mae.mean().item()} +/- {mae.std().item()}")
        logging.info(f"Mean Similarity Index: {similarity_index.mean().item()} +/- {similarity_index.std().item()}")

        pickle.dump(inference_output, open(cfg.inference_output, "wb"))

def montecarlo(model, loader):
    """
    Performs Monte Carlo simulations to evaluate the model's performance under random rotations.
    Args:
        model (torch.nn.Module): The trained model to be evaluated.
        loader (torch.utils.data.DataLoader): DataLoader providing the dataset for evaluation.
    The function runs multiple iterations (e.g., 100) where it:
    - Applies a random rotation to the input batch data.
    - Performs a forward pass to obtain predictions.
    - Computes evaluation metrics such as Intersection over Union (IoU), Mean Absolute Error (MAE), and similarity index.
    - Stores and logs the results for each iteration.
    After all iterations, it aggregates the metrics to compute the mean and standard deviation, providing insights into the model's robustness to rotations.
    Results are saved to output files specified in the configuration, and important metrics are logged for analysis.
    """
    from train.metrics import compute_3D_IoU, get_similarity_index
    import roma

    model.eval()
    iou_montecarlo = []
    similarity_index_montecarlo = []
    mae_montecarlo = []
    with torch.no_grad():
        for i in tqdm(range(100), ncols=50, desc="Montecarlo"):
            inference_output = {"pred": [], "true": [], "cell": [], "refcode": [], "pos": [], "atoms": [], "mae": [], "iou": [], "similarity_index": []}
            for iter, batch in tqdm(enumerate(loader), total=len(loader), ncols=50):
                batch_copy = batch.clone()
                batch.to(cfg.device)
                inference_output["cell"].append(batch.cell.detach().to("cpu"))
                inference_output["atoms"].append(batch.x[batch.non_H_mask].detach().to("cpu"))
                inference_output["pos"].append(batch.pos[batch.non_H_mask].detach().to("cpu"))
                inference_output["refcode"].append(batch.refcode[0])
                pseudo_true, _ = model(batch)
                R = roma.utils.random_rotmat(size=1, device=pseudo_true.device).squeeze(0)
                batch_copy.to(cfg.device)
                batch_copy.cart_dir = batch_copy.cart_dir @ R
                pseudo_true =  R.transpose(-1,-2) @ pseudo_true @ R
                pred, _ = model(batch_copy)
                inference_output["pred"].append(pred.detach().to("cpu"))
                inference_output["true"].append(pseudo_true.detach().to("cpu"))
                inference_output["iou"].append(compute_3D_IoU(pred, pseudo_true).detach().to("cpu"))
                inference_output["similarity_index"].append(get_similarity_index(pred, pseudo_true).detach().to("cpu"))
                inference_output["mae"].append(F.l1_loss(pred, pseudo_true, reduce="none").detach().to("cpu"))
            pickle.dump(inference_output, open(cfg.inference_output.replace(".pkl", "_montecarlo_"+str(i)+".pkl"), "wb"))
            logging.info(f"Montecarlo {i}")
            logging.info(f"IoU: {torch.cat(inference_output['iou'], dim=0).mean().item()}")
            logging.info(f"MAE: {torch.cat(inference_output['mae'], dim=0).mean().item()}")
            logging.info(f"Similarity Index: {torch.cat(inference_output['similarity_index'], dim=0).mean().item()}")
            iou_montecarlo+=inference_output["iou"]
            mae_montecarlo+=inference_output["mae"]
            similarity_index_montecarlo+=inference_output["similarity_index"]
    
    iou_montecarlo = torch.cat(iou_montecarlo, dim=0)
    mae_montecarlo = torch.cat(mae_montecarlo, dim=0)
    similarity_index_montecarlo = torch.cat(similarity_index_montecarlo, dim=0)

    logging.info(f"Montecarlo IoU: {iou_montecarlo.mean().item()} +/- {iou_montecarlo.std().item()}")
    logging.info(f"Montecarlo MAE: {mae_montecarlo.mean().item()} +/- {mae_montecarlo.std().item()}")
    logging.info(f"Montecarlo Similarity Index: {similarity_index_montecarlo.mean().item()} +/- {similarity_index_montecarlo.std().item()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123, help='Seed for the experiment')
    parser.add_argument('--name', type=str, default="CartNet", help="name of the Wandb experiment" )
    parser.add_argument("--run_dir", type=str, default=None, help="Output directory; defaults to results/<name>/<seed>")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--batch_accumulation", type=int, default=16, help="Batch Accumulation")
    parser.add_argument(
        "--dataset",
        type=str,
        default="ADP",
        help=(
            "Dataset name. Available: ADP, jarvis, megnet, slme, bulk_new, "
            "avg, max, mbj, mepsz, optb88, seebeck, shear, spillage"
        ),
    )
    parser.add_argument("--dataset_path", type=str, default="./dataset/ADP_DATASET/")
    parser.add_argument("--inference", action="store_true", help="Inference")
    parser.add_argument("--montecarlo", action="store_true", help="Montecarlo")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path of the checkpoints of the model")
    parser.add_argument("--pretrained_graph_checkpoint", type=str, default=None, help="Warm-start graph model checkpoint for training")
    parser.add_argument("--inference_output", type=str, default="./inference.pkl", help="Path to the inference output")
    parser.add_argument("--figshare_target", type=str, default="formation_energy_peratom", help="Figshare dataset target")
    parser.add_argument("--wandb_project", type=str, default="ADP", help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default="aiquaneuro", help="Name of the wandb entity")
    parser.add_argument("--loss", type=str, default="MAE", help="Loss function")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=0,
        help="Stop training after this many epochs without validation MAE improvement; set <=0 to disable",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--graph_lr", type=float, default=None, help="Learning rate for pretrained graph backbone/head")
    parser.add_argument("--fusion_lr", type=float, default=None, help="Learning rate for text and fusion modules")
    parser.add_argument("--warmup", type=float, default=0.01, help="Warmup")
    parser.add_argument('--model', type=str, default="CartNet", help="Model Name")
    parser.add_argument("--max_neighbours", type=int, default=25, help="Max neighbours (only for iComformer/eComformer)")
    parser.add_argument("--radius", type=float, default=5.0, help="Radius for the Radius Graph Neighbourhood")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--dim_in", type=int, default=256, help="Input dimension")
    parser.add_argument("--dim_rbf", type=int, default=64, help="Number of RBF")
    parser.add_argument('--augment', action='store_true', help='augment')
    parser.add_argument("--invariant", action="store_true", help="Rotation Invariant model")
    parser.add_argument("--disable_temp", action="store_false", help="Disable Temperature")
    parser.add_argument("--no_standarize_temp", action="store_false", help="Standarize temperature")
    parser.add_argument("--disable_envelope", action="store_false", help="Disable envelope")
    parser.add_argument('--disable_H', action='store_false', help='Hydrogens')
    parser.add_argument('--disable_atom_types', action='store_false', help='Atom types')
    parser.add_argument("--threads", type=int, default=8, help="Number of threads")
    parser.add_argument("--workers", type=int, default=5, help="Number of workers")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device, e.g. cuda:0 or cpu")
    parser.add_argument("--use_text", type=str2bool, nargs="?", const=True, default=False, help="Enable text embeddings for multimodal training")
    parser.add_argument("--description_file", type=str, default="description.csv", help="Description CSV filename inside dataset_path")
    parser.add_argument("--text_embedding_file", type=str, default="text_embeddings.npy", help="Text embedding .npy filename inside dataset_path")
    parser.add_argument("--use_late_fusion", type=str2bool, nargs="?", const=True, default=True, help="Use graph/text late fusion head when text is enabled")
    parser.add_argument("--text_embedding_dim", type=int, default=768, help="Dimension of precomputed text_embeddings.npy rows")
    parser.add_argument("--text_projection_dim", type=int, default=64, help="Shared graph/text projection dimension")
    parser.add_argument("--late_fusion_type", type=str, default="gated", choices=["concat", "gated"], help="Late fusion type for multimodal scalar prediction")
    parser.add_argument("--late_fusion_output_dim", type=int, default=64, help="Output dimension of late fusion")
    parser.add_argument("--fusion_dropout", type=float, default=0.1, help="Dropout used in multimodal projection/fusion layers")
    parser.add_argument("--use_middle_fusion", type=str2bool, nargs="?", const=True, default=False, help="Inject text features into intermediate CartNet layers")
    parser.add_argument("--middle_fusion_type", type=str, default="residual", choices=["residual", "conditioned_update"], help="Middle fusion module type")
    parser.add_argument("--middle_fusion_layers", type=str, default="2", help="Comma-separated 0-based CartNet layer indices for middle fusion")
    parser.add_argument("--middle_fusion_hidden_dim", type=int, default=128, help="Hidden dimension of the middle fusion text MLP")
    parser.add_argument("--middle_fusion_num_heads", type=int, default=2, help="Compatibility option; middle fusion uses gated MLP")
    parser.add_argument("--middle_fusion_dropout", type=float, default=0.1, help="Dropout for middle fusion")
    parser.add_argument("--middle_fusion_use_gate_norm", type=str2bool, nargs="?", const=True, default=False, help="Use LayerNorm before middle-fusion gate")
    parser.add_argument("--middle_fusion_use_learnable_scale", type=str2bool, nargs="?", const=True, default=False, help="Use a learnable scale on middle-fusion text features")
    parser.add_argument("--middle_fusion_initial_scale", type=float, default=1.0, help="Initial middle-fusion text scale")
    parser.add_argument("--middle_fusion_gate_bias", type=float, default=-3.0, help="Initial gate bias for conditioned middle fusion")
    parser.add_argument("--middle_fusion_correction_scale", type=float, default=0.1, help="Tanh scale for conditioned middle-fusion gamma/beta")
    parser.add_argument("--text_sample_dropout", type=float, default=0.0, help="Drop full text embeddings for this fraction of training samples")
    parser.add_argument("--contrastive_weight", type=float, default=0.0, help="Weight for graph-text contrastive loss")
    parser.add_argument("--contrastive_temperature", type=float, default=0.1, help="Temperature for graph-text contrastive loss")
    parser.add_argument("--contrastive_projection_dim", type=int, default=128, help="Projection dimension for graph-text contrastive loss")
    
    set_cfg(cfg)

    args = parser.parse_args()
    cfg.seed = args.seed
    cfg.name = args.name
    cfg.run_dir = args.run_dir if args.run_dir is not None else "results/"+cfg.name+"/"+str(cfg.seed)
    cfg.inference_output = args.inference_output
    cfg.pretrained_graph_checkpoint = args.pretrained_graph_checkpoint
    cfg.dataset.task_type = "regression"
    cfg.batch = args.batch
    cfg.batch_accumulation = args.batch_accumulation
    cfg.dataset.name = args.dataset
    cfg.dataset_path = args.dataset_path
    cfg.figshare_target = args.figshare_target
    cfg.wandb_project = args.wandb_project
    cfg.wandb_entity = args.wandb_entity
    cfg.loss = args.loss
    cfg.optim.max_epoch = args.epochs
    cfg.early_stop_patience = args.early_stop_patience
    cfg.lr = args.lr
    cfg.graph_lr = args.graph_lr
    cfg.fusion_lr = args.fusion_lr
    cfg.warmup = args.warmup
    cfg.model = args.model
    cfg.max_neighbours = -1 if cfg.model== "CartNet" else args.max_neighbours
    cfg.radius = args.radius
    cfg.num_layers = args.num_layers
    cfg.dim_in = args.dim_in
    cfg.dim_rbf = args.dim_rbf
    cfg.augment = False if cfg.model in ["icomformer", "ecomformer"] else args.augment
    cfg.invariant = args.invariant
    cfg.use_temp = False if cfg.dataset.name != "ADP" else args.disable_temp
    cfg.standarize_temp = args.no_standarize_temp
    cfg.envelope = args.disable_envelope
    cfg.use_H = args.disable_H
    cfg.use_atom_types = args.disable_atom_types
    cfg.workers = args.workers
    cfg.device = args.device
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


    torch.set_num_threads(args.threads)

    set_printing()

    #Seed
    seed_everything(cfg.seed)

    logging.info(f"Experiment will be saved at: {cfg.run_dir}")

    loaders = create_loader()

    model = create_model()
    if cfg.pretrained_graph_checkpoint is not None and not args.inference and not args.montecarlo:
        load_matching_pretrained_weights(model, cfg.pretrained_graph_checkpoint)

    logging.info(model)
    cfg.params_count = params_count(model)
    logging.info(f"Number of parameters: {cfg.params_count}")

    optimizer = create_optimizer(model)

    loggers = create_logger()

    if args.inference:
        assert args.checkpoint_path is not None, "Weights path not provided"
        assert cfg.dataset.name == "ADP", "Inference only for ADP dataset"
        ckpt = torch.load(
            args.checkpoint_path,
            weights_only=False,
            map_location=torch.device(cfg.device),
        )
        model.load_state_dict(ckpt["model_state"])
        cfg.inference_output = args.inference_output
        inference(model, loaders[-1])
    elif args.montecarlo:
        assert args.checkpoint_path is not None, "Weights path not provided"
        assert cfg.dataset.name == "ADP", "Montecarlo only for ADP dataset"
        ckpt = torch.load(
            args.checkpoint_path,
            weights_only=False,
            map_location=torch.device(cfg.device),
        )
        model.load_state_dict(ckpt["model_state"])
        cfg.inference_output = args.inference_output
        montecarlo(model, loaders[-1])
    else:
        train(model, loaders, optimizer, loggers)
