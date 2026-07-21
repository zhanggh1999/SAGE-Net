# Copyright Universitat Politècnica de Catalunya 2024 https://imatge.upc.edu
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

import torch
import torch_geometric.nn as pyg_nn
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.graphgym.config import cfg
from torch_scatter import scatter
from models.utils import ExpNormalSmearing, CosineCutoff


def _parse_layer_indices(layer_spec):
    if layer_spec is None:
        return []
    if isinstance(layer_spec, (list, tuple, set)):
        return [int(idx) for idx in layer_spec]
    return [int(idx.strip()) for idx in str(layer_spec).split(",") if idx.strip()]


class ProjectionHead(nn.Module):
    """Projection head used to align graph and text representations."""

    def __init__(self, embedding_dim, projection_dim=64, dropout=0.1):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        return self.layer_norm(x)


class MiddleFusionModule(nn.Module):
    """Inject graph-level text features into node representations."""

    def __init__(
        self,
        node_dim=256,
        text_dim=64,
        hidden_dim=128,
        num_heads=2,
        dropout=0.1,
        use_gate_norm=False,
        use_learnable_scale=False,
        initial_scale=1.0,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_gate_norm = use_gate_norm

        self.text_transform = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_dim),
        )

        if use_learnable_scale:
            self.text_scale = nn.Parameter(torch.tensor(initial_scale, dtype=torch.float32))
        else:
            self.register_buffer("text_scale", torch.tensor(1.0, dtype=torch.float32))

        if use_gate_norm:
            self.gate_norm = nn.LayerNorm(node_dim * 2)

        self.gate = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.Sigmoid(),
        )
        self.layer_norm = nn.LayerNorm(node_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_feat, text_feat, batch_index=None):
        if text_feat.dim() == 1:
            text_feat = text_feat.unsqueeze(0)

        text_transformed = self.text_transform(text_feat) * self.text_scale

        if batch_index is not None:
            text_broadcasted = text_transformed[batch_index]
        elif node_feat.size(0) == text_transformed.size(0):
            text_broadcasted = text_transformed
        else:
            text_broadcasted = text_transformed.mean(dim=0, keepdim=True).repeat(node_feat.size(0), 1)

        gate_input = torch.cat([node_feat, text_broadcasted], dim=-1)
        if self.use_gate_norm:
            gate_input = self.gate_norm(gate_input)

        gate_values = self.gate(gate_input)
        enhanced = node_feat + gate_values * text_broadcasted
        enhanced = self.layer_norm(enhanced)
        return self.dropout(enhanced)


class TextConditionedMiddleModule(nn.Module):
    """Baseline-preserving text-conditioned modulation of a CartNet update."""

    def __init__(
        self,
        node_dim=256,
        text_dim=64,
        hidden_dim=128,
        dropout=0.1,
        use_gate_norm=True,
        use_learnable_scale=True,
        initial_scale=1.0,
        gate_bias=-3.0,
        correction_scale=0.1,
    ):
        super().__init__()
        self.use_gate_norm = use_gate_norm
        self.gate_bias = gate_bias
        self.correction_scale = correction_scale

        self.graph_projection = ProjectionHead(
            embedding_dim=node_dim,
            projection_dim=text_dim,
            dropout=dropout,
        )
        condition_dim = text_dim * 2
        if use_gate_norm:
            self.gate_norm = nn.LayerNorm(condition_dim)

        self.conditioner = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.modulation = nn.Linear(hidden_dim, node_dim * 3)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

        if use_learnable_scale:
            self.text_scale = nn.Parameter(torch.tensor(initial_scale, dtype=torch.float32))
        else:
            self.register_buffer("text_scale", torch.tensor(initial_scale, dtype=torch.float32))

    def forward(self, x_before, x_after, text_feat, batch_index):
        dim_size = int(batch_index.max().item() + 1)
        graph_feat = scatter(x_before, batch_index, dim=0, reduce="mean", dim_size=dim_size)
        graph_context = self.graph_projection(graph_feat)
        condition = torch.cat([graph_context, text_feat], dim=-1)

        if self.use_gate_norm:
            condition = self.gate_norm(condition)

        hidden = self.conditioner(condition)
        gamma, beta, gate_logits = self.modulation(hidden).chunk(3, dim=-1)
        gamma = self.correction_scale * torch.tanh(gamma)
        beta = self.correction_scale * torch.tanh(beta)
        gate = torch.sigmoid(gate_logits + self.gate_bias)

        delta = x_after - x_before
        node_correction = gate[batch_index] * (gamma[batch_index] * delta + beta[batch_index])
        return x_after + self.text_scale * node_correction


def symmetric_graph_text_contrastive_loss(graph_emb, text_emb, temperature=0.1):
    if graph_emb.size(0) < 2:
        return graph_emb.new_zeros(())

    graph_emb = F.normalize(graph_emb, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)
    logits = torch.matmul(graph_emb, text_emb.t()) / max(float(temperature), 1e-8)
    labels = torch.arange(graph_emb.size(0), device=graph_emb.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


class GatedFusion(nn.Module):
    """Late gated fusion matching the SGA-fusion graph/text gate design."""

    def __init__(self, graph_dim=64, text_dim=64, output_dim=64, dropout=0.1):
        super().__init__()
        self.gate_graph = nn.Sequential(
            nn.Linear(graph_dim, graph_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(graph_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.gate_text = nn.Sequential(
            nn.Linear(text_dim, text_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(text_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.graph_transform = nn.Linear(graph_dim, output_dim)
        self.text_transform = nn.Linear(text_dim, output_dim)
        self.fusion_transform = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, graph_feat, text_feat):
        gate_g = self.gate_graph(graph_feat)
        gate_t = self.gate_text(text_feat)
        gate_sum = gate_g + gate_t + 1e-8
        gate_g = gate_g / gate_sum
        gate_t = gate_t / gate_sum

        graph_transformed = self.graph_transform(graph_feat)
        text_transformed = self.text_transform(text_feat)
        fused = gate_g * graph_transformed + gate_t * text_transformed
        return self.fusion_transform(fused)


class CartNet(torch.nn.Module):
    """
    CartNet model from Cartesian Encoding Graph Neural Network for Crystal Structures Property Prediction: Application to Thermal Ellipsoid Estimation.
    Args:
        dim_in (int): Dimensionality of the input features.
        dim_rbf (int): Dimensionality of the radial basis function embeddings.
        num_layers (int): Number of CartNet layers in the model.
        radius (float, optional): Radius cutoff for neighbor interactions. Default is 5.0.
        invariant (bool, optional): If `True`, enforces rotational invariance in the encoder. Default is `False`.
        temperature (bool, optional): If `True`, includes temperature information in the encoder. Default is `True`.
        use_envelope (bool, optional): If `True`, applies an envelope function to the interactions. Default is `True`.
        cholesky (bool, optional): If `True`, uses a Cholesky head for the output. If `False`, uses a scalar head. Default is `True`.
    Methods:
        forward(batch):
            Performs a forward pass of the model.
            Args:
                batch: A batch of input data.
            Returns:
                pred: The model's predictions.
                true: The ground truth values corresponding to the input batch.
    """


    def __init__(self, 
        dim_in: int, 
        dim_rbf: int, 
        num_layers: int,
        radius: float = 5.0,
        invariant: bool = False,
        temperature: bool = True, 
        use_envelope: bool = True,
        atom_types: bool = True,
        cholesky: bool = True,
        use_text: bool = False,
        use_late_fusion: bool = True,
        text_embedding_dim: int = 768,
        text_projection_dim: int = 64,
        late_fusion_type: str = "gated",
        late_fusion_output_dim: int = 64,
        fusion_dropout: float = 0.1,
        use_middle_fusion: bool = False,
        middle_fusion_type: str = "residual",
        middle_fusion_layers: str = "2",
        middle_fusion_hidden_dim: int = 128,
        middle_fusion_num_heads: int = 2,
        middle_fusion_dropout: float = 0.1,
        middle_fusion_use_gate_norm: bool = False,
        middle_fusion_use_learnable_scale: bool = False,
        middle_fusion_initial_scale: float = 1.0,
        middle_fusion_gate_bias: float = -3.0,
        middle_fusion_correction_scale: float = 0.1,
        text_sample_dropout: float = 0.0,
        contrastive_weight: float = 0.0,
        contrastive_temperature: float = 0.1,
        contrastive_projection_dim: int = 128):
        super().__init__()
    
        self.encoder = Encoder(dim_in, dim_rbf=dim_rbf, radius=radius, invariant=invariant, temperature=temperature, atom_types=atom_types)
        self.dim_in = dim_in
        self.use_text = use_text
        self.use_late_fusion = use_late_fusion
        self.text_embedding_dim = text_embedding_dim
        self.use_middle_fusion = use_middle_fusion
        self.middle_fusion_type = middle_fusion_type
        self.text_sample_dropout = float(text_sample_dropout)
        self.contrastive_weight = float(contrastive_weight)
        self.contrastive_temperature = float(contrastive_temperature)
        self.extra_loss = {}

        if self.use_middle_fusion and not self.use_text:
            raise ValueError("Middle fusion requires use_text=True.")
        if self.use_text and cholesky:
            raise ValueError("Multimodal CartNet is only implemented for scalar targets.")
        if self.use_middle_fusion and self.middle_fusion_type not in ["residual", "conditioned_update"]:
            raise ValueError(f"Unknown middle_fusion_type: {self.middle_fusion_type}")

        layers = []
        for _ in range(num_layers):
            layers.append(CartNet_layer(
                dim_in=dim_in,
                use_envelope=use_envelope,
            ))
        self.layers = torch.nn.Sequential(*layers)

        self.middle_fusion_modules = nn.ModuleDict()
        self.middle_fusion_layer_indices = []
        if self.use_text:
            self.text_projection = ProjectionHead(
                embedding_dim=text_embedding_dim,
                projection_dim=text_projection_dim,
                dropout=fusion_dropout,
            )
            if self.contrastive_weight > 0:
                self.contrastive_graph_projection = ProjectionHead(
                    embedding_dim=dim_in,
                    projection_dim=contrastive_projection_dim,
                    dropout=fusion_dropout,
                )
                self.contrastive_text_projection = ProjectionHead(
                    embedding_dim=text_projection_dim,
                    projection_dim=contrastive_projection_dim,
                    dropout=fusion_dropout,
                )
            if self.use_middle_fusion:
                self.middle_fusion_layer_indices = _parse_layer_indices(middle_fusion_layers)
                for layer_idx in self.middle_fusion_layer_indices:
                    if layer_idx < 0 or layer_idx >= num_layers:
                        raise ValueError(
                            f"middle_fusion layer index {layer_idx} is outside "
                        f"the valid 0-based range [0, {num_layers - 1}]."
                        )
                    if self.middle_fusion_type == "conditioned_update":
                        self.middle_fusion_modules[f"layer_{layer_idx}"] = TextConditionedMiddleModule(
                            node_dim=dim_in,
                            text_dim=text_projection_dim,
                            hidden_dim=middle_fusion_hidden_dim,
                            dropout=middle_fusion_dropout,
                            use_gate_norm=middle_fusion_use_gate_norm,
                            use_learnable_scale=middle_fusion_use_learnable_scale,
                            initial_scale=middle_fusion_initial_scale,
                            gate_bias=middle_fusion_gate_bias,
                            correction_scale=middle_fusion_correction_scale,
                        )
                    else:
                        self.middle_fusion_modules[f"layer_{layer_idx}"] = MiddleFusionModule(
                            node_dim=dim_in,
                            text_dim=text_projection_dim,
                            hidden_dim=middle_fusion_hidden_dim,
                            num_heads=middle_fusion_num_heads,
                            dropout=middle_fusion_dropout,
                            use_gate_norm=middle_fusion_use_gate_norm,
                            use_learnable_scale=middle_fusion_use_learnable_scale,
                            initial_scale=middle_fusion_initial_scale,
                        )

        if cholesky:
            self.head = Cholesky_head(dim_in)
        elif self.use_text and self.use_late_fusion:
            self.head = MultimodalScalarHead(
                dim_in=dim_in,
                text_dim=text_projection_dim,
                late_fusion_type=late_fusion_type,
                late_fusion_output_dim=late_fusion_output_dim,
                dropout=fusion_dropout,
            )
        else:
            self.head = Scalar_head(dim_in)
        
    def forward(self, batch):
        self.extra_loss = {}
        batch = self.encoder(batch)
        text_emb = None

        if self.use_text:
            if not hasattr(batch, "text_embedding"):
                raise AttributeError("Batch is missing text_embedding. Rebuild the dataset with --use_text.")
            text_embedding = batch.text_embedding
            if text_embedding.dim() == 1:
                if text_embedding.numel() % self.text_embedding_dim != 0:
                    raise ValueError(
                        f"text_embedding has {text_embedding.numel()} values, "
                        f"not a multiple of text_embedding_dim={self.text_embedding_dim}."
                    )
                text_embedding = text_embedding.view(-1, self.text_embedding_dim)
            text_embedding = text_embedding.float()
            if self.training and self.text_sample_dropout > 0:
                keep = torch.rand(
                    text_embedding.size(0), 1,
                    device=text_embedding.device,
                    dtype=text_embedding.dtype,
                ) >= self.text_sample_dropout
                text_embedding = text_embedding * keep
            text_emb = self.text_projection(text_embedding)

        for idx, layer in enumerate(self.layers):
            x_before = batch.x
            batch = layer(batch)
            if self.use_middle_fusion and idx in self.middle_fusion_layer_indices:
                middle_module = self.middle_fusion_modules[f"layer_{idx}"]
                if self.middle_fusion_type == "conditioned_update":
                    batch.x = middle_module(x_before, batch.x, text_emb, batch.batch)
                else:
                    batch.x = middle_module(batch.x, text_emb, batch.batch)
        
        if self.use_text and self.training and self.contrastive_weight > 0:
            dim_size = int(batch.batch.max().item() + 1)
            graph_context = scatter(batch.x, batch.batch, dim=0, reduce="mean", dim_size=dim_size)
            graph_contrastive = self.contrastive_graph_projection(graph_context)
            text_contrastive = self.contrastive_text_projection(text_emb)
            self.extra_loss["graph_text_contrastive_loss"] = self.contrastive_weight * symmetric_graph_text_contrastive_loss(
                graph_contrastive,
                text_contrastive,
                self.contrastive_temperature,
            )

        if self.use_text and self.use_late_fusion:
            pred, true = self.head(batch, text_emb)
        else:
            pred, true = self.head(batch)
        
        return pred,true

class Encoder(torch.nn.Module):
    """
    Encoder module for the CartNet model.
    This module encodes node and edge features for input into the CartNet model, incorporating optional temperature information and rotational invariance.
    Args:
        dim_in (int): Dimension of the input features after embedding.
        dim_rbf (int): Dimension of the radial basis function used for edge attributes.
        radius (float, optional): Cutoff radius for neighbor interactions. Defaults to 5.0.
        invariant (bool, optional): If True, the encoder enforces rotational invariance by excluding directional information from edge attributes. Defaults to False.
        temperature (bool, optional): If True, includes temperature data in the node embeddings. Defaults to True.
    Attributes:
        dim_in (int): Dimension of the input features.
        invariant (bool): Indicates if rotational invariance is enforced.
        temperature (bool): Indicates if temperature information is included.
        embedding (nn.Embedding): Embedding layer mapping atomic numbers to feature vectors.
        temperature_proj_atom (pyg_nn.Linear): Linear layer projecting temperature to embedding dimensions (used if temperature is True).
        bias (nn.Parameter): Bias term added to embeddings (used if temperature is False).
        activation (nn.Module): Activation function (SiLU).
        encoder_atom (nn.Sequential): Sequential network encoding node features.
        encoder_edge (nn.Sequential): Sequential network encoding edge features.
        rbf (ExpNormalSmearing): Radial basis function for encoding distances.
    """
    
    def __init__(
        self,
        dim_in: int,
        dim_rbf: int,
        radius: float = 5.0,
        invariant: bool = False, 
        temperature: bool = True,
        atom_types: bool = True
    ):
        super(Encoder, self).__init__()
        self.dim_in = dim_in
        self.invariant = invariant
        self.temperature = temperature
        self.atom_types = atom_types
        if self.atom_types:
            self.embedding = nn.Embedding(119, self.dim_in*2)
            torch.nn.init.xavier_uniform_(self.embedding.weight.data)
        elif not self.temperature:
            self.embedding = nn.Embedding(1, self.dim_in)

        if self.temperature:
            self.temperature_proj_atom = pyg_nn.Linear(1, self.dim_in*2, bias=True)
        elif self.atom_types:
            self.bias = nn.Parameter(torch.zeros(self.dim_in*2))
        self.activation = nn.SiLU(inplace=True)
        
        if self.temperature or self.atom_types:
            self.encoder_atom = nn.Sequential(self.activation,
                                        pyg_nn.Linear(self.dim_in*2, self.dim_in),
                                        self.activation)
        if self.invariant:
            dim_edge = dim_rbf
        else:
            dim_edge = dim_rbf + 3
        
        self.encoder_edge = nn.Sequential(pyg_nn.Linear(dim_edge, self.dim_in*2),
                                        self.activation,
                                        pyg_nn.Linear(self.dim_in*2, self.dim_in),
                                        self.activation)

        self.rbf = ExpNormalSmearing(0.0,radius,dim_rbf,False)  
        
        

    def forward(self, batch):

        if self.temperature and self.atom_types:
            x = self.embedding(batch.x) + self.temperature_proj_atom(batch.temperature.unsqueeze(-1))[batch.batch]
        elif not self.temperature and self.atom_types:
            x = self.embedding(batch.x) + self.bias
        elif self.temperature and not self.atom_types:
            x = self.temperature_proj_atom(batch.temperature.unsqueeze(-1))[batch.batch]
        else:
            batch.x = self.embedding.weight.repeat(batch.x.shape[0],1)
        
        if self.temperature or self.atom_types:
            batch.x = self.encoder_atom(x)

        if cfg.invariant:
            batch.edge_attr = self.encoder_edge(self.rbf(batch.cart_dist))
        else:
            batch.edge_attr = self.encoder_edge(torch.cat([self.rbf(batch.cart_dist), batch.cart_dir], dim=-1))

        return batch

class CartNet_layer(pyg_nn.conv.MessagePassing):
    """
    The message-passing layer used in the CartNet architecture.
    Parameters:
        dim_in (int): Dimension of the input node features.
        use_envelope (bool, optional): If True, applies an envelope function to the distances. Defaults to True.
    Attributes:
        dim_in (int): Dimension of the input node features.
        activation (nn.Module): Activation function (SiLU) used in the layer.
        MLP_aggr (nn.Sequential): MLP used for aggregating messages.
        MLP_gate (nn.Sequential): MLP used for computing gating coefficients.
        norm (nn.BatchNorm1d): Batch normalization applied to the gating coefficients.
        norm2 (nn.BatchNorm1d): Batch normalization applied to the aggregated messages.
        use_envelope (bool): Indicates if the envelope function is used.
        envelope (CosineCutoff): Envelope function applied to the distances.
    """
    
    def __init__(self, 
        dim_in: int, 
        use_envelope: bool = True
    ):
        super().__init__()
        self.dim_in = dim_in
        self.activation = nn.SiLU(inplace=True) 
        self.MLP_aggr = nn.Sequential(
            pyg_nn.Linear(dim_in*3, dim_in, bias=True),
            self.activation,
            pyg_nn.Linear(dim_in, dim_in, bias=True),
        )
        self.MLP_gate = nn.Sequential(
            pyg_nn.Linear(dim_in*3, dim_in, bias=True),
            self.activation,
            pyg_nn.Linear(dim_in, dim_in, bias=True),
        )
        
        self.norm = nn.BatchNorm1d(dim_in)
        self.norm2 = nn.BatchNorm1d(dim_in)
        self.use_envelope = use_envelope
        self.envelope = CosineCutoff(0, cfg.radius)
        

    def forward(self, batch):

        x, e, edge_index, dist = batch.x, batch.edge_attr, batch.edge_index, batch.cart_dist
        """
        x               : [n_nodes, dim_in]
        e               : [n_edges, dim_in]
        edge_index      : [2, n_edges]
        dist            : [n_edges]
        batch           : [n_nodes]
        """
        
        x_in = x
        e_in = e

        x, e = self.propagate(edge_index,
                                Xx=x, Ee=e,
                                He=dist,
                            )
 
        batch.x = self.activation(x) + x_in
        
        batch.edge_attr = e_in + e 

        return batch


    def message(self, Xx_i, Ee, Xx_j, He):
        """
        x_i           : [n_edges, dim_in]
        x_j           : [n_edges, dim_in]
        e             : [n_edges, dim_in]
        """

        e_ij = self.MLP_gate(torch.cat([Xx_i, Xx_j, Ee], dim=-1))
        e_ij = F.sigmoid(self.norm(e_ij))
        
        if self.use_envelope:
            sigma_ij = self.envelope(He).unsqueeze(-1)*e_ij
        else:
            sigma_ij = e_ij
        
        self.e = sigma_ij
        return sigma_ij

    def aggregate(self, sigma_ij, index, Xx_i, Xx_j, Ee, Xx):
        """
        sigma_ij        : [n_edges, dim_in]  ; is the output from message() function
        index           : [n_edges]
        x_j           : [n_edges, dim_in]
        """
        dim_size = Xx.shape[0]  

        sender = self.MLP_aggr(torch.cat([Xx_i, Xx_j, Ee], dim=-1))
        

        out = scatter(sigma_ij*sender, index, 0, None, dim_size,
                                   reduce='sum')

        return out

    def update(self, aggr_out):
        """
        aggr_out        : [n_nodes, dim_in] ; is the output from aggregate() function after the aggregation
        x             : [n_nodes, dim_in]
        """
        x = self.norm2(aggr_out)
       
        e_out = self.e
        del self.e

        return x, e_out

class Cholesky_head(torch.nn.Module):
    """
    The Cholesky head used in the CartNet model.
    It enforce the positive definiteness of the output covariance matrix.

    Args:
        dim_in (int): The input dimension of the features.
    """
    
    def __init__(self, 
        dim_in: int
    ):
        super(Cholesky_head, self).__init__()
        self.MLP = nn.Sequential(pyg_nn.Linear(dim_in, dim_in//2),
                                nn.SiLU(inplace=True), 
                                pyg_nn.Linear(dim_in//2, 6))

    def forward(self, batch):
        pred = self.MLP(batch.x[batch.non_H_mask])

        diag_elements = F.softplus(pred[:, :3])

        i,j = torch.tensor([0,1,2,0,0,1]), torch.tensor([0,1,2,1,2,2])
        L_matrix = torch.zeros(pred.size(0),3,3, device=pred.device, dtype=pred.dtype)
        L_matrix[:,i[:3], i[:3]] = diag_elements
        L_matrix[:,i[3:], j[3:]] = pred[:,3:]

        U = torch.bmm(L_matrix.transpose(1, 2), L_matrix)
        
        return U, batch.y

class Scalar_head(torch.nn.Module):
    """
    A head to predict scalar values.
    Args:
        dim_in (int): The dimension of the input features.
    """
    
    def __init__(self,
        dim_in
    ):
        super(Scalar_head, self).__init__()

        self.MLP = nn.Sequential(pyg_nn.Linear(dim_in, dim_in//2), 
                                nn.SiLU(inplace=True), 
                                pyg_nn.Linear(dim_in//2, 1))

    def forward(self, batch):
        dim_size = int(batch.batch.max().item() + 1)
        batch.x = self.MLP(batch.x)
        batch.x = scatter(batch.x, batch.batch, dim=0, reduce="mean", dim_size=dim_size).squeeze(-1)
        return batch.x, batch.y


class MultimodalScalarHead(torch.nn.Module):
    """
    Scalar head with graph/text late fusion.
    """

    def __init__(
        self,
        dim_in,
        text_dim=64,
        late_fusion_type="gated",
        late_fusion_output_dim=64,
        dropout=0.1,
    ):
        super(MultimodalScalarHead, self).__init__()
        self.late_fusion_type = late_fusion_type
        self.graph_projection = ProjectionHead(
            embedding_dim=dim_in,
            projection_dim=text_dim,
            dropout=dropout,
        )

        if late_fusion_type == "concat":
            self.fusion_module = None
            self.MLP = nn.Sequential(
                nn.Linear(text_dim * 2, late_fusion_output_dim),
                nn.SiLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(late_fusion_output_dim, 1),
            )
        elif late_fusion_type == "gated":
            self.fusion_module = GatedFusion(
                graph_dim=text_dim,
                text_dim=text_dim,
                output_dim=late_fusion_output_dim,
                dropout=dropout,
            )
            self.MLP = nn.Linear(late_fusion_output_dim, 1)
        else:
            raise ValueError(f"Unknown late_fusion_type: {late_fusion_type}")

    def forward(self, batch, text_emb):
        dim_size = int(batch.batch.max().item() + 1)
        graph_feat = scatter(batch.x, batch.batch, dim=0, reduce="mean", dim_size=dim_size)
        graph_emb = self.graph_projection(graph_feat)

        if text_emb.size(0) != graph_emb.size(0):
            raise ValueError(
                f"text batch size {text_emb.size(0)} does not match graph batch size {graph_emb.size(0)}."
            )

        if self.late_fusion_type == "concat":
            fused = torch.cat([graph_emb, text_emb], dim=-1)
        else:
            fused = self.fusion_module(graph_emb, text_emb)

        pred = self.MLP(fused).squeeze(-1)
        return pred, batch.y
