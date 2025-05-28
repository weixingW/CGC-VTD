import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GATConv, SAGEConv
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.loader import NeighborLoader
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import wandb
import argparse
import sys
from torchvision import transforms
from torchvision.datasets import ImageFolder
import random
from collections import defaultdict
import math
import os
sys.path.append(".")
sys.path.append("..")
from cgc_utils import CodebookGraphDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodebookGNN(nn.Module):
    """Graph Neural Network for learning codebook embeddings"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 32,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.1,
        num_classes: int = 1000,  # ImageNet classes
        edge_dim: int = 1,  # Dimension of edge features
        gnn_type: str = 'gatv2'  # 'gat', 'gatv2', or 'sage'
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.gnn_type = gnn_type.lower()
        
        # Ensure hidden_dim is divisible by num_heads for GAT variants
        if self.gnn_type in ['gat', 'gatv2']:
            assert hidden_dim % num_heads == 0, f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            assert self.output_dim % num_heads == 0, f"output_dim ({self.output_dim}) must be divisible by num_heads ({num_heads})"
        
        # Input projection to match hidden dimension
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        
        # Input layer
        if self.gnn_type == 'gat':
            self.gnn_layers.append(
                GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True,
                    edge_dim=edge_dim,
                    add_self_loops=True
                )
            )
        elif self.gnn_type == 'gatv2':
            self.gnn_layers.append(
                GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True,
                    edge_dim=edge_dim,
                    add_self_loops=True
                )
            )
        elif self.gnn_type == 'sage':
            self.gnn_layers.append(
                SAGEConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    normalize=True
                )
            )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            if self.gnn_type == 'gat':
                self.gnn_layers.append(
                    GATConv(
                        in_channels=hidden_dim,
                        out_channels=hidden_dim // num_heads,
                        heads=num_heads,
                        dropout=dropout,
                        concat=True,
                        edge_dim=edge_dim,
                        add_self_loops=True
                    )
                )
            elif self.gnn_type == 'gatv2':
                self.gnn_layers.append(
                    GATv2Conv(
                        in_channels=hidden_dim,
                        out_channels=hidden_dim // num_heads,
                        heads=num_heads,
                        dropout=dropout,
                        concat=True,
                        edge_dim=edge_dim,
                        add_self_loops=True
                    )
                )
            elif self.gnn_type == 'sage':
                self.gnn_layers.append(
                    SAGEConv(
                        in_channels=hidden_dim,
                        out_channels=hidden_dim,
                        normalize=True
                    )
                )
        
        # Output layer
        if self.gnn_type in ['gat', 'gatv2']:
            self.gnn_layers.append(
                GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=self.output_dim // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True,
                    edge_dim=edge_dim,
                    add_self_loops=True
                )
            )
        elif self.gnn_type == 'sage':
            self.gnn_layers.append(
                SAGEConv(
                    in_channels=hidden_dim,
                    out_channels=self.output_dim,
                    normalize=True
                )
            )
        
        # Layer normalization after each GNN layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim if i < num_layers - 1 else self.output_dim)
            for i in range(num_layers)
        ])
        
    def forward(self, x, edge_index, edge_attr=None, return_all=False):
        # Project input to hidden dimension
        hidden = self.input_proj(x)  # [num_nodes, hidden_dim]
        
        # Store intermediate representations for skip connections
        hidden_states = [hidden]
        
        # Apply GNN layers with skip connections
        for i, (gnn_layer, norm) in enumerate(zip(self.gnn_layers, self.layer_norms)):
            # Ensure edge_attr is properly shaped for GAT variants
            if self.gnn_type in ['gat', 'gatv2'] and edge_attr is not None:
                # Reshape edge_attr to [num_edges, 1] if it's 1D
                if edge_attr.dim() == 1:
                    edge_attr = edge_attr.unsqueeze(1)
            
            # Apply GNN layer
            if self.gnn_type in ['gat', 'gatv2']:
                hidden = gnn_layer(hidden, edge_index, edge_attr=edge_attr)
            else:  # GraphSAGE
                hidden = gnn_layer(hidden, edge_index)
            
            # Skip connection before normalization
            if i < len(self.gnn_layers) - 1:  # Skip connection for all but last layer
                # Project skip connection if dimensions don't match
                prev_hidden = hidden_states[-1]
                if prev_hidden.size(-1) != hidden.size(-1):
                    prev_hidden = nn.Linear(prev_hidden.size(-1), hidden.size(-1), 
                                          device=hidden.device)(prev_hidden)
                hidden = hidden + prev_hidden
            
            # Apply normalization and activation
            hidden = norm(hidden)
            hidden = F.relu(hidden)
            hidden = F.dropout(hidden, p=0.1, training=self.training)
            
            hidden_states.append(hidden)
        
        # Get node embeddings
        node_embeddings = hidden
        
        return node_embeddings

def info_nce_loss(embeddings, edge_index, batch_size, edge_attr=None, temperature=0.1, target_pos_sim=0.5,use_pos_reg=True):
    """InfoNCE contrastive loss with edge attribute weighting and positive similarity regularization
    
    Args:
        embeddings: Node embeddings [batch_size, embedding_dim]
        edge_index: Graph connectivity [2, num_edges]
        batch_size: Number of central nodes in the batch
        edge_attr: Edge attributes representing connection strength [num_edges]
        temperature: Temperature parameter for scaling similarities
        target_pos_sim: Target value for positive similarities (higher = encourage stronger connections)
    """
    # Normalize embeddings
    embeddings = F.normalize(embeddings, dim=1)
    
    # Get edges that connect only to nodes in the batch
    mask = (edge_index[0] < batch_size) & (edge_index[1] < batch_size)
    batch_edge_index = edge_index[:, mask]
        
    # Get corresponding edge attributes if provided
    batch_edge_attr = None
    if edge_attr is not None:
        batch_edge_attr = edge_attr[mask]
        # Handle edge attributes to prevent numerical issues
        if batch_edge_attr.dim() > 1:
            batch_edge_attr = batch_edge_attr[:, 0]  # Use first dimension if multi-dimensional
        if batch_edge_attr.numel()==0:
            print(mask.sum().item())
            print("no connections, return 0")
            return torch.tensor(0)
        # Normalize to [0, 1] range with clamping to prevent extreme values
        if batch_edge_attr.max() > 0:  # Prevent division by zero
            batch_edge_attr = torch.clamp(batch_edge_attr / (batch_edge_attr.max() + 1e-8), 0.0, 1.0)
    
    # Skip if no edges in batch
    if batch_edge_index.shape[1] == 0:
        return torch.tensor(0.0, device=embeddings.device)
    
    # Compute similarity matrix only for nodes in the batch
    similarity_matrix = torch.matmul(embeddings, embeddings.T)
    
    # Create positive pair mask from edge_index
    positive_mask = torch.zeros((batch_size, batch_size), device=embeddings.device)
    positive_mask[batch_edge_index[0], batch_edge_index[1]] = 1.0
    positive_mask[batch_edge_index[1], batch_edge_index[0]] = 1.0  # Make symmetric
    
    # If edge attributes are available, use them to weight positive pairs
    edge_weight_matrix = torch.zeros_like(positive_mask)
    if batch_edge_attr is not None:
        # Convert edge weights to a matrix form
        edge_weight_matrix[batch_edge_index[0], batch_edge_index[1]] = batch_edge_attr
        edge_weight_matrix[batch_edge_index[1], batch_edge_index[0]] = batch_edge_attr  # Make symmetric
    else:
        edge_weight_matrix = positive_mask.clone()  # Default to binary weights
    
    # Remove self-loops from positive mask and weights
    positive_mask.fill_diagonal_(0)
    edge_weight_matrix.fill_diagonal_(0)
    
    # Skip if no positive pairs
    if positive_mask.sum() == 0:
        return torch.tensor(0.0, device=embeddings.device)
    
    # Calculate average positive similarity (before temperature scaling)
    # This will be used for the positive similarity regularization
    if positive_mask.sum() > 0:
        if batch_edge_attr is not None:
            # Weight by edge attributes
            pos_sim_avg = (similarity_matrix * edge_weight_matrix).sum() / (edge_weight_matrix.sum() + 1e-8)
        else:
            # Unweighted average
            pos_sim_avg = similarity_matrix[positive_mask.bool()].mean()
    else:
        pos_sim_avg = torch.tensor(0.0, device=embeddings.device)
    
    # Create negative mask (all non-connected pairs)
    negative_mask = 1 - positive_mask
    negative_mask.fill_diagonal_(0)  # Remove self-loops
    
    # Scale similarities by temperature
    temperature = max(temperature, 0.05)  # Ensure temperature isn't too small
    similarity_matrix = similarity_matrix / temperature
    
    # For numerical stability
    logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
    similarity_matrix = similarity_matrix - logits_max.detach()
    
    # Compute positive and negative similarities
    # Weight positive pairs by edge attributes if available
    if batch_edge_attr is not None:
        positive_similarities = similarity_matrix * edge_weight_matrix
    else:
        positive_similarities = similarity_matrix * positive_mask
    
    # InfoNCE loss calculation with edge attribute weighting
    exp_sim = torch.exp(similarity_matrix)
    
    # Sum of all exponentiated similarities
    exp_sum = exp_sim.sum(dim=1, keepdim=True)
    
    # Compute log probability
    log_prob = positive_similarities - torch.log(exp_sum + 1e-8)
    
    # Weighted mean over positive pairs based on edge weights
    if batch_edge_attr is not None:
        # Use edge weights to weight the loss contribution of each positive pair
        weight_sum = edge_weight_matrix.sum()
        if weight_sum > 0:  # Ensure there are weighted edges
            weighted_log_prob = (log_prob * edge_weight_matrix).sum() / weight_sum
        else:
            weighted_log_prob = log_prob[positive_mask.bool()].mean()
    else:
        # Standard unweighted mean
        weighted_log_prob = log_prob[positive_mask.bool()].mean()
    
    # Positive similarity regularization
    # Encourages positive similarities to be at least target_pos_sim
    # Create adaptive target based on edge weights when available
    if batch_edge_attr is not None:
        # Higher edge weight = higher target similarity
        target_similarities = torch.zeros_like(similarity_matrix)
        # Scale target similarities based on edge weight: stronger connections -> higher target
        min_target = target_pos_sim * 0.5  # Minimum target for any positive pair
        max_target = target_pos_sim       # Maximum target for strongest connections
        
        # Set target similarities for positive pairs based on edge weight
        for i in range(batch_edge_index.shape[1]):
            src, dst = batch_edge_index[0, i], batch_edge_index[1, i]
            if src < batch_size and dst < batch_size:
                # Scale target based on edge weight
                edge_weight = batch_edge_attr[i]
                # strategy1: min_target + edge_weight * (max_target - min_target)
                #pair_target = min_target + edge_weight * (max_target - min_target)
                # strategy2: edge_weight*beta
                pair_target = edge_weight*0.8
                target_similarities[src, dst] = pair_target
                target_similarities[dst, src] = pair_target  # Make symmetric
                
        # Calculate positive similarity penalty
        pos_sim_penalty_mask = (similarity_matrix < target_similarities) & positive_mask.bool()
        if pos_sim_penalty_mask.sum() > 0:
            pos_sim_penalty = ((target_similarities - similarity_matrix) * pos_sim_penalty_mask).sum() / pos_sim_penalty_mask.sum()
        else:
            pos_sim_penalty = torch.tensor(0.0, device=embeddings.device)
    else:
        # Simpler approach for unweighted edges
        # Penalize if average positive similarity is below target
        pos_sim_penalty = F.relu(target_pos_sim - pos_sim_avg)
    
    # Combined loss with regularization
    # Balance the regularization strength - adjust weight as needed
    reg_weight = 1.0  # Weight for positive similarity regularization
    
    # The negative log prob is the main InfoNCE loss
    nce_loss = -weighted_log_prob
    
    # Combined loss
    if use_pos_reg:
        combined_loss = nce_loss + reg_weight * pos_sim_penalty
    else:
        combined_loss = nce_loss
    
    # For logging, track the components separately
    # Store these as attributes of the tensor for retrieval later
    combined_loss.nce_loss = nce_loss.detach()
    combined_loss.pos_sim_penalty = pos_sim_penalty.detach()
    combined_loss.pos_sim_avg = pos_sim_avg.detach()
    
    # Final safety check to prevent NaN or infinite loss
    if torch.isnan(combined_loss) or torch.isinf(combined_loss):
        return torch.tensor(0.1, device=embeddings.device, requires_grad=True)
    
    return combined_loss



def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    """
    
    def lr_lambda(current_step):
        # Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

class ImageNetClassifier:
    """Wrapper for classifying ImageNet images using codebook embeddings"""
    
    def __init__(self, gnn_model, clusterer, processor=None, transform=None):
        self.gnn_model = gnn_model
        self.clusterer = clusterer
        self.processor = processor
        self.transform = transform
        
    def get_image_embedding(self, image):
        """Get GNN embeddings for codebook entries used in image"""
        device = next(self.gnn_model.parameters()).device
        model_dtype = next(self.gnn_model.parameters()).dtype
        
        # Process image based on model type
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
            
        if hasattr(self.clusterer.model, 'model') and hasattr(self.clusterer.model.model, 'vqmodel'):
            # Chameleon model
            dummy_prompt = "<image>"
            inputs = self.processor(text=dummy_prompt, images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device, dtype=model_dtype)
            
            with torch.no_grad():
                encoder_output = self.clusterer.model.model.vqmodel.encoder(pixel_values)
                hidden_states = self.clusterer.model.model.vqmodel.quant_conv(encoder_output)
                _, _, indices = self.clusterer.model.model.vqmodel.quantize(hidden_states)
                
        elif hasattr(self.clusterer.model, 'gen_vision_model'):
            # Janus model
            conversation = [
                {"role": "<|User|>", "content": "<image_placeholder>", "images": [image]},
                {"role": "<|Assistant|>", "content": ""}
            ]
            inputs = self.processor(conversations=conversation, images=[image], force_batchify=True)
            pixel_values = inputs["pixel_values"].squeeze(0).to(device, dtype=model_dtype)
            
            with torch.no_grad():
                vqmodel = self.clusterer.model.gen_vision_model
                encoder_output = vqmodel.encoder(pixel_values)
                hidden_states = vqmodel.quant_conv(encoder_output)
                _, _, info = vqmodel.quantize(hidden_states)
                indices = info[2]
                
        else:
            # LlamaGen model
            if self.transform:
                pixel_values = self.transform(image).unsqueeze(0).to(device, dtype=model_dtype)
            else:
                pixel_values = image.unsqueeze(0).to(device, dtype=model_dtype)
                
            with torch.no_grad():
                encoder_output = self.clusterer.model.encoder(pixel_values)
                hidden_states = self.clusterer.model.quant_conv(encoder_output)
                _, _, info = self.clusterer.model.quantize(hidden_states)
                indices = info[2]
        
        # Get unique codebook indices
        indices = torch.unique(indices)
        
        # Get GNN embeddings for these indices
        with torch.no_grad():
            node_embeddings = self.gnn_model(indices)
        
        # Pool embeddings (mean pooling)
        image_embedding = node_embeddings.mean(dim=0)
        
        return image_embedding

def compute_clustering_loss(similarity_matrix, edge_index, batch_size, modularity_weight=0.5):
    """Compute a clustering regularization loss to encourage more distinct clusters"""
    device = similarity_matrix.device
    
    # Get edges that connect only to nodes in the batch
    mask = (edge_index[0] < batch_size) & (edge_index[1] < batch_size)
    batch_edge_index = edge_index[:, mask]
    
    # Skip if no edges in batch
    if batch_edge_index.shape[1] == 0:
        return torch.tensor(0.0, device=device)
    
    # Create adjacency matrix for the batch
    adj_matrix = torch.zeros((batch_size, batch_size), device=device)
    adj_matrix[batch_edge_index[0], batch_edge_index[1]] = 1.0
    adj_matrix[batch_edge_index[1], batch_edge_index[0]] = 1.0  # Make symmetric
    
    # Calculate node degrees
    degrees = adj_matrix.sum(dim=1)
    total_edges = degrees.sum() / 2
    
    if total_edges == 0:
        return torch.tensor(0.0, device=device)
    
    # Calculate modularity-like loss
    expected_connections = torch.outer(degrees, degrees) / (2 * total_edges)
    modularity_matrix = adj_matrix - modularity_weight * expected_connections
    
    # Calculate modularity Q using similarity as community assignments
    modularity_term = (modularity_matrix * similarity_matrix).sum()
    
    # Add a repulsion term for non-connected nodes
    neg_mask = 1.0 - adj_matrix
    neg_mask.fill_diagonal_(0)
    repulsion_term = (similarity_matrix * neg_mask).sum() / (neg_mask.sum() + 1e-8)
    
    # Combined loss
    combined_loss = -modularity_term + 1.5 * repulsion_term
    
    # Normalize the loss to keep it in a reasonable range
    # Divide by batch size to make it scale-invariant
    normalized_loss = combined_loss.abs() / batch_size
    
    return normalized_loss

def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    num_epochs,
    device,
    save_dir,
    contrastive_weight=1.0,
    classification_weight=0.0,
    temperature=0.05,
    margin=0.1,
    k_hard_negatives=32,
    use_cluster_loss=False,
    max_grad_norm=0.5,
    log_interval=10,
    clustering_weight=0.2,  # Add explicit clustering weight parameter
    use_pos_reg=True
):
    """Train the GNN model with subgraph sampling"""
    model.train()
    best_val_loss = float('inf')
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Add learning rate scheduler with warmup
    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = num_training_steps // 10  # 10% of total steps for warmup
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Early stopping
    patience = 10  # Reduced from 20 for faster convergence
    no_improve_epochs = 0
    
    # Add gradient clipping to prevent exploding gradients
    max_grad_norm = max_grad_norm
    
    # Track metrics for dynamic temperature adjustment
    avg_positive_sim = 0.0
    avg_negative_sim = 0.0
    dynamic_temperature = temperature
    
    for epoch in range(num_epochs):
        total_loss = 0
        contrastive_losses = 0
        classification_losses = 0
        num_batches = 0
        epoch_positive_sims = []
        epoch_negative_sims = []
        
        # Training loop
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Get batch size (number of central nodes)
            batch_size = train_loader.batch_size
            
            # Normalize edge attributes (always do this to ensure consistency)
            if batch.edge_attr is not None:
                # Scale edge attributes moderately
                batch.edge_attr = torch.clamp(batch.edge_attr, min=0.0)  # Ensure non-negative
                if batch.edge_attr.max() > 0:
                    # Scale to a reasonable range while preserving relative values
                    batch.edge_attr = batch.edge_attr / (batch.edge_attr.max() + 1e-8) 
            
            # Apply edge dropout for regularization
            if epoch > 0:
                edge_mask = torch.rand(batch.edge_index.size(1), device=device) > 0.02
                edge_index_dropout = batch.edge_index[:, edge_mask]
                if batch.edge_attr is not None:
                    edge_attr_dropout = batch.edge_attr[edge_mask]
                else:
                    edge_attr_dropout = None
            else:
                edge_index_dropout = batch.edge_index
                edge_attr_dropout = batch.edge_attr
            mask = (batch.edge_index[0] < batch_size) & (batch.edge_index[1] < batch_size)
            if mask.sum().item()==0:
                print("no connections, skip for this batch")
                continue
            # Forward pass only on sampled nodes
            node_embeddings = model(
                batch.x, edge_index_dropout, edge_attr_dropout, return_all=True
            )
            # Get embeddings for the central nodes
            central_embeddings = node_embeddings[:batch_size]
            
            # Compute normalized embeddings for similarity matrix
            norm_embeddings = F.normalize(central_embeddings, dim=1)
            sim_matrix = torch.matmul(norm_embeddings, norm_embeddings.T)
            
            # Track positive and negative similarities for temperature adjustment
            with torch.no_grad():
                # Create positive pair mask from edge_index
                pos_mask = torch.zeros((batch_size, batch_size), device=device)
                mask = (batch.edge_index[0] < batch_size) & (batch.edge_index[1] < batch_size)
                batch_edge_index = batch.edge_index[:, mask]
                if batch_edge_index.shape[1] > 0:
                    pos_mask[batch_edge_index[0], batch_edge_index[1]] = 1.0
                    pos_mask[batch_edge_index[1], batch_edge_index[0]] = 1.0
                
                pos_mask.fill_diagonal_(0)
                
                neg_mask = 1 - pos_mask
                neg_mask.fill_diagonal_(0)
                
                if pos_mask.sum() > 0:
                    pos_sim = sim_matrix[pos_mask > 0].mean().item()
                    epoch_positive_sims.append(pos_sim)
                
                if neg_mask.sum() > 0:
                    neg_sim = sim_matrix[neg_mask > 0].mean().item()
                    epoch_negative_sims.append(neg_sim)
            
            # Compute contrastive loss with hard negative mining and edge attributes
            """contrastive_loss = info_nce_loss_with_hard_negatives(
                central_embeddings,
                batch.edge_index,
                batch_size,
                batch.edge_attr,  # Pass edge attributes to the loss function
                temperature=dynamic_temperature,
                k_hard_negatives=k_hard_negatives,
                margin=margin
            )"""
            # use info_nce_loss:
            contrastive_loss = info_nce_loss(
                central_embeddings,
                batch.edge_index,
                batch_size,
                batch.edge_attr,
                temperature=dynamic_temperature,
                target_pos_sim=0.8,  # Target positive similarity (adjust as needed),
                use_pos_reg=use_pos_reg
            )
            
            # Extract loss components for logging
            nce_loss = getattr(contrastive_loss, 'nce_loss', torch.tensor(0.0))
            pos_sim_penalty = getattr(contrastive_loss, 'pos_sim_penalty', torch.tensor(0.0))
            pos_sim_avg = getattr(contrastive_loss, 'pos_sim_avg', torch.tensor(0.0))
            
            # Total contrastive loss with clustering regularization
            loss = contrastive_loss
            
            if loss.requires_grad:
                loss.backward()
                
                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                optimizer.step()
                scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            contrastive_losses += nce_loss.item() if isinstance(nce_loss, torch.Tensor) else 0.0
            num_batches += 1
            
            # Update progress bar with additional clustering loss info
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'nce_loss': f'{nce_loss.item() if isinstance(nce_loss, torch.Tensor) else 0.0:.4f}',
                'pos_sim': f'{pos_sim_avg.item() if isinstance(pos_sim_avg, torch.Tensor) else 0.0:.4f}',
                'reg_loss': f'{pos_sim_penalty.item() if isinstance(pos_sim_penalty, torch.Tensor) else 0.0:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}',
                'temp': f'{dynamic_temperature:.4f}'
            })
            
            if (batch_idx + 1) % log_interval == 0 and wandb.run is not None:
                wandb.log({
                    "train/total_loss": loss.item(),
                    "train/nce_loss": nce_loss.item() if isinstance(nce_loss, torch.Tensor) else 0.0,
                    "train/pos_sim_penalty": pos_sim_penalty.item() if isinstance(pos_sim_penalty, torch.Tensor) else 0.0,
                    "train/pos_sim_avg": pos_sim_avg.item() if isinstance(pos_sim_avg, torch.Tensor) else 0.0,
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "train/temperature": dynamic_temperature,
                    "train/batch": batch_idx + 1,
                    "train/epoch": epoch + 1,
                })
        
        # Update dynamic temperature based on similarity statistics
        if epoch_positive_sims and epoch_negative_sims:
            avg_positive_sim = sum(epoch_positive_sims) / len(epoch_positive_sims)
            avg_negative_sim = sum(epoch_negative_sims) / len(epoch_negative_sims)
            sim_gap = avg_positive_sim - avg_negative_sim
            
            # More aggressive temperature adjustment
            if sim_gap < 0.3:  # Increased threshold
                # Decrease temperature more slowly and maintain higher minimum
                dynamic_temperature = max(0.05, dynamic_temperature * 0.95)
            elif sim_gap > 0.5:
                # Increase temperature more aggressively when gap is large
                dynamic_temperature = min(0.15, dynamic_temperature * 1.1)
            
            # More aggressive correction when negative similarity approaches positive
            if avg_positive_sim - avg_negative_sim < 0.1:  # Dangerous zone
                # Increase temperature significantly to prevent collapse
                dynamic_temperature = min(0.2, dynamic_temperature * 1.5)
            
            # Track the trend of negative similarity
            if batch_idx == 0:  # First batch of epoch
                if hasattr(model, 'prev_neg_sim'):
                    neg_sim_increasing = avg_negative_sim > model.prev_neg_sim
                    if neg_sim_increasing and avg_negative_sim > 0.5:  # High and increasing negative similarity
                        # Add diversity regularization weight
                        effective_clustering_weight *= 1.5  # Increase clustering weight to encourage diversity
                model.prev_neg_sim = avg_negative_sim
            
            if wandb.run is not None:
                wandb.log({
                    "train/avg_positive_sim": avg_positive_sim,
                    "train/avg_negative_sim": avg_negative_sim,
                    "train/sim_gap": sim_gap,
                    "train/epoch": epoch + 1
                })
        
        # Log epoch metrics
        avg_loss = total_loss / num_batches
        avg_contrastive = contrastive_losses / num_batches

        logger.info(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Loss: {avg_loss:.4f} "
            f"(Contrastive: {avg_contrastive:.4f}) "
            f"Temp: {dynamic_temperature:.4f} "
            f"Pos/Neg Sim: {avg_positive_sim:.4f}/{avg_negative_sim:.4f}"
        )
        
        if wandb.run is not None:
            wandb.log({
                "train/epoch_loss": avg_loss,
                "train/epoch_contrastive": avg_contrastive,
                "train/epoch": epoch + 1
            })
        
        # Validation
        model.eval()
        val_loss = 0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                batch_size = val_loader.batch_size
                
                node_embeddings = model(
                    batch.x, batch.edge_index, batch.edge_attr, return_all=True
                )
                
                central_embeddings = node_embeddings[:batch_size]
                
                contrastive_loss = info_nce_loss(
                    central_embeddings,
                    batch.edge_index,
                    batch_size,
                    batch.edge_attr,
                    temperature=dynamic_temperature,
                    target_pos_sim=0.8  # Target positive similarity (adjust as needed)
                )
                
                loss = contrastive_loss
                val_loss += loss.item()
                num_val_batches += 1
        
        val_loss /= max(num_val_batches, 1)
        logger.info(f"Validation loss: {val_loss:.4f}")
        
        if wandb.run is not None:
            wandb.log({
                "val/loss": val_loss,
                "val/epoch": epoch + 1
            })
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
            
            # Save model
            model_path = save_dir / "best_gnn_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
            }, model_path)
            
            # Compute and save node representations for entire codebook
            model.eval()
            with torch.no_grad():
                all_embeddings = []
                for batch in tqdm(val_loader, desc="Computing embeddings"):
                    batch = batch.to(device)
                    batch_size = val_loader.batch_size
                    node_embeddings = model(batch.x, batch.edge_index, batch.edge_attr)
                    all_embeddings.append(node_embeddings[:batch_size].cpu())
                
                node_embeddings = torch.cat(all_embeddings, dim=0)
                
                # Save embeddings
                torch.save({
                    'node_embeddings': node_embeddings,
                    'epoch': epoch,
                }, save_dir / "node_embeddings.pt")
            
            logger.info(f"Saved best model and node embeddings at epoch {epoch+1}")
        else:
            no_improve_epochs += 1
            
        # Early stopping
        if no_improve_epochs >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        model.train()

def main():
    parser = argparse.ArgumentParser(description="Train GNN for codebook clustering")
    parser.add_argument("--model_path", required=True, help="Path to the model")
    parser.add_argument("--model_type", choices=["chameleon", "llamagen", "janus", "Emu3"],
                       default="chameleon", help="Type of model to use")
    parser.add_argument("--graph_data_dir", required=True, help="Directory containing graph dataset")
    parser.add_argument("--save_dir", required=True, help="Directory to save trained model and embeddings")
    parser.add_argument("--imagenet_root", required=False, help="Root path to ImageNet dataset")
    parser.add_argument("--batch_size", type=int, default=2048, help="Number of nodes per batch")
    parser.add_argument("--num_neighbors", type=int, default=[48, 16], nargs='+', 
                       help="Number of neighbors to sample for each node in each layer")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=4e-3, help="Learning rate")  # Reduced initial learning rate
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--output_dim", type=int, default=256, help="Output dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of GNN layers")
    parser.add_argument("--num_heads", type=int, default=2, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--use_cluster_loss", type=bool, default=False, help="Use clustering loss")
    parser.add_argument("--contrastive_weight", type=float, default=1.0,
                       help="Weight for contrastive loss")
    parser.add_argument("--classification_weight", type=float, default=1.0,
                       help="Weight for classification loss")
    parser.add_argument("--temperature", type=float, default=0.02,
                       help="Temperature for contrastive loss")
    parser.add_argument("--margin", type=float, default=0.1,
                       help="Margin for contrastive loss")
    parser.add_argument("--k_hard_negatives", type=int, default=128,
                       help="Number of hard negatives to use")
    parser.add_argument("--wandb_project", type=str, default="codebook-gnn",
                       help="W&B project name")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, 
                       help="Max gradient norm for clipping")
    parser.add_argument("--edge_dropout", type=float, default=0.1,
                       help="Probability of dropping edges during training")
    parser.add_argument("--dynamic_temp", action="store_true",
                       help="Enable dynamic temperature adjustment")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay for optimizer")
    parser.add_argument("--clustering_weight", type=float, default=0.2,
                       help="Weight for clustering regularization")
    parser.add_argument("--discard_low_cooccurrence", action="store_true",
                       help="Discard 10%% of edges with lowest co-occurrence scores")
    parser.add_argument("--discard_percentage", type=float, default=0.3,
                       help="Percentage of low co-occurrence edges to discard (default: 0.1)")
    parser.add_argument("--use_pos_reg", type=bool, default=True, help="Use Positive pair regularization")
    parser.add_argument("--gnn_type", type=str, choices=['gat', 'gatv2', 'sage'], default='gatv2',
                       help="Type of GNN architecture to use (default: gatv2)")
    
    args = parser.parse_args()
    
    # Initialize wandb
    wandb.init(project=args.wandb_project, config=args)
    
    # Load graph dataset
    graph_dataset = CodebookGraphDataset.load(args.graph_data_dir)
    
    # Filter out edges with low co-occurrence scores if requested
    if args.discard_low_cooccurrence:
        # Get the current graph components
        edge_index = graph_dataset.edge_index
        edge_attr = graph_dataset.edge_attr
        
        logger.info(f"Original graph has {edge_index.size(1)} edges")
        
        # Get the edge scores (use first dimension if multi-dimensional)
        edge_scores = edge_attr
        if edge_scores.dim() > 1:
            edge_scores = edge_scores[:, 0]
            
        # Determine threshold to keep top (1-discard_percentage) of edges
        k = int(edge_index.size(1) * (1 - args.discard_percentage))
        threshold_value, _ = torch.kthvalue(edge_scores, k)
        
        # Filter edges
        mask = edge_scores >= threshold_value
        
        # Update graph dataset attributes directly
        graph_dataset.edge_index = edge_index[:, mask]
        graph_dataset.edge_attr = edge_attr[mask]
        
        # If edge_types exists, update it too
        if hasattr(graph_dataset, 'edge_types'):
            graph_dataset.edge_types = graph_dataset.edge_types[mask]
        
        logger.info(f"Filtered graph has {graph_dataset.edge_index.size(1)} edges after removing {args.discard_percentage:.1%} lowest co-occurrence edges")
        logger.info(f"Co-occurrence score threshold: {threshold_value.item():.6f}")
        
        if wandb.run is not None:
            wandb.log({
                "graph/original_edges": edge_index.size(1),
                "graph/filtered_edges": graph_dataset.edge_index.size(1),
                "graph/edge_threshold": threshold_value.item()
            })
    
    # Create a single graph from the dataset
    graph = graph_dataset.get(0)  # Get the single graph
    
    train_loader = NeighborLoader(
        graph,  # Use the graph directly
        num_neighbors=args.num_neighbors,  # Number of neighbors to sample for each layer
        batch_size=args.batch_size,  # Number of nodes per batch
        shuffle=True,
        num_workers=4
    )
    
    val_loader = NeighborLoader(
        graph,  # Use the graph directly
        num_neighbors=args.num_neighbors,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CodebookGNN(
        input_dim=graph_dataset.node_features.size(1),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        output_dim=args.output_dim,
        gnn_type=args.gnn_type
    ).to(device)
    
    # Optimizer with weight decay (L2 regularization)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Train model
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        device=device,
        save_dir=args.save_dir,
        contrastive_weight=args.contrastive_weight,
        classification_weight=args.classification_weight,
        temperature=args.temperature,
        margin=args.margin,
        k_hard_negatives=args.k_hard_negatives,
        clustering_weight=args.clustering_weight,
        use_cluster_loss=args.use_cluster_loss,
        max_grad_norm=args.max_grad_norm,
        use_pos_reg=args.use_pos_reg
    )
    
    wandb.finish()

if __name__ == "__main__":
    main() 
