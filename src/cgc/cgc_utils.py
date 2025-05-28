import torch
import numpy as np
import os
import pickle
from tqdm import tqdm
import torch_geometric
from pathlib import Path
import logging
from janus.models import MultiModalityCausalLM, VLChatProcessor
from emu3.mllm.processing_emu3 import Emu3Processor
from transformers import ChameleonForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer, AutoImageProcessor, AutoModel
from src.cgc.coco_panoptic import load_coco_panoptic_data, get_panoptic_image_and_masks
from PIL import Image
from src.model_utils import load_model_and_processor
from torch_geometric.data import Data, Dataset, DataLoader

COMMON_CATEGORY = ["wall-brick","wall-stone","wall-tile","wall-wood","water-other","window-blind","window-other","tree-merged","fence-merged","ceiling-merged", "sky-other-merged", "cabinet-merged", "table-merged",
                   "floor-other-merged","pavement-merged", "mountain-merged", "grass-merged", "dirt-merged", "paper-merged", "food-other-merged", "building-other-merged", "rock-merged", "wall-other-merged",
                   "rug-merged"]

# define logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodebookGraphDataset(torch_geometric.data.Dataset):
    """Dataset for creating graph representations of codebook entries"""
    
    def __init__(
        self,
        codebook=None,
        coco_root=None,
        model_type="chameleon",
        processor=None,
        transform=None,
        top_k_edges=0.02,
        clusterer=None,
        save_dir=None,
        memory_efficient=True,
        not_use_seg=False,
        save_interval=100  # Save to disk every N images
    ):
        # Initialize parent class first
        super().__init__()
        
        # If codebook is None, this is a loading operation
        if codebook is None:
            return
            
        self.codebook = torch.from_numpy(codebook).float()
        self.num_embeddings, self.embedding_dim = codebook.shape
        self.model_type = model_type
        self.processor = processor
        self.save_dir = Path(save_dir) if save_dir else None
        self._indices = None
        self.memory_efficient = memory_efficient
        self.save_interval = save_interval
        self.not_use_seg = not_use_seg
        
        # Create save directory if it doesn't exist
        if self.save_dir and self.memory_efficient:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.temp_stats_dir = self.save_dir / "temp_segment_stats"
            self.temp_stats_dir.mkdir(exist_ok=True)
        
        # Load COCO panoptic dataset
        panoptic_json = os.path.join(coco_root, "annotations/panoptic_val2017.json")
        image_dir = os.path.join(coco_root, "val2017")
        panoptic_dir = os.path.join(coco_root, "annotations/panoptic_val2017")
        
        self.panoptic_coco, self.categories_dict = load_coco_panoptic_data(
            panoptic_json, image_dir, panoptic_dir
        )
        
        # Store paths for later use
        self.image_dir = image_dir
        self.panoptic_dir = panoptic_dir
        
        # Initialize co-occurrence matrix and segment statistics
        self.co_occurrence = np.zeros((self.num_embeddings, self.num_embeddings))
        
        # Initialize segment statistics array
        # Get all unique category IDs from the categories dictionary
        self.category_ids = sorted(set(self.categories_dict.keys()))
        self.num_categories = len(self.category_ids)
        self.seg_statistics = np.zeros((self.num_embeddings, self.num_categories), dtype=np.int32)
        
        # In memory-efficient mode, we'll use a smaller buffer and periodically save to disk
        if self.memory_efficient:
            self.segment_stats_buffer = {}  # Temporary buffer
            self.processed_images_count = 0
        else:
            self.segment_stats = {}  # Will store {(token1, token2): {category: count}}
        
        if clusterer is None:
            raise ValueError("Clusterer is required for computing co-occurrences")
            
        # Compute co-occurrences before building graph
        logger.info("Computing co-occurrences...")
        self._compute_co_occurrences(clusterer)
        
        # In memory-efficient mode, merge all temporary files
        if self.memory_efficient:
            logger.info("Merging temporary segment statistics files...")
            self._merge_segment_stats()
        
        # Build graph structure
        self._build_graph( top_k_edges)
        
        # Save graph data if save_dir is provided
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.save_graph_data()
            
            # Clean up temporary files if in memory-efficient mode
            if self.memory_efficient and self.temp_stats_dir.exists():
                import shutil
                shutil.rmtree(self.temp_stats_dir)
                logger.info(f"Cleaned up temporary segment statistics files")
    
    def _get_2d_position(self, flattened_idx, height, width):
        """Convert flattened index to 2D position"""
        row = flattened_idx // width
        col = flattened_idx % width
        return row, col

    def _in_same_segment(self, pos1, pos2,height,width, panoptic_seg_id, segments_info):
        """Check if two positions are in the same segment by considering overlapping patches"""
        row1, col1 = pos1
        row2, col2 = pos2
        
        # Get original image dimensions
        img_height, img_width = panoptic_seg_id.shape
        
        
        # Calculate patch size in original image
        patch_height = img_height / height
        patch_width = img_width / width
        
        # Calculate patch boundaries in original image coordinates
        # For position 1
        start_row1 = max(0, int(row1 * patch_height))
        end_row1 = min(img_height, int((row1 + 1) * patch_height))
        start_col1 = max(0, int(col1 * patch_width))
        end_col1 = min(img_width, int((col1 + 1) * patch_width))
        
        # For position 2
        start_row2 = max(0, int(row2 * patch_height))
        end_row2 = min(img_height, int((row2 + 1) * patch_height))
        start_col2 = max(0, int(col2 * patch_width))
        end_col2 = min(img_width, int((col2 + 1) * patch_width))
        
        # Extract segment IDs for both patches
        patch1_segments = panoptic_seg_id[start_row1:end_row1, start_col1:end_col1].flatten()
        patch2_segments = panoptic_seg_id[start_row2:end_row2, start_col2:end_col2].flatten()
        
        # Find most common segment ID in each patch
        from collections import Counter
        most_common_segment1 = Counter(patch1_segments).most_common(1)[0][0]
        most_common_segment2 = Counter(patch2_segments).most_common(1)[0][0]
        
        # If they're in the same segment, return the category info
        if most_common_segment1 == most_common_segment2:
            for segment in segments_info:
                if segment['id'] == most_common_segment1:
                    category = self.categories_dict[segment['category_id']]
                    return True, category['name']
        return False, None
    
    def _in_same_grid(self, pos1, pos2, height, width):
        """Check if two positions are in the same grid of a 3x3 boxes
        
        Args:
            pos1 (tuple): First position as (row, col)
            pos2 (tuple): Second position as (row, col)
            height (int): Total height of the area
            width (int): Total width of the area
            
        Returns:
            bool: True if positions are in the same grid, False otherwise
        """
        row1, col1 = pos1
        row2, col2 = pos2
        
        grid_row1, grid_col1 = row1 // 3, col1 // 3
        grid_row2, grid_col2 = row2 // 3, col2 // 3


        return (grid_row1 == grid_row2) and (grid_col1 == grid_col2)

    def _save_segment_stats_buffer(self):
        """Save the current segment statistics buffer to disk"""
        if not self.memory_efficient or not self.save_dir:
            return
            
        # Create a unique filename based on the current count
        filename = self.temp_stats_dir / f"segment_stats_{self.processed_images_count}.pkl"
        
        # Save the current buffer to disk
        with open(filename, 'wb') as f:
            pickle.dump(self.segment_stats_buffer, f)
            
        # Clear the buffer
        self.segment_stats_buffer = {}
        logger.debug(f"Saved segment statistics buffer to {filename}")

    def _merge_segment_stats(self):
        """Merge all temporary segment statistics files into a single dictionary"""
        if not self.memory_efficient or not self.save_dir:
            return
            
        # Initialize the merged dictionary
        self.segment_stats = {}
        
        # Get all temporary files
        temp_files = list(self.temp_stats_dir.glob("segment_stats_*.pkl"))
        
        if not temp_files:
            logger.warning("No temporary segment statistics files found to merge")
            return
            
        logger.info(f"Merging {len(temp_files)} temporary segment statistics files...")
        
        # Load and merge each file
        for file_path in tqdm(temp_files):
            try:
                with open(file_path, 'rb') as f:
                    temp_stats = pickle.load(f)
                    
                # Merge into the main dictionary
                for pair_key, category_counts in temp_stats.items():
                    if pair_key not in self.segment_stats:
                        self.segment_stats[pair_key] = {}
                        
                    for category, count in category_counts.items():
                        if category not in self.segment_stats[pair_key]:
                            self.segment_stats[pair_key][category] = 0
                        self.segment_stats[pair_key][category] += count
            except Exception as e:
                logger.warning(f"Error merging file {file_path}: {e}")
                
        logger.info(f"Merged segment statistics: {len(self.segment_stats)} token pairs")

    def _compute_co_occurrences(self, clusterer):
        """Compute co-occurrence matrix using COCO panoptic dataset"""
        logger.info("Computing co-occurrence matrix...")
        device = next(clusterer.model.parameters()).device if self.model_type != "Emu3" else clusterer.vision_tokenizer.device
        model_dtype = next(clusterer.model.parameters()).dtype if self.model_type != "Emu3" else clusterer.vision_tokenizer.dtype
        
        # Get list of images and sample 10% if using Emu3
        images = self.panoptic_coco['images']

        #if self.model_type == "Emu3":
        #    num_samples = max(1, int(0.2 * len(images)))
        #     images = np.random.choice(images, size=num_samples, replace=False)
        #    logger.info(f"Using {num_samples} images (10%) for Emu3 model")
        
        
        
        # Process all images in the dataset
        for img_idx, img_id in enumerate(tqdm(images)):
            try:
                # Get image and panoptic segmentation data
                img, segments_info, _, panoptic_seg_id = get_panoptic_image_and_masks(
                    self.panoptic_coco, 
                    self.categories_dict, 
                    img_id['id'], 
                    self.image_dir, 
                    self.panoptic_dir
                )
                
                
                # Convert PIL Image to tensor based on model type
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                
                # Process image based on model type
                if self.model_type == "chameleon":
                    # Add a dummy text prompt since the processor requires it
                    dummy_prompt = "<image>"
                    inputs = self.processor(text=dummy_prompt, images=img, return_tensors="pt")
                    pixel_values = inputs["pixel_values"].to(device, dtype=model_dtype)
                    
                    # Get codebook indices for this image
                    with torch.no_grad():
                        encoder_output = clusterer.model.model.vqmodel.encoder(pixel_values)
                        hidden_states = clusterer.model.model.vqmodel.quant_conv(encoder_output)
                        _, _, indices = clusterer.model.model.vqmodel.quantize(hidden_states)
                        indices = indices.cpu().numpy().flatten()
                        height = width = int(np.sqrt(len(indices)))  # Assuming square spatial dimensions
                        
                elif self.model_type == "janus":
                    # Process image for Janus
                    conversation = [
                        {
                            "role": "<|User|>",
                            "content": "<image_placeholder>",
                            "images": [img],
                        },
                        {"role": "<|Assistant|>", "content": ""},
                    ]
                    inputs = self.processor(
                        conversations=conversation, 
                        images=[img], 
                        force_batchify=True
                    )
                    pixel_values = inputs["pixel_values"].squeeze(0).to(device, dtype=model_dtype)
                    
                    # Get codebook indices for this image
                    with torch.no_grad():
                        vqmodel = clusterer.model.gen_vision_model
                        encoder_output = vqmodel.encoder(pixel_values)
                        hidden_states = vqmodel.quant_conv(encoder_output)
                        _, _, info = vqmodel.quantize(hidden_states)
                        indices = info[2].cpu().numpy().flatten()
                        height = width = int(np.sqrt(len(indices)))  # Assuming square spatial dimensions
                        
                elif self.model_type == "Emu3":
                    # Process image for Emu3
                    if isinstance(img, Image.Image):
                        img = [img]
                    pixel_values = self.processor.image_processor(img, return_tensors="pt")["pixel_values"]
                    pixel_values = pixel_values.to(device, dtype=model_dtype)
                    
                    # Get codebook indices for this image
                    with torch.no_grad():
                        ndim = pixel_values.ndim
                        if ndim == 4:
                            t = clusterer.vision_tokenizer.config.temporal_downsample_factor
                            b, c, h, w = pixel_values.shape
                            pixel_values = pixel_values.unsqueeze(1).repeat(1, t, 1, 1, 1)
                        elif ndim == 5:
                            b, t, c, h, w = pixel_values.shape
                        
                        encoder_output = clusterer.vision_tokenizer.encoder(pixel_values)
                        # b t c h w -> b c t h w
                        encoder_output = encoder_output.permute(0, 2, 1, 3, 4)
                        hidden_state = clusterer.vision_tokenizer.quant_conv(encoder_output)
                        # b c t h w -> b t c h w
                        hidden_state = hidden_state.permute(0, 2, 1, 3, 4)
                        indices = clusterer.vision_tokenizer.quantize(hidden_state)
                        
                        if ndim == 4:
                            indices = indices.squeeze(1)
                        
                        # Get height and width from indices shape
                        height, width = indices.shape[1:]  # indices shape is [batch, height, width]
                        indices = indices.cpu().numpy().flatten()
                        
                else:  # llamagen
                    pixel_values = self.transform(img).unsqueeze(0).to(device, dtype=model_dtype)
                    
                    # Get codebook indices for this image
                    with torch.no_grad():
                        encoder_output = clusterer.model.encoder(pixel_values)
                        hidden_states = clusterer.model.quant_conv(encoder_output)
                        _, _, info = clusterer.model.quantize(hidden_states)
                        indices = info[2].cpu().numpy().flatten()
                        height = width = int(np.sqrt(len(indices)))  # Assuming square spatial dimensions
                
                # Update co-occurrence matrix with segment constraints
                for i in range(len(indices)):
                    pos1 = self._get_2d_position(i, height, width)
                    for j in range(i + 1, len(indices)):
                        pos2 = self._get_2d_position(j, height, width)
                        same_segment = False
                        category = None
                        # Check if positions are in same segment
                        if self._in_same_grid(pos1, pos2, height, width):
                            
                            
                            token1, token2 = indices[i], indices[j]
                            # Update co-occurrence matrix
                            self.co_occurrence[token1, token2] += 1
                            self.co_occurrence[token2, token1] += 1
                            #same_segment, category = self._in_same_segment(pos1, pos2, height, width,  panoptic_seg_id, segments_info)
                            # Update segment statistics
                            #if same_segment and category not in COMMON_CATEGORY:
                            #    category_id = next((k for k, v in self.categories_dict.items() if v['name'] == category), None)
                            #    if category_id is not None:
                            #        category_idx = self.category_ids.index(category_id)
                            #         self.seg_statistics[token1, category_idx] += 1
                            #        self.seg_statistics[token2, category_idx] += 1
                        else:
                            if not self.not_use_seg:
                                same_segment, category = self._in_same_segment(pos1, pos2, height, width,  panoptic_seg_id, segments_info)
                                if same_segment and category not in COMMON_CATEGORY:
                                    token1, token2 = indices[i], indices[j]
                                    self.co_occurrence[token1, token2] += 1
                                    self.co_occurrence[token2, token1] += 1

                                    
                            


                # In memory-efficient mode, periodically save buffer to disk
                if self.memory_efficient:
                    self.processed_images_count += 1
                    if self.processed_images_count % self.save_interval == 0:
                        self._save_segment_stats_buffer()
                        
            except Exception as e:
                logger.warning(f"Error processing image {img_id['id']}: {e}")
                continue
                
        # Save any remaining stats in the buffer
        if self.memory_efficient and self.segment_stats_buffer:
            self._save_segment_stats_buffer()

    def _build_graph(self, top_k_edges):
        """Build graph structure with similarity and co-occurrence edges"""
        logger.info("Building graph structure...")
        
        
        
        # Create edges based on top co-occurrences
        # Get indices of upper triangle (including diagonal) to avoid duplicates
        triu_indices = np.triu_indices(self.co_occurrence.shape[0])
        triu_values = self.co_occurrence[triu_indices]
        
        # Get non-zero values from upper triangle
        nonzero_mask = triu_values > 0
        nonzero_values = triu_values[nonzero_mask]
        nonzero_rows = triu_indices[0][nonzero_mask]
        nonzero_cols = triu_indices[1][nonzero_mask]
        
        # Sort by value
        sorted_indices = np.argsort(nonzero_values)[::-1]
        
        # Keep top p% of actual co-occurrences
        k = int(top_k_edges * len(nonzero_values))
        top_k_sorted_indices = sorted_indices[:k]
        
        # Get row and column indices for top k co-occurrences
        row_indices = nonzero_rows[top_k_sorted_indices]
        col_indices = nonzero_cols[top_k_sorted_indices]
        
        # Add symmetric edges (fixed to avoid dimension mismatch)
        original_row_indices = row_indices.copy()
        original_col_indices = col_indices.copy()
        
        row_indices = np.concatenate([original_row_indices, original_col_indices])
        col_indices = np.concatenate([original_col_indices, original_row_indices])
        
        cooccurrence_edge_index = torch.tensor(
            np.vstack([row_indices, col_indices]),
            dtype=torch.long
        )
        
        # Normalize co-occurrence weights to [0,1] range
        cooccurrence_weights = self.co_occurrence[row_indices, col_indices]
        if cooccurrence_weights.max() > cooccurrence_weights.min():  # Avoid division by zero
            cooccurrence_weights = (cooccurrence_weights - cooccurrence_weights.min()) / (cooccurrence_weights.max() - cooccurrence_weights.min())
        cooccurrence_edge_weights = torch.tensor(cooccurrence_weights, dtype=torch.float)
        
        
        
        # Only use co-occurrence edges
        self.edge_index = cooccurrence_edge_index
        self.edge_attr = cooccurrence_edge_weights
        
        # All edges are co-occurrence type
        self.edge_types = torch.ones(cooccurrence_edge_index.shape[1], dtype=torch.long)
        
        # Store node features (codebook vectors)
        self.node_features = self.codebook
        
        # Filter segment_stats to only keep pairs that are in the graph
        
        edge_pairs = set()
        for i in range(self.edge_index.shape[1]):
            src, dst = int(self.edge_index[0, i]), int(self.edge_index[1, i])
            edge_pairs.add(tuple(sorted([src, dst])))
        
        # Only keep statistics for edges that are in the graph
        filtered_segment_stats = {
            pair: stats for pair, stats in self.segment_stats.items()
            if pair in edge_pairs
        }

        self.segment_stats = filtered_segment_stats
        if self.include_similarity_edges:
            logger.info(f"Graph built with {self.edge_index.size(1)} edges "
                    f"({similarity_edge_index.size(1)} similarity edges, "
                    f"{cooccurrence_edge_index.size(1)} co-occurrence edges)")
        else:
            logger.info(f"Graph built with {self.edge_index.size(1)} co-occurrence edges "
                    f"(similarity edges disabled)")
    
    def len(self):
        """Return the number of graphs in the dataset"""
        return 1  # We only have one graph
    
    def get(self, idx):
        """Return the graph at index idx"""
        if idx != 0:
            raise IndexError(f"Dataset contains only one graph, but index {idx} was requested")
            
        return Data(
            x=self.node_features,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            num_nodes=self.num_embeddings
        )
    
    @property
    def indices(self):
        """Return the indices of the dataset"""
        if self._indices is None:
            self._indices = range(self.len())
        return self._indices
    
    def set_indices(self, value):
        """Set the indices of the dataset"""
        self._indices = value

    def visualize_graph(self, output_path, max_nodes=500, min_edge_weight=0.1, height="750px", width="1200px"):
        """
        Visualize the graph using PyVis with interactive features
        
        Args:
            output_path (str): Path to save the HTML visualization
            max_nodes (int): Maximum number of nodes to visualize (to avoid overcrowding)
            min_edge_weight (float): Minimum edge weight to include in visualization
            height (str): Height of the visualization
            width (str): Width of the visualization
        """
        logger.info("Creating graph visualization...")
        
        # Create NetworkX graph from edge_index and edge_attr
        G = nx.Graph()
        
        # Add nodes with features
        node_features = self.node_features.numpy()
        
        # Use t-SNE to get 2D positions for nodes
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42)
        node_positions = tsne.fit_transform(node_features)
        
        # Scale positions to reasonable values
        scaler = MinMaxScaler(feature_range=(-100, 100))
        node_positions = scaler.fit_transform(node_positions)
        
        # Calculate node sizes based on degree
        edge_index = self.edge_index.numpy()
        degrees = np.zeros(self.num_embeddings)
        for i in range(edge_index.shape[1]):
            degrees[edge_index[0, i]] += 1
            degrees[edge_index[1, i]] += 1
        
        # Scale node sizes
        node_sizes = MinMaxScaler(feature_range=(10, 50)).fit_transform(degrees.reshape(-1, 1)).flatten()
        
        # Add nodes to graph with explicit integer conversion
        for i in range(min(max_nodes, self.num_embeddings)):
            node_id = int(i)  # Ensure node ID is integer
            G.add_node(node_id, 
                      size=int(node_sizes[i]),
                      x=float(node_positions[i, 0]),  # Ensure coordinates are float
                      y=float(node_positions[i, 1]),
                      title=f"Node {node_id}\nDegree: {int(degrees[i])}")
        
        # Add edges to graph with explicit integer conversion
        edge_weights = self.edge_attr.numpy()
        edge_types = self.edge_types.numpy()
        for i in range(edge_index.shape[1]):
            src, dst = int(edge_index[0, i]), int(edge_index[1, i])  # Ensure edge endpoints are integers
            if src < max_nodes and dst < max_nodes and edge_weights[i] >= min_edge_weight:
                # Different colors for similarity vs co-occurrence edges
                edge_color = "#848484" if edge_types[i] == 0 else "#ff7f0e"  # Gray for similarity, orange for co-occurrence
                G.add_edge(src, dst, 
                          weight=float(edge_weights[i]),  # Ensure weight is float
                          color=edge_color,
                          title=f"{'Similarity' if edge_types[i] == 0 else 'Co-occurrence'}: {edge_weights[i]:.3f}")
        
        # Create PyVis network
        nt = Network(height=height, width=width, notebook=False)
        
        # Configure physics
        nt.barnes_hut(gravity=-2000, central_gravity=0.3, spring_length=200)
        
        # Add nodes and edges from NetworkX graph
        nt.from_nx(G)
        
        # Configure visualization options
        nt.toggle_physics(True)
        nt.show_buttons(filter_=['physics'])
        
        # Add custom styling with legend
        nt.set_options("""
        var options = {
            "nodes": {
                "font": {
                    "size": 12
                },
                "borderWidth": 2,
                "borderWidthSelected": 4,
                "color": {
                    "border": "#2B7CE9",
                    "background": "#97C2FC",
                    "highlight": {
                        "border": "#2B7CE9",
                        "background": "#D2E5FF"
                    }
                }
            },
            "edges": {
                "smooth": {
                    "type": "continuous"
                },
                "width": 2
            },
            "physics": {
                "stabilization": {
                    "iterations": 100
                }
            },
            "legend": {
                "enabled": true,
                "useGraphSettings": false,
                "position": "right",
                "width": 0.1,
                "align": "right"
            }
        }
        """)
        
        # Save visualization
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nt.save_graph(str(output_path))
        logger.info(f"Graph visualization saved to {output_path}")

    def save_graph_data(self):
        """Save graph data including segment statistics"""
        save_dir = Path(self.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save tensor data
        tensor_data = {
            'node_features': self.node_features.half(),
            'edge_index': self.edge_index,
            'edge_attr': self.edge_attr.half(),
            'edge_types': self.edge_types,
            'num_embeddings': self.num_embeddings,
            'embedding_dim': self.embedding_dim,
            'model_type': self.model_type,
            'seg_statistics': self.seg_statistics,  # Save segment statistics array
            'category_ids': self.category_ids,  # Save category IDs for reference
            'include_similarity_edges': self.include_similarity_edges,
            'total_co_occurrences': sum(sum(cats.values()) for cats in self.segment_stats.values()),
            'unique_categories': len(set(cat for stats in self.segment_stats.values() 
                                       for cat in stats.keys())),
        }
        torch.save(tensor_data, save_dir / 'graph_data.pt')
        
        # Save segment statistics
        segment_stats_path = save_dir / 'segment_statistics.json'
        
        # Convert tuple keys to strings for JSON serialization
        serializable_stats = {}
        for (token1, token2), category_counts in self.segment_stats.items():
            # Sort tokens to ensure consistent key format
            key = f"{min(token1, token2)}_{max(token1, token2)}"
            serializable_stats[key] = {
                'category_counts': category_counts,
                'total_co_occurrences': sum(category_counts.values()),
                'categories': list(category_counts.keys())
            }
        
        with open(segment_stats_path, 'w') as f:
            json.dump(serializable_stats, f, indent=2)
        
        # Save metadata
        metadata = {
            'num_nodes': self.num_embeddings,
            'num_edges': self.edge_index.size(1),
            'num_similarity_edges': (self.edge_types == 0).sum().item() if self.include_similarity_edges else 0,
            'num_cooccurrence_edges': (self.edge_types == 1).sum().item(),
            'include_similarity_edges': self.include_similarity_edges,
            'total_co_occurrences': sum(sum(cats.values()) for cats in self.segment_stats.values()),
            'unique_categories': len(set(cat for stats in self.segment_stats.values() 
                                       for cat in stats.keys())),
            'model_type': self.model_type,
            'memory_efficient': self.memory_efficient
        }
        
        with open(save_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Graph data saved to {save_dir}")
        logger.info(f"Saved {len(self.segment_stats)} token pair statistics")
        logger.info(f"Total co-occurrences: {metadata['total_co_occurrences']}")
        logger.info(f"Unique semantic categories: {metadata['unique_categories']}")

    @classmethod
    def load(cls, save_dir):
        
        """Load a saved graph dataset"""
        save_dir = Path(save_dir)
        
        # Create instance with minimal initialization
        dataset = cls()
        
        # Initialize required attributes
        dataset._indices = None
        dataset.save_dir = save_dir
        dataset.memory_efficient = False  # Set to False for loaded datasets
        dataset.include_similarity_edges = True  # Default value
        
        # Load tensor data
        try:
            tensor_data = torch.load(save_dir / 'graph_data.pt')
            # Convert half precision tensors back to float
            for key, value in tensor_data.items():
                if isinstance(value, torch.Tensor) and value.dtype == torch.half:
                    value = value.float()
                setattr(dataset, key, value)
        except Exception as e:
            logger.error(f"Error loading tensor data: {e}")
            raise
        
        # Load segment statistics
        try:
            with open(save_dir / 'segment_statistics.json', 'r') as f:
                serialized_stats = json.load(f)
                
            # Convert string keys back to tuples
            dataset.segment_stats = {}
            for key, value in serialized_stats.items():
                token1, token2 = map(int, key.split('_'))
                dataset.segment_stats[(token1, token2)] = value['category_counts']
        except Exception as e:
            logger.warning(f"Could not load segment statistics: {e}")
            dataset.segment_stats = {}
        
        # Load metadata
        try:
            with open(save_dir / 'metadata.json', 'r') as f:
                metadata = json.load(f)
                
            # Set attributes from metadata
            dataset.include_similarity_edges = metadata.get('include_similarity_edges', True)
            dataset.memory_efficient = metadata.get('memory_efficient', False)
        except Exception as e:
            logger.warning(f"Could not load metadata: {e}")
        
        # Initialize required attributes if not loaded
        if not hasattr(dataset, 'num_embeddings'):
            dataset.num_embeddings = dataset.node_features.size(0)
        if not hasattr(dataset, 'embedding_dim'):
            dataset.embedding_dim = dataset.node_features.size(1)
        
        logger.info(f"Graph data loaded from {save_dir}")
        if dataset.segment_stats:
            logger.info(f"Loaded {len(dataset.segment_stats)} token pair statistics")
            total_co_occurrences = sum(sum(cats.values()) for cats in dataset.segment_stats.values())
            unique_categories = len(set(cat for stats in dataset.segment_stats.values() 
                                      for cat in stats.keys()))
            logger.info(f"Total co-occurrences: {total_co_occurrences}")
            logger.info(f"Unique semantic categories: {unique_categories}")
        
        return dataset


class CodebookClusterer:
    """Class to handle different clustering methods for the Chameleon codebook"""
    
    def __init__(self, model_path, model_type="chameleon", vq_model_type="VQ-16", codebook_size=16384, codebook_embed_dim=8, vq_ckpt=None, cache_dir=None, gnn_model_path=None, use_lm_embeddings=False):
        """
        Initialize the clusterer with a Chameleon, Janus, Emu3 or LlamaGen VQ model
        
        Args:
            model_path (str): Path to the model
            model_type (str): Type of model to use ("chameleon", "janus", "Emu3" or "llamagen")
            vq_model_type (str): Type of VQ model for Janus ("VQ-16" or "VQ-8")
            codebook_size (int): Size of codebook for Janus VQ model
            codebook_embed_dim (int): Embedding dimension for Janus VQ model
            vq_ckpt (str): Path to VQ model checkpoint for Janus
            cache_dir (str): Directory for caching model files
            gnn_model_path (str): Path to trained GNN model and embeddings
        """
        self.use_lm_embeddings = use_lm_embeddings
        self.model_type = model_type
        if model_type == "chameleon":
            self.model, self.processor = load_model_and_processor(model_path, cache_dir=cache_dir)
            self.codebook = self.model.model.vqmodel.quantize.embedding.weight.detach().to(torch.float32).cpu().numpy()
            _, _, indices = self.model.model.vqmodel.quantize(self.model.model.vqmodel.quantize.embedding.weight)
            indices = indices.view(-1,)
            self.lm_embeddings = self.model.model.embed_tokens(indices).detach().to(torch.float32).cpu().numpy()
        elif model_type == "janus":
            # Load Janus model and processor
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                output_hidden_states=True,
                cache_dir=cache_dir
            ).cuda().eval()
            
            self.model = model
            self.processor = vl_chat_processor
            # Get the codebook from the vision tower's VQ model
            self.codebook = model.gen_vision_model.quantize.embedding.weight.detach().to(torch.float32).cpu().numpy()
            
            
        elif model_type == "Emu3":
            # Load Emu3 model and processor
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                cache_dir=cache_dir
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
            image_processor = AutoImageProcessor.from_pretrained("BAAI/Emu3-VisionTokenizer", trust_remote_code=True)
            image_tokenizer = AutoModel.from_pretrained(
                "BAAI/Emu3-VisionTokenizer", 
                device_map="cuda:0", 
                trust_remote_code=True
            ).eval()
            self.vision_tokenizer = image_tokenizer
            self.model = model
            self.processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)
            # Get the codebook from the vision tokenizer
            self.codebook = image_tokenizer.quantize.embedding.weight.detach().to(torch.float32).cpu().numpy()
        else:  # llamagen
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            vq_model = VQ_models[vq_model_type](
                codebook_size=codebook_size,
                codebook_embed_dim=codebook_embed_dim)
            vq_model.to(device)
            vq_model.eval()
            if vq_ckpt:
                checkpoint = torch.load(vq_ckpt, map_location="cpu")
                vq_model.load_state_dict(checkpoint["model"])
                del checkpoint
            logger.info("LlamaGen VQ model loaded")
            self.model = vq_model
            self.codebook = vq_model.quantize.embedding.weight.detach().to(torch.float32).cpu().numpy()
            
        self.num_embeddings, self.embedding_dim = self.codebook.shape
        logger.info(f"Loaded codebook with shape: {self.codebook.shape}")
        
        # Load GNN embeddings if provided
        self.gnn_embeddings = None
        if gnn_model_path:
            gnn_model_path = Path(gnn_model_path)
            if (gnn_model_path / "node_embeddings.pt").exists():
                logger.info("Loading pre-computed GNN embeddings...")
                embeddings_data = torch.load(gnn_model_path / "node_embeddings.pt")
                self.gnn_embeddings = embeddings_data['node_embeddings'].numpy()
                logger.info(f"Loaded GNN embeddings with shape: {self.gnn_embeddings.shape}")
            else:
                raise FileNotFoundError(f"GNN embeddings file not found at {gnn_model_path / 'node_embeddings.pt'}")
    
    def visualize_cluster_sizes(self, cluster_labels, method_name, save_path=None):
        """
        Visualize the size distribution of clusters
        
        Args:
            cluster_labels (np.ndarray): Cluster assignments
            method_name (str): Name of the clustering method
            save_path (str, optional): Path to save the visualization
        """
        # Calculate cluster sizes
        unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
        
        # Sort clusters by size for better visualization
        sort_idx = np.argsort(counts)[::-1]  # Sort in descending order
        unique_clusters = unique_clusters[sort_idx]
        counts = counts[sort_idx]
        
        # Create figure
        plt.figure(figsize=(15, 6))
        
        # Plot cluster sizes
        plt.bar(range(len(unique_clusters)), counts)
        plt.title(f'Cluster Size Distribution ({method_name})')
        plt.xlabel('Cluster ID (sorted by size)')
        plt.ylabel('Number of Codebook Vectors')
        
        # Add size statistics as text
        stats_text = f'Total Clusters: {len(unique_clusters)}\n'
        stats_text += f'Mean Size: {counts.mean():.1f}\n'
        stats_text += f'Median Size: {np.median(counts):.1f}\n'
        stats_text += f'Min Size: {counts.min()}\n'
        stats_text += f'Max Size: {counts.max()}'
        
        # Position the text box in figure coords
        plt.text(0.95, 0.95, stats_text,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add grid for better readability
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'cluster_counts': counts,
            'stats': {
                'total_clusters': len(unique_clusters),
                'mean_size': counts.mean(),
                'median_size': np.median(counts),
                'min_size': counts.min(),
                'max_size': counts.max()
            }
        }

    def train_kmeans(self, n_clusters, random_state=42, balanced=False):
        """
        Train K-means clustering on the codebook
        
        Args:
            n_clusters (int): Number of clusters
            random_state (int): Random seed for reproducibility
            balanced (bool): Whether to use balanced K-means
            
        Returns:
            dict: Clustering results including model and labels
        """
        # Use GNN embeddings if available, otherwise use raw codebook
        if self.use_lm_embeddings:
            data_to_cluster = self.lm_embeddings
            logger.info(f"Training {'Balanced' if balanced else ''} K-means with {n_clusters} clusters using LM embeddings...")
        else:
            data_to_cluster = self.gnn_embeddings if self.gnn_embeddings is not None else self.codebook
            logger.info(f"Training {'Balanced' if balanced else ''} K-means with {n_clusters} clusters using {'GNN embeddings' if self.gnn_embeddings is not None else 'raw codebook'}...")
        
        if balanced:
            from train_balanced_clustering import BalancedKMeans
            kmeans = BalancedKMeans(n_clusters=n_clusters, verbose=True)
            labels = kmeans.fit_predict(data_to_cluster)
            cluster_centers = kmeans.centroids
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
            labels = kmeans.fit_predict(data_to_cluster)
            cluster_centers = kmeans.cluster_centers_
        
        # Calculate cluster statistics
        cluster_sizes = np.bincount(labels)
        
        return {
            'model': kmeans,
            'labels': labels,
            'cluster_sizes': cluster_sizes,
            'cluster_centers': cluster_centers
        }
    
    def train_dbscan(self, eps=0.5, min_samples=5):
        """
        Train DBSCAN clustering on the codebook
        
        Args:
            eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other
            min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point
            
        Returns:
            dict: Clustering results including model and labels
        """
        # Use GNN embeddings if available, otherwise use raw codebook
        data_to_cluster = self.gnn_embeddings if self.gnn_embeddings is not None else self.codebook
        logger.info(f"Training DBSCAN with eps={eps}, min_samples={min_samples} using {'GNN embeddings' if self.gnn_embeddings is not None else 'raw codebook'}...")
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(data_to_cluster)
        
        # Calculate cluster statistics
        cluster_sizes = np.bincount(labels[labels >= 0])
        n_noise = np.sum(labels == -1)
        
        return {
            'model': dbscan,
            'labels': labels,
            'cluster_sizes': cluster_sizes,
            'n_noise': n_noise
        }
    
    def train_knn(self, n_neighbors=5):
        """
        Train k-nearest neighbors on the codebook
        
        Args:
            n_neighbors (int): Number of neighbors to find
            
        Returns:
            dict: KNN results including distances and indices
        """
        # Use GNN embeddings if available, otherwise use raw codebook
        data_to_cluster = self.gnn_embeddings if self.gnn_embeddings is not None else self.codebook
        logger.info(f"Training KNN with {n_neighbors} neighbors using {'GNN embeddings' if self.gnn_embeddings is not None else 'raw codebook'}...")
        
        knn = NearestNeighbors(n_neighbors=n_neighbors)
        knn.fit(data_to_cluster)
        
        # Find nearest neighbors for all points
        distances, indices = knn.kneighbors(data_to_cluster)
        
        return {
            'model': knn,
            'distances': distances,
            'indices': indices
        }
    
    def visualize_clusters(self, labels, save_path=None):
        """
        Visualize clustering results using dimensionality reduction
        
        Args:
            labels (np.ndarray): Cluster labels
            save_path (str, optional): Path to save the visualization
        """
        from sklearn.manifold import TSNE
        
        # Use GNN embeddings if available, otherwise use raw codebook
        data_to_visualize = self.gnn_embeddings if self.gnn_embeddings is not None else self.codebook
        data_type = "GNN embeddings" if self.gnn_embeddings is not None else "raw codebook"
        
        # Reduce dimensionality for visualization
        logger.info(f"Running t-SNE on {data_type}...")
        tsne = TSNE(n_components=2, random_state=42)
        reduced_data = tsne.fit_transform(data_to_visualize)
        
        # Plot clusters
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis')
        plt.colorbar(scatter)
        plt.title(f'Codebook Clusters Visualization (t-SNE) using {data_type}')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_cluster_semantics(self, cluster_labels, sample_images, output_dir):
        """
        Analyze the semantic meaning of clusters by visualizing representative images
        
        Args:
            cluster_labels (np.ndarray): Cluster assignments
            sample_images (list): List of image paths to analyze
            output_dir (str): Directory to save analysis results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Get model's dtype
        model_dtype = next(self.model.parameters()).dtype
        device = next(self.model.parameters()).device
        
        # Process sample images and get their codebook usage
        for img_path in tqdm(sample_images, desc="Analyzing images"):
            img_name = Path(img_path).stem
            
            # Load and process image
            image = Image.open(img_path)
            
            if self.model_type == "chameleon":  # Chameleon model
                # Add a dummy text prompt since the processor requires it
                dummy_prompt = "<image>"
                inputs = self.processor(text=dummy_prompt, images=image, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(device, dtype=model_dtype)
            elif self.model_type == "janus":  # Janus model
                # Process image for Janus
                """
                conversation = [
                    {
                        "role": "<|User|>",
                        "content": "<image_placeholder>",
                        "images": [image],
                    },
                    {"role": "<|Assistant|>", "content": ""},
                ]
                
                pil_images = [image]
                inputs = self.processor(
                    conversations=conversation, 
                    images=pil_images, 
                    force_batchify=True
                )
                pixel_values = inputs["pixel_values"].squeeze(0).to(device, dtype=model_dtype)
                """
                # Convert image to tensor and normalize
                from torchvision import transforms
                transform = transforms.Compose([
                    transforms.Resize((384, 384)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
                pixel_values = transform(image).unsqueeze(0).to(device, dtype=model_dtype)
            elif self.model_type == "Emu3":  # Emu3 model
                pixel_values = self.processor.image_processor([image], return_tensors="pt")["pixel_values"]
                pixel_values = pixel_values.to(self.vision_tokenizer.device, self.vision_tokenizer.dtype)
                
                # Get quantized representation
                with torch.no_grad():
                    ndim = pixel_values.ndim
                    if ndim == 4:
                        t = self.vision_tokenizer.config.temporal_downsample_factor
                        b, c, h, w = pixel_values.shape
                        pixel_values = pixel_values.unsqueeze(1).repeat(1, t, 1, 1, 1)
                    elif ndim == 5:
                        b, t, c, h, w = pixel_values.shape
                    
                    encoder_output = self.vision_tokenizer.encoder(pixel_values)
                    # b t c h w -> b c t h w
                    encoder_output = encoder_output.permute(0, 2, 1, 3, 4)
                    hidden_state = self.vision_tokenizer.quant_conv(encoder_output)
                    # b c t h w -> b t c h w
                    hidden_state = hidden_state.permute(0, 2, 1, 3, 4)
                    indices = self.vision_tokenizer.quantize(hidden_state)
                    
                    if ndim == 4:
                        indices = indices.squeeze(1)

                    # Get quantized states and decoded image
                    b, h, w = indices.shape
                    quant = self.vision_tokenizer.quantize.embedding(indices.flatten())
                    c = quant.shape[-1]
                    quant = quant.view(b, 1, h, w, c).permute(0, 4, 1, 2, 3).contiguous()
                    quant2 = self.vision_tokenizer.post_quant_conv(quant)

                    quant = quant.permute(0, 2, 1, 3, 4)
                    quant2 = quant2.permute(0, 2, 1, 3, 4)
                    
                    decoded = self.vision_tokenizer.decoder(quant2, quant)
                    decoded = decoded.reshape(
                        b,
                        1 * self.vision_tokenizer.config.temporal_downsample_factor,
                        self.vision_tokenizer.config.out_channels,
                        h * self.vision_tokenizer.spatial_scale_factor,
                        w * self.vision_tokenizer.spatial_scale_factor,
                    )
                    decoded = decoded[:, 0]
            else:  # LlamaGen VQ model
                # Convert image to tensor and normalize
                from torchvision import transforms
                transform = transforms.Compose([
                    transforms.Resize((384, 384)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
                pixel_values = transform(image).unsqueeze(0).to(device, dtype=model_dtype)
            
            # Get quantized representation
            with torch.no_grad():
                if self.model_type == "chameleon":  # Chameleon model
                    encoder_output = self.model.model.vqmodel.encoder(pixel_values)
                    hidden_states = self.model.model.vqmodel.quant_conv(encoder_output)
                    quant, _, indices = self.model.model.vqmodel.quantize(hidden_states)
                    decoded = self.model.model.vqmodel.post_quant_conv(quant)
                    decoded = self.model.model.vqmodel.decoder(decoded)
                elif self.model_type == "janus":  # Janus model
                    vqmodel = self.model.gen_vision_model
                    encoder_output = vqmodel.encoder(pixel_values)
                    hidden_states = vqmodel.quant_conv(encoder_output)
                    quant, _, info = vqmodel.quantize(hidden_states)
                    decoded = vqmodel.post_quant_conv(quant)
                    decoded = vqmodel.decoder(decoded)
                    indices = info[2]  # info = (perplexity, min_encodings, min_encoding_indices)
                elif self.model_type == "Emu3":  # Emu3 model
                    # Note: For Emu3, we already have the indices and decoded image from the previous step
                    # Just need to ensure they're in the right format
                    indices = indices  # Already computed above
                    decoded = decoded  # Already computed above
                    quant = quant  # Already computed above
                else:  # LlamaGen VQ model
                    encoder_output = self.model.encoder(pixel_values)
                    hidden_states = self.model.quant_conv(encoder_output)
                    quant, _, info = self.model.quantize(hidden_states)
                    decoded = self.model.post_quant_conv(quant)
                    decoded = self.model.decoder(decoded)
                    indices = info[2]  # info = (perplexity, min_encodings, min_encoding_indices)
            
            # Convert indices to cluster labels
            indices = indices.cpu().numpy().flatten()
            image_cluster_labels = cluster_labels[indices]
            
            # Reshape indices into grid (assuming square shape)
            grid_size = int(np.sqrt(len(indices)))
            cluster_grid = image_cluster_labels.reshape(grid_size, grid_size) if self.model_type != "Emu3" else image_cluster_labels.reshape(h, w)
            
            # Visualize the image with its cluster distribution
            plt.figure(figsize=(20, 5))
            
            # Original image
            plt.subplot(1, 4, 1)
            plt.imshow(image)
            plt.title("Original Image")
            plt.axis('off')
            
            # Cluster distribution
            plt.subplot(1, 4, 2)
            unique_clusters, counts = np.unique(image_cluster_labels, return_counts=True)
            plt.bar(unique_clusters, counts)
            plt.title(f"Cluster Distribution\n(using {'GNN embeddings' if self.gnn_embeddings is not None else 'raw codebook'})")
            plt.xlabel("Cluster ID")
            plt.ylabel("Count")
            
            # Cluster indices grid
            plt.subplot(1, 4, 3)
            im = plt.imshow(cluster_grid, cmap='viridis')
            plt.title("Cluster Indices Grid")
            plt.colorbar(im, orientation='horizontal', pad=0.15)
            plt.axis('off')
            
            # Reconstructed image
            plt.subplot(1, 4, 4)
            decoded_image = decoded.cpu().to(torch.float32)
            
            if hasattr(self.processor, 'postprocess'):  # Chameleon model
                # Process the decoded image for visualization using processor
                decoded_image = self.processor.postprocess(
                    decoded_image,
                    do_unnormalize=True,
                    rescale_factor=127.5,  # Inverse of 0.0078
                    return_tensors="pil"
                )["pixel_values"][0]
            else:  # LlamaGen or Janus VQ model
                # Denormalize and convert to PIL image
                decoded_image = decoded_image.squeeze(0)  # Remove batch dimension
                decoded_image = decoded_image * 0.5 + 0.5  # Denormalize
                decoded_image = decoded_image.clamp(0, 1)
                decoded_image = ToPILImage()(decoded_image)
                
            plt.imshow(decoded_image)
            plt.title("Reconstructed Image")
            plt.axis('off')
            
            # Save visualization
            plt.savefig(output_dir / f"{img_name}_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()