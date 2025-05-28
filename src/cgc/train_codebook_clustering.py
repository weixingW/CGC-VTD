import torch
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from transformers import ChameleonForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer, AutoImageProcessor, AutoModel
import argparse
from pathlib import Path
import logging
from tqdm import tqdm
import pickle
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_utils import load_model_and_processor
from PIL import Image
from torchvision.transforms import ToPILImage
from cgc_utils import CodebookClusterer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def main():
    parser = argparse.ArgumentParser(description="Train and analyze codebook clustering")
    parser.add_argument("--model_path", required=True, help="Path to the model")
    parser.add_argument("--model_type", choices=["chameleon", "llamagen", "janus", "Emu3"], default="chameleon", help="Type of model to use")
    parser.add_argument("--vq_model", choices=["VQ-16", "VQ-8"], default="VQ-16", help="Type of VQ model for LlamaGen")
    parser.add_argument("--codebook_size", type=int, default=16384, help="Size of codebook for LlamaGen VQ model")
    parser.add_argument("--codebook_embed_dim", type=int, default=8, help="Embedding dimension for LlamaGen VQ model")
    parser.add_argument("--vq_ckpt", type=str, help="Path to VQ model checkpoint for LlamaGen")
    parser.add_argument("--n_clusters", type=int, default=30, help="Number of clusters for K-means")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--sample_images", nargs="+", help="Paths to sample images for analysis")
    parser.add_argument("--cache_dir", type=str, default=None, help="Directory for caching model files")
    parser.add_argument("--gnn_model_path", type=str, help="Path to trained GNN model and embeddings")
    parser.add_argument("--balanced", action="store_true", help="Use balanced K-means clustering")
    parser.add_argument("--use_lm_embeddings", action="store_true", help="Use LM embeddings for clustering")
    args = parser.parse_args()
    
    # Initialize clusterer
    clusterer = CodebookClusterer(
        model_path=args.model_path,
        model_type=args.model_type,
        vq_model_type=args.vq_model,
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim,
        vq_ckpt=args.vq_ckpt,
        cache_dir=args.cache_dir,
        gnn_model_path=args.gnn_model_path,
        use_lm_embeddings=args.use_lm_embeddings
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Train different clustering methods
    # 1. K-means
    kmeans_results = clusterer.train_kmeans(args.n_clusters, balanced=args.balanced)
    #clusterer.visualize_clusters(kmeans_results['labels'], 
    #                           save_path=output_dir / f"{'balanced_' if args.balanced else ''}kmeans_clusters.png")
    # Visualize K-means cluster sizes
    kmeans_size_stats = clusterer.visualize_cluster_sizes(
        kmeans_results['labels'], 
        "Balanced K-means" if args.balanced else "K-means",
        save_path=output_dir / f"{'balanced_' if args.balanced else ''}kmeans_cluster_sizes.png"
    )
    
    # 2. DBSCAN
    #dbscan_results = clusterer.train_dbscan()
    #clusterer.visualize_clusters(dbscan_results['labels'], 
    #                           save_path=output_dir / "dbscan_clusters.png")
    
    # 3. KNN
    #knn_results = clusterer.train_knn()
    
    # Save results
    #results = {
    #    'kmeans': {**kmeans_results, 'size_stats': kmeans_size_stats},
    #    'dbscan': {**dbscan_results, 'size_stats': dbscan_size_stats},
    #    'knn': knn_results
    #}
    
    #with open(output_dir / "clustering_results.pkl", "wb") as f:
    #    pickle.dump(results, f)
    
    # Analyze sample images if provided
    if args.sample_images:
        semantic_analysis_dir = output_dir / "semantic_analysis"
        semantic_analysis_dir.mkdir(exist_ok=True)
        clusterer.analyze_cluster_semantics(
            kmeans_results['labels'],
            args.sample_images,
            semantic_analysis_dir
        )

if __name__ == "__main__":
    main() 