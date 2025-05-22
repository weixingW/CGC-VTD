import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import pickle
from PIL import Image
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import sys
from pyvis.network import Network
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
import json
import os
from coco_panoptic import load_coco_panoptic_data, get_panoptic_image_and_masks


sys.path.append(".")
sys.path.append("..")
from cgc_utils import CodebookClusterer, CodebookGraphDataset
from emu3.mllm.processing_emu3 import Emu3Processor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




def main():
    parser = argparse.ArgumentParser(description="Build graph dataset for codebook clustering")
    parser.add_argument("--model_path", required=True, help="Path to the model")
    parser.add_argument("--model_type", choices=["chameleon", "llamagen", "janus", "Emu3"], 
                       default="chameleon", help="Type of model to use")
    parser.add_argument("--coco_root", required=True, help="Path to COCO dataset root")
    parser.add_argument("--top_k_edges", type=float, default=0.5,
                       help="Fraction of top co-occurrences to keep as edges")
    parser.add_argument("--save_dir", required=True, help="Directory to save graph data")
    parser.add_argument("--memory_efficient", action="store_true", default=False,
                       help="Use memory-efficient mode to periodically save segment statistics to disk")
    parser.add_argument("--save_interval", type=int, default=100,
                       help="Save segment statistics to disk every N images (memory-efficient mode only)")
    parser.add_argument("--model_cache_dir", type=str, default=None,
                       help="Directory to cache model")
    parser.add_argument("--not_use_seg", action="store_true", default=False,
                                   help="Include segments-level cooccurrent or not")
    
    args = parser.parse_args()
    
    # Initialize codebook clusterer
    clusterer = CodebookClusterer(
        model_path=args.model_path,
        model_type=args.model_type,
        cache_dir=args.model_cache_dir
    )
    if args.not_use_seg:
        logger.info("Build graph without segments masks")
    # Create graph dataset
    dataset = CodebookGraphDataset(
        codebook=clusterer.codebook,
        coco_root=args.coco_root,
        model_type=args.model_type,
        processor=clusterer.processor,
        top_k_edges=args.top_k_edges,
        clusterer=clusterer,
        save_dir=args.save_dir,
        memory_efficient=args.memory_efficient,
        save_interval=args.save_interval,
        not_use_seg=args.not_use_seg
    )
    
    logger.info("Graph dataset creation completed")
    dataset.save_graph_data()

if __name__ == "__main__":
    main() 
