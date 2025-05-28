
import os
import sys
import json
import glob
import re
import logging
import argparse
from typing import Dict, List, Optional, Tuple
from functools import lru_cache

import torch
import tqdm
from PIL import Image
from torchvision import transforms
from transformers import (
    ChameleonForConditionalGeneration,
    ChameleonProcessor,
    set_seed,
    GenerationConfig,
)

# Add project root to path
sys.path.append(".")
sys.path.append("..")

from eval_utils import load_image
from latent_gen import run_latent_generation_with_gnn, run_latent_generation, load_models
from model_utils import load_model_and_processor
from vcd_utils.vcd_sample import evolve_vcd_sampling
from vcd_utils.vcd_add_noise import add_diffusion_noise
from internal_confidence import calculate_internal_confidence, load_coco_objects
from eval.chair import CHAIR

# Configure logging
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger()

# Constants
NEG_WORDS = ["No", "not", "no", "NO"]
POS_WORDS = ["Yes", "yes", "YES"]

def extract(line: str) -> str:
    """
    Extract yes/no answer from a line of text.
    
    Args:
        line: Input text line
        
    Returns:
        "Yes" or "No" based on the presence of positive/negative words
    """
    line = line.replace('.', '').replace(',', '')
    words = line.split(' ')
    
    for word in words:
        if word in NEG_WORDS or word.endswith("n't"):
            return "No"
        elif word in POS_WORDS:
            return "Yes"
    return "No"

def load_query_data(query_file: str) -> List[Dict]:
    """
    Load query data from JSON file.
    
    Args:
        query_file: Path to the JSON file containing queries
        
    Returns:
        List of query dictionaries
        
    Raises:
        FileNotFoundError: If query file doesn't exist
    """
    if not os.path.exists(query_file):
        raise FileNotFoundError(f"Query file not found: {query_file}")
        
    with open(query_file, 'r') as f:
        queries = json.load(f)
    logger.info(f"Loaded {len(queries)} queries from {query_file}")
    return queries

def get_amber_images_with_queries(image_dir: str, queries: List[Dict]) -> List[Tuple[str, int, str]]:
    """
    Get AMBER image paths with their IDs and queries.
    
    Args:
        image_dir: Directory containing AMBER images
        queries: List of query dictionaries containing image names and IDs
        
    Returns:
        List of tuples containing (image_path, image_id, query)
    """
    image_tuples = []
    for query in queries:
        image_name = query["image"]
        image_path = os.path.join(image_dir, image_name)
        if os.path.exists(image_path):
            image_tuples.append((image_path, query["id"], query["query"]))
    
    # Sort by ID
    sorted_images = sorted(image_tuples, key=lambda x: x[1])
    logger.info(f"Found {len(sorted_images)} matching images with queries")
    return sorted_images

@lru_cache(maxsize=1)
def load_existing_captions(output_file: str) -> Dict[int, str]:
    """
    Load existing captions from output file if it exists.
    
    Args:
        output_file: Path to the output JSON file
        
    Returns:
        Dictionary mapping image IDs to their captions
    """
    if not os.path.exists(output_file):
        return {}
        
    try:
        with open(output_file, "r") as f:
            data = json.load(f)
            return {item["id"]: item["response"] for item in data}
    except json.JSONDecodeError:
        logger.warning(f"Could not parse JSON from {output_file}, treating as empty")
        return {}

def save_amber_response(id: int, response: str, output_file: str) -> None:
    """
    Save AMBER response to JSON file with proper error handling.
    
    Args:
        id: Image ID
        response: Generated caption
        output_file: Path to output JSON file
        
    Raises:
        Exception: If there's an error saving the response
    """
    try:
        # Read current data with proper error handling
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse JSON from {output_file}, treating as empty")
                    data = []
        else:
            data = []
            
        # Update existing entry or add new one
        entry = {"id": id, "response": response}
        for i, item in enumerate(data):
            if item["id"] == id:
                data[i] = entry
                break
        else:
            data.append(entry)
        
        # Write to temporary file first, then rename to avoid partial writes
        temp_file = f"{output_file}.tmp"
        with open(temp_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Atomic rename operation
        os.replace(temp_file, output_file)
            
    except Exception as e:
        logger.error(f"Error saving response for id {id}: {str(e)}")
        raise

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a tensor to PIL Image in RGB mode.
    
    Args:
        tensor: Input tensor of shape (C, H, W) or (B, C, H, W)
        
    Returns:
        PIL Image in RGB mode
        
    Raises:
        ValueError: If tensor has unsupported number of channels
    """
    # Remove batch dimension if present
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Make sure tensor is on CPU
    tensor = tensor.cpu()
    
    # If values are in [0, 1], scale to [0, 255]
    if tensor.max() <= 1:
        tensor = tensor * 255
    
    # Convert to uint8
    tensor = tensor.clamp(0, 255).byte()
    
    # Convert to PIL
    if tensor.shape[0] == 1:  # Grayscale
        pil_image = Image.fromarray(tensor.squeeze().numpy(), mode='L').convert('RGB')
    elif tensor.shape[0] == 3:  # RGB
        pil_image = Image.fromarray(tensor.permute(1, 2, 0).numpy(), mode='RGB')
    elif tensor.shape[0] == 4:  # RGBA
        pil_image = Image.fromarray(tensor.permute(1, 2, 0).numpy(), mode='RGBA').convert('RGB')
    else:
        raise ValueError(f"Unsupported number of channels: {tensor.shape[0]}")
    
    return pil_image

def extract_coco_objects_from_text(text: str, coco_objects: List[str]) -> List[str]:
    """
    Extract COCO objects mentioned in the text.
    
    Args:
        text: Input text to analyze
        coco_objects: List of valid COCO object classes
        
    Returns:
        List of COCO objects found in the text
    """
    evaluator = CHAIR()
    words, node_words, idxs, double_words = evaluator.caption_to_words(text)
    return words

def get_low_confidence_objects(
    model_type: str,
    model,
    processor,
    image: Image.Image, 
    detected_objects: List[str],
    confidence_threshold: float = 0.5,
    enable_layer_editing: bool = False,
    edit_layer: Optional[int] = None,
    edit_weight: Optional[float] = None,
    cluster_results_path: Optional[str] = None,
    num_clusters: Optional[int] = 2,
    use_mean: Optional[bool] = True,
) -> Tuple[List[str], Dict[str, float]]:
    """Calculate confidence for detected objects and return those with low confidence."""
    # We only need to calculate confidence for detected objects
    confidence_scores, _, _ = calculate_internal_confidence(
        model_type=model_type,
        model=model,
        processor=processor,
        image=image,  # Function accepts either path or Image object
        coco_objects=detected_objects,  # Only calculate for detected objects
        enable_layer_editing=enable_layer_editing,
        edit_layer=edit_layer,
        edit_weight=edit_weight,
        cluster_results_path=cluster_results_path,
        num_clusters=num_clusters,
        use_mean=use_mean
    )
    
    # Extract objects below threshold
    low_confidence_objects = []
    for obj, confidence in confidence_scores.items():
        logger.info(f"Object '{obj}' confidence: {confidence:.4f}")
        
        if confidence < confidence_threshold:
            low_confidence_objects.append(obj)
    
    logger.info(f"Low confidence objects: {low_confidence_objects}")
    return low_confidence_objects, confidence_scores

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Generate text with latent image/text embedding manipulation."
    )
    
    # Input/output arguments
    parser.add_argument(
        "--query_file",
        type=str,
        required=False,
        default="data/query/query_all.json",
        help="Path to query_all.json containing image IDs and queries",
    )
    parser.add_argument("--image_dir", type=str, default=None, help="Image directory")
    parser.add_argument("--output_file", type=str, required=True, help="Output file")
    
    # Model arguments
    parser.add_argument("--model_id", type=str, default="leloy/Anole-7b-v0.1-hf", help="Model ID")
    parser.add_argument("--model_type", type=str, default="chameleon", help="Model type")
    parser.add_argument("--model_cache_dir", type=str, default="None", help="Model cache directory")
    
    # Generation arguments
    parser.add_argument("--gen_type", type=str, default="sample", 
                       choices=["sample", "opera", "vcd", "gnn", "sid", "gnn_opera", "gnn_vcd", "gnn_sid", "confidence", "confidence_gnn"],
                       help="Generation type")
    parser.add_argument("--operation", type=str, default="subtract", help="Operation to perform")
    parser.add_argument("--weight", type=float, default=1.0, help="Weight for projection subtraction")
    parser.add_argument("--layer", type=int, default=12, help="Layer to perform the operation")
    parser.add_argument("--text_edit_layer", type=int, default=21, help="Layer to perform the text edit")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of layers to modify")
    parser.add_argument("--layer_weights", type=float, nargs='+', default=[1.0], help="Weights for consecutive layers")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum number of new tokens")
    parser.add_argument("--prompt", type=str, default="Please describe this image in detail.", help="Prompt for caption generation")
    
    # GNN arguments
    parser.add_argument("--cluster_results_path", type=str, help="Path to cluster results (required for GNN generation)")
    parser.add_argument("--num_clusters", type=int, default=2, help="Number of top clusters to use")
    parser.add_argument("--find_non_input_indices", action="store_true", help="Whether to find indices that don't appear in the input image")
    
    # VCD arguments
    parser.add_argument("--noise_step", type=int, default=500, help="Number of noise steps for VCD")
    parser.add_argument("--cd_alpha", type=float, default=0.5, help="Alpha parameter for contrastive decoding")
    parser.add_argument("--cd_beta", type=float, default=0.1, help="Beta parameter for contrastive decoding")
    
    # Confidence arguments
    parser.add_argument("--coco_objects_path", type=str, default="data/coco_unique_objs.txt", 
                       help="Path to COCO objects file")
    parser.add_argument("--confidence_threshold", type=float, default=0.5,
                       help="Threshold for confidence-based generation")
    
    # Fast token merging arguments
    parser.add_argument("--fast-v-inplace", default=False)
    parser.add_argument("--fast-v-attention-rank", type=int, default=100)
    parser.add_argument("--fast-v-attention-rank-add", type=int, default=100)
    parser.add_argument("--fast-v-agg-layer", type=int, default=2)
    parser.add_argument("--fast-v-sys-length", default=None, type=int, help='the length of system prompt')
    parser.add_argument("--fast-v-image-token-length", default=None, type=int, help='the length of image token')
    
    # Other arguments
    parser.add_argument("--use_mean", action="store_true", help="Use mean pooling for text embeddings")
    parser.add_argument("--fast", action="store_true", help="Use fast settings")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser.parse_args()

def get_gpu_memory_usage() -> Tuple[float, float, float]:
    """
    Get current GPU memory usage in MB.
    
    Returns:
        Tuple of (allocated_memory, cached_memory, max_allocated_memory) in MB
    """
    if not torch.cuda.is_available():
        return 0.0, 0.0, 0.0
    
    # Get memory allocated for all tensors
    allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # Convert to MB
    # Get memory cached by the allocator
    cached = torch.cuda.memory_reserved() / (1024 * 1024)  # Convert to MB
    # Get maximum memory allocated
    max_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)  # Convert to MB
    
    return allocated, cached, max_allocated

def clear_gpu_memory() -> None:
    """Clear GPU memory cache and empty cache."""
    if torch.cuda.is_available():
        # Empty cache
        torch.cuda.empty_cache()
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()
        # Clear memory cache
        torch.cuda.memory.empty_cache()

def track_gpu_usage(func):
    """
    Decorator to track GPU memory usage during function execution.
    
    Args:
        func: Function to decorate
        
    Returns:
        Wrapped function that tracks GPU memory usage
    """
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            return func(*args, **kwargs)
            
        # Clear memory before starting
        clear_gpu_memory()
            
        # Get initial memory usage
        initial_allocated, initial_cached, _ = get_gpu_memory_usage()
        
        # Execute the function
        result = func(*args, **kwargs)
        
        # Get final memory usage
        final_allocated, final_cached, max_allocated = get_gpu_memory_usage()
        
        # Log memory usage
        logger.info(f"GPU Memory Usage:")
        logger.info(f"  Initial - Allocated: {initial_allocated:.2f}MB, Cached: {initial_cached:.2f}MB")
        logger.info(f"  Final   - Allocated: {final_allocated:.2f}MB, Cached: {final_cached:.2f}MB")
        logger.info(f"  Peak    - Allocated: {max_allocated:.2f}MB")
        
        # Clear memory after processing
        clear_gpu_memory()
        
        return result
    return wrapper

def get_generation_config(
    model_type: str,
    gen_type: str,
    args: argparse.Namespace,
    processor: object,
    inputs: Dict,
    model: object,
    key_position: Optional[Dict] = None
) -> GenerationConfig:
    """
    Get appropriate generation config based on model type and generation mode.
    
    Args:
        model_type: Type of model being used
        gen_type: Type of generation strategy
        args: Command line arguments
        processor: Model processor
        inputs: Model inputs
        model: Model instance
        key_position: Optional dictionary containing key positions
        
    Returns:
        GenerationConfig object configured for the specified generation strategy
    """
    if gen_type in ["vcd", "sid"]:
        # Prepare noisy image for VCD
        if isinstance(inputs["pixel_values"], str):
            # If it's a path, load the image first
            image = Image.open(inputs["pixel_values"])
            image = image.convert("RGB")
            if model_type == "chameleon":
                # Process with dummy prompt since processor requires text
                dummy_inputs = processor(text="<image>", images=image, return_tensors="pt")
                image_tensor = dummy_inputs["pixel_values"]
            elif model_type == "Emu3":
                image_tensor = processor.image_processor([image], return_tensors="pt")["pixel_values"]
            elif model_type == "janus":
                conversation = [
                    {
                        "role": "<|User|>",
                        "content": "<image_placeholder>",
                        "images": [image],
                    },
                    {"role": "<|Assistant|>", "content": ""},
                ]
                prepare_inputs = processor(
                    conversations=conversation,
                    images=[image],
                    force_batchify=True
                )
                image_tensor = prepare_inputs["pixel_values"]
        else:
            image_tensor = inputs["pixel_values"]

        if gen_type == "vcd":
            image_cd = add_diffusion_noise(image_tensor, args.noise_step)
        elif gen_type == "sid":
            image_cd = image_tensor
        image_cd = image_cd.to(model.device, dtype=torch.bfloat16)
        
        config = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": True,
            "vcd_sample": True,
            "cd_alpha": args.cd_alpha,
            "cd_beta": args.cd_beta,
            "vcd_inputs": image_cd,
            "output_attentions": False,
            "output_hidden_states": False,
            "return_dict_in_generate": True,
            "key_position": key_position
        }
        if gen_type == "sid":
            config["use_sid"] = True
        
        if model_type == "Emu3":
            config.update({
                "pad_token_id": processor.tokenizer.pad_token_id,
                "bos_token_id": processor.tokenizer.bos_token_id,
                "eos_token_id": processor.tokenizer.eos_token_id,
            })
            
        return GenerationConfig(**config)
        
    elif gen_type == "opera":
        config = {
            "max_length": 1600,
            "output_attentions": True,
            "num_beams": 5,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "opera_decoding": True,
            "key_position": key_position,
            "scale_factor": 50,
            "threshold": 15,
            "num_attn_candidates": 2,
            "penalty_weights": 1,
            "return_dict_in_generate": True,
            "pad_token_id": model.config.pad_token_id if model.config.pad_token_id is not None else processor.tokenizer.pad_token_id,
            "eos_token_id": model.config.eos_token_id if model.config.eos_token_id is not None else processor.tokenizer.eos_token_id
        }
        
        if model_type == "Emu3":
            config.update({
                "pad_token_id": processor.tokenizer.pad_token_id,
                "bos_token_id": processor.tokenizer.bos_token_id,
                "eos_token_id": processor.tokenizer.eos_token_id,
            })
            
        return GenerationConfig(**config)
        
    else:  # Default sampling or gnn
        config = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "pad_token_id": model.config.pad_token_id if model.config.pad_token_id is not None else processor.tokenizer.pad_token_id,
            "eos_token_id": model.config.eos_token_id if model.config.eos_token_id is not None else processor.tokenizer.eos_token_id
        }
        
        if model_type == "Emu3":
            config.update({
                "pad_token_id": processor.tokenizer.pad_token_id,
                "bos_token_id": processor.tokenizer.bos_token_id,
                "eos_token_id": processor.tokenizer.eos_token_id,
            })
            
        return GenerationConfig(**config)

@track_gpu_usage
def process_single_sample(
    args: argparse.Namespace,
    model: object,
    processor: object,
    image_path: str,
    amber_id: int,
    query: str,
    existing_captions: Dict[int, str],
    coco_objects: Optional[List[str]] = None
) -> Optional[str]:
    """
    Process a single sample with GPU memory tracking.
    
    Args:
        args: Command line arguments
        model: Model instance
        processor: Model processor
        image_path: Path to input image
        amber_id: AMBER image ID
        query: Input query
        existing_captions: Dictionary of existing captions
        coco_objects: Optional list of COCO object classes
        
    Returns:
        Generated caption or None if processing failed
    """
    if amber_id in existing_captions:
        logger.info(f"Skipping AMBER_{amber_id}, caption already exists")
        return None
        
    # For confidence-based generation
    if args.gen_type in ["confidence", "confidence_gnn"]:
        logger.info(f"Running confidence-based generation for AMBER_{amber_id}")
        
        # Load the image
        image = load_image(image_path)
        image = image.convert("RGB")
        
        if args.gen_type == "confidence":
            operation = "normal"
            logger.info(f"Running normal generation to generate baseline caption")
        elif args.gen_type == "confidence_gnn":
            operation = "subtract"
            logger.info(f"Running GNN generation to generate baseline caption")
            
        baseline_caption = run_latent_generation_with_gnn(
            model_type=args.model_type,
            model=model,
            processor=processor,
            image_1_path=image_path,
            prompt=query,
            layer=args.layer,
            operation=operation,
            weight=args.weight,
            use_mean=args.use_mean,
            max_new_tokens=args.max_new_tokens,
            fast=args.fast,
            model_cache_dir=args.model_cache_dir,
            seed=args.seed,
            cluster_results_path=args.cluster_results_path,
            find_non_input_indices=args.find_non_input_indices,
            num_layers=args.num_layers,
            num_clusters=args.num_clusters,
            layer_weights=args.layer_weights,
        )
        logger.info(f"Baseline caption generated")
        
        # Extract COCO objects from caption
        detected_objects = extract_coco_objects_from_text(baseline_caption, coco_objects)
        logger.info(f"Detected objects: {detected_objects}")
        
        if not detected_objects:
            logger.info("No COCO objects detected in caption, using baseline caption")
            response = baseline_caption
        else:
            # Calculate confidence for detected objects
            enable_layer_editing = args.gen_type == "confidence_gnn"
            
            low_confidence_objects, confidence_scores = get_low_confidence_objects(
                model_type=args.model_type,
                model=model,
                processor=processor,
                image=image,
                detected_objects=detected_objects,
                confidence_threshold=args.confidence_threshold,
                enable_layer_editing=enable_layer_editing,
                edit_layer=args.layer,
                edit_weight=args.weight,
                cluster_results_path=args.cluster_results_path if args.gen_type == "confidence_gnn" else None,
                num_clusters=args.num_clusters,
                use_mean=args.use_mean
            )
            
            if not low_confidence_objects:
                logger.info("No low confidence objects detected, using baseline caption")
                response = baseline_caption
            else:
                logger.info(f"Generating without low confidence objects: {low_confidence_objects}")

                if args.gen_type == "confidence":
                    response = run_latent_generation(
                        model_type=args.model_type,
                        model=model,
                        processor=processor,
                        image_1=image,
                        text_input=low_confidence_objects,
                        prompt=query,
                        layer=args.text_edit_layer,
                        operation="subtract",
                        weight=args.weight,
                        use_mean=args.use_mean,
                        max_new_tokens=args.max_new_tokens,
                        fast=args.fast
                    )
                elif args.gen_type == "confidence_gnn":
                    response = run_latent_generation_with_gnn(
                        model_type=args.model_type,
                        model=model,
                        processor=processor,
                        model_id=args.model_id,
                        image_1_path=image_path,
                        prompt=query,
                        layer=args.layer,
                        operation=args.operation,
                        weight=args.weight,
                        use_mean=args.use_mean,
                        max_new_tokens=args.max_new_tokens,
                        fast=args.fast,
                        model_cache_dir=args.model_cache_dir,
                        seed=args.seed,
                        cluster_results_path=args.cluster_results_path,
                        find_non_input_indices=args.find_non_input_indices,
                        num_layers=args.num_layers,
                        num_clusters=args.num_clusters,
                        layer_weights=args.layer_weights,
                        text_input=low_confidence_objects,
                        text_edit_layer=args.text_edit_layer
                    )
    elif args.gen_type in ["gnn", "gnn_opera", "gnn_vcd", "gnn_sid"]:
        if not args.cluster_results_path:
            raise ValueError("cluster_results_path must be provided for GNN generation")
            
        response = run_latent_generation_with_gnn(
            model_type=args.model_type,
            model=model,
            processor=processor,
            model_id=args.model_id,
            image_1_path=image_path,
            prompt=query,
            layer=args.layer,
            operation=args.operation,
            weight=args.weight,
            use_mean=args.use_mean,
            max_new_tokens=args.max_new_tokens,
            fast=args.fast,
            model_cache_dir=args.model_cache_dir,
            seed=args.seed,
            cluster_results_path=args.cluster_results_path,
            find_non_input_indices=args.find_non_input_indices,
            num_layers=args.num_layers,
            layer_weights=args.layer_weights,
            num_clusters=args.num_clusters,
            use_opera=args.gen_type == "gnn_opera",
            use_vcd=args.gen_type == "gnn_vcd",
            use_sid=args.gen_type == "gnn_sid",
        )
    else:
        image = load_image(image_path)
        image = image.convert("RGB")
        
        # Process inputs based on model type
        if args.model_type == "chameleon":
            inputs = processor(
                text=query + "<image>",
                images=image,
                return_tensors="pt"
            )
            # Convert inputs to bfloat16 to match model dtype
            model_dtype = next(model.parameters()).dtype
            device = next(model.parameters()).device
            inputs = {
                "input_ids": inputs["input_ids"].to(device, dtype=torch.long),
                "attention_mask": inputs["attention_mask"].to(device, dtype=torch.long),
                "pixel_values": inputs["pixel_values"].to(device, dtype=model_dtype)
            }
            
            # Get key positions
            image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
            image_token_pos = (inputs["input_ids"] == image_token_id).nonzero()[0, 1].item()
            key_position = {
                "image_start": torch.tensor(image_token_pos).to(model.device),
                "image_end": torch.tensor(image_token_pos + 1024).to(model.device),
                "response_start": torch.tensor(inputs["input_ids"].shape[-1]).to(model.device),
            }
            if args.gen_type == "sid":
                model.model.config.use_fast_v = True
                model.model.config.fast_v_inplace = args.fast_v_inplace
                model.model.config.fast_v_sys_length = args.fast_v_sys_length
                model.model.config.fast_v_image_token_length = args.fast_v_image_token_length
                model.model.config.fast_v_attention_rank = args.fast_v_attention_rank
                model.model.config.fast_v_attention_rank_add = args.fast_v_attention_rank_add
                model.model.config.fast_v_agg_layer = args.fast_v_agg_layer
            else:
                model.model.config.use_fast_v = False
            model.model.reset_fastv()
            
        elif args.model_type == "Emu3":
            inputs, image_start_list, image_end_list = processor(
                text=query,
                image=image,
                mode="U",
                return_tensors="pt",
                padding="longest",
            )
            
            image_tensor = processor.image_processor([image], return_tensors="pt")["pixel_values"]
            image_cd = add_diffusion_noise(image_tensor, args.noise_step)
            image_cd =  image_cd.to(model.device, dtype=torch.bfloat16)
            image_cd = tensor_to_pil(image_cd)
            inputs_cd, _, _ = processor(
                text=query,
                image=image_cd,
                mode="U",
                return_tensors="pt",
                padding="longest",
            )
            
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            key_position = {
                "image_start": torch.tensor(image_start_list[0]).to(model.device),
                "image_end": torch.tensor(image_end_list[0]).to(model.device),
                "response_start": torch.tensor(inputs["input_ids"].shape[-1]).to(model.device),
            }
            if args.gen_type == "sid":
                model.model.config.use_fast_v = True
                model.model.config.fast_v_inplace = args.fast_v_inplace
                model.model.config.fast_v_sys_length = args.fast_v_sys_length
                model.model.config.fast_v_image_token_length = args.fast_v_image_token_length
                model.model.config.fast_v_attention_rank = args.fast_v_attention_rank
                model.model.config.fast_v_attention_rank_add = args.fast_v_attention_rank_add
                model.model.config.fast_v_agg_layer = args.fast_v_agg_layer
            else:
                model.model.config.use_fast_v = False
            model.model.reset_fastv()
            
        elif args.model_type == "janus":
            conversation = [
                {
                    "role": "<|User|>",
                    "content": f"<image_placeholder>\n{query}",
                    "images": [image],
                },
                {"role": "<|Assistant|>", "content": ""},
            ]
            
            prepare_inputs = processor(
                conversations=conversation,
                images=[image],
                force_batchify=True
            ).to(model.device)
            inputs = model.prepare_inputs_embeds(**prepare_inputs)
            
            img_start_id = processor.image_start_id
            img_end_id = processor.image_end_id
            img_start_pos = (prepare_inputs["input_ids"] == img_start_id).nonzero()[0, 1].item()+1
            img_end_pos = (prepare_inputs["input_ids"] == img_end_id).nonzero()[0, 1].item()
            
            key_position = {
                "image_start": torch.tensor(img_start_pos).to(model.device),
                "image_end": torch.tensor(img_end_pos).to(model.device),
                "response_start": torch.tensor(inputs.shape[-2]).to(model.device),
            }
            if args.gen_type == "sid":
                model.language_model.model.config.use_fast_v = True
                model.language_model.model.config.fast_v_inplace = args.fast_v_inplace
                model.language_model.model.config.fast_v_sys_length = args.fast_v_sys_length
                model.language_model.model.config.fast_v_image_token_length = args.fast_v_image_token_length
                model.language_model.model.config.fast_v_attention_rank = args.fast_v_attention_rank
                model.language_model.model.config.fast_v_attention_rank_add = args.fast_v_attention_rank_add
                model.language_model.model.config.fast_v_agg_layer = args.fast_v_agg_layer
            else:
                model.language_model.model.config.use_fast_v = False
            model.language_model.model.reset_fastv()
        
        # Get generation config
        generation_config = get_generation_config(
            model_type=args.model_type,
            gen_type=args.gen_type,
            args=args,
            processor=processor,
            inputs={"pixel_values": image_path, "text": query},
            model=model,
            key_position=key_position
        )
        
        # Generate response
        with torch.inference_mode():
            if args.model_type == "chameleon":
                if args.gen_type == "vcd":
                    image_cd = tensor_to_pil(generation_config.vcd_inputs.squeeze(0))
                    inputs_cd = processor(text=query + "<image>", images=image_cd, return_tensors="pt")
                    inputs_cd_ids = inputs_cd.input_ids.to(device, dtype=torch.long)
                    generation_config.vcd_inputs = inputs_cd_ids
                elif args.gen_type == "sid":
                    generation_config.vcd_inputs = inputs["input_ids"]
                output = model.generate(
                    **inputs,
                    generation_config=generation_config,
                )
                
                if isinstance(output, dict):
                    output = output["sequences"]
                
                response = processor.decode(
                    output[0][len(inputs["input_ids"][0]):], 
                    skip_special_tokens=True
                )
                
            elif args.model_type == "Emu3":
                try:
                    if args.gen_type == "vcd":
                        generation_config.vcd_inputs = inputs_cd["input_ids"].to(model.device, dtype=torch.long)
                    elif args.gen_type == "sid":
                        generation_config.vcd_inputs = inputs["input_ids"].to(model.device, dtype=torch.long)

                    outputs = model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        generation_config=generation_config
                    )
                    if isinstance(outputs, dict):
                        generated_sequence = outputs.sequences
                    else:
                        generated_sequence = outputs
                    outputs = generated_sequence[:, inputs["input_ids"].shape[-1]:]
                    response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
                except Exception as e:
                    logger.info(f"Error: {e}")
                    response = f"Error: {e}"
                
            elif args.model_type == "janus":
                detached_inputs = inputs.detach().clone()
                detached_attention_mask = prepare_inputs.attention_mask.detach().clone()
                if args.gen_type == "vcd":
                    image_cd = tensor_to_pil(generation_config.vcd_inputs.squeeze(0))
                    conversation = [
                        {
                            "role": "<|User|>",
                            "content": f"<image_placeholder>\n{query}",
                            "images": [image_cd],
                        },
                        {"role": "<|Assistant|>", "content": ""},
                    ]
                    prepare_inputs_cd = processor(
                        conversations=conversation,
                        images=[image_cd],
                        return_tensors="pt"
                    ).to(model.device)
                    inputs_cd = model.prepare_inputs_embeds(**prepare_inputs_cd)
                    generation_config.vcd_inputs=inputs_cd.detach().clone()
                elif args.gen_type == "sid":
                    generation_config.vcd_inputs = inputs.detach().clone()
                                
                outputs = model.language_model.generate(
                    inputs_embeds=detached_inputs,
                    attention_mask=detached_attention_mask,
                    generation_config=generation_config
                )
                
                if isinstance(outputs, dict):
                    generated_sequence = outputs.sequences
                else:
                    generated_sequence = outputs
                    
                response = processor.tokenizer.decode(
                    generated_sequence[0].cpu().tolist(), 
                    skip_special_tokens=True
                )
    
    if response.startswith("Error"):
        logger.info(f"Error: {response}")
        return None
        
    logger.info(f"Response: {response}")
    
    # Extract yes/no answer if needed
    if amber_id >= 1005:
        response = extract(response)
        
    return response

def main() -> None:
    """Main function to run the AMBER image caption generation."""
    args = parse_arguments()
    logger.info(f"Loading model and processor from {args.model_id}")
    
    # Clear GPU memory before loading model
    clear_gpu_memory()
    
    # Load model and processor based on model type
    model, processor = load_model_and_processor(
        model_path=args.model_id,
        model_type=args.model_type,
        cache_dir=args.model_cache_dir
    )
    
    # Load query data and get images
    queries = load_query_data(args.query_file)
    image_dir = args.image_dir
    image_data = get_amber_images_with_queries(image_dir, queries)
    logger.info(f"Found {len(image_data)} AMBER images with queries")
    
    # Load existing captions
    existing_captions = load_existing_captions(args.output_file)
    logger.info(f"Loaded {len(existing_captions)} existing captions")
    
    # Load COCO objects if needed for confidence-based generation
    coco_objects = None
    if args.gen_type in ["confidence", "confidence_gnn"]:
        coco_objects = load_coco_objects(args.coco_objects_path)
        logger.info(f"Loaded {len(coco_objects)} COCO object classes")
    
    for image_path, amber_id, query in tqdm.tqdm(image_data, desc="Generating captions"):
        # Process single sample with GPU memory tracking
        response = process_single_sample(
            args=args,
            model=model,
            processor=processor,
            image_path=image_path,
            amber_id=amber_id,
            query=query,
            existing_captions=existing_captions,
            coco_objects=coco_objects
        )
        
        if response is None:
            continue
            
        # Save response with proper error handling
        save_amber_response(amber_id, response, args.output_file)
        logger.info(f"Generated caption for AMBER_{amber_id}")
        
        # Clear GPU memory between samples
        clear_gpu_memory()

    logger.info("Finished generating captions")

if __name__ == "__main__":
    main() 
