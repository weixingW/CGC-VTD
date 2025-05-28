"""
MME (Multimodal Evaluation) Generation Script

This script handles generation of responses for multimodal evaluation tasks using various
large vision-language models. It supports multiple generation strategies including:
- Standard sampling
- VCD (Vision-Contrastive Decoding)
- GNN-based generation
- Confidence-based generation
- And more
"""

import os
import sys
import glob
import re
import json
import logging
import argparse
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

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

# Local imports
from latent_gen import run_latent_generation_with_gnn, run_latent_generation, load_models
from model_utils import load_model_and_processor
from vcd_utils.vcd_sample import evolve_vcd_sampling
from vcd_utils.vcd_add_noise import add_diffusion_noise
from internal_confidence import calculate_internal_confidence, load_coco_objects
from eval.chair import CHAIR

# Add project root to path
sys.path.extend([".", ".."])

# Configure logging
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Type aliases
ImageType = Union[str, Image.Image]
ModelType = Union[ChameleonForConditionalGeneration, torch.nn.Module]
ProcessorType = Union[ChameleonProcessor, torch.nn.Module]

def load_image(image_path: str) -> Image.Image:
    """Load an image from a path or URL.
    
    Args:
        image_path: Path to image file or URL
        
    Returns:
        PIL Image object
    """
    if image_path.startswith("http"):
        import requests
        image = Image.open(requests.get(image_path, stream=True).raw)
    else:
        image = Image.open(image_path)
    return image.convert("RGB")

def format_response(response: str) -> str:
    """Clean and format a response string by removing special characters.
    
    Args:
        response: Raw response string
        
    Returns:
        Cleaned response string
    """
    return re.sub(r'[\n\t\r\f\v]', ' ', response)

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a tensor to PIL Image in RGB mode.
    
    Args:
        tensor: Input tensor of shape [C, H, W] or [B, C, H, W]
        
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
    
    # Convert to PIL based on number of channels
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
    """Extract COCO objects mentioned in the text.
    
    Args:
        text: Input text to analyze
        coco_objects: List of valid COCO object classes
        
    Returns:
        List of detected COCO objects
    """
    evaluator = CHAIR()
    words, _, _, _ = evaluator.caption_to_words(text)
    return words

def get_low_confidence_objects(
    model_type: str,
    model: ModelType,
    processor: ProcessorType,
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
    """Calculate confidence for detected objects and return those with low confidence.
    
    Args:
        model_type: Type of model being used
        model: Model instance
        processor: Model processor
        image: Input image
        detected_objects: List of objects to check confidence for
        confidence_threshold: Threshold below which objects are considered low confidence
        enable_layer_editing: Whether to enable layer editing
        edit_layer: Layer to edit if layer editing is enabled
        edit_weight: Weight for layer editing
        cluster_results_path: Path to cluster results
        num_clusters: Number of clusters to use
        use_mean: Whether to use mean pooling
        
    Returns:
        Tuple of (low confidence objects list, confidence scores dict)
    """
    confidence_scores, _, _ = calculate_internal_confidence(
        model_type=model_type,
        model=model,
        processor=processor,
        image=image,
        coco_objects=detected_objects,
        enable_layer_editing=enable_layer_editing,
        edit_layer=edit_layer,
        edit_weight=edit_weight,
        cluster_results_path=cluster_results_path,
        num_clusters=num_clusters,
        use_mean=use_mean
    )
    
    # Extract objects below threshold
    low_confidence_objects = [
        obj for obj, confidence in confidence_scores.items()
        if confidence < confidence_threshold
    ]
    
    logger.info(f"Low confidence objects: {low_confidence_objects}")
    return low_confidence_objects, confidence_scores

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the MME generation script.
    
    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Generate text with latent image/text embedding manipulation."
    )
    
    # Model configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--model_id", type=str, default="leloy/Anole-7b-v0.1-hf",
                           help="Model ID to load")
    model_group.add_argument("--model_type", type=str, default="chameleon",
                           choices=["chameleon", "Emu3", "janus"],
                           help="Type of model to use")
    model_group.add_argument("--model_cache_dir", type=str, default=None,
                           help="Model cache directory")
    
    # Generation configuration
    gen_group = parser.add_argument_group("Generation Configuration")
    gen_group.add_argument("--gen_type", type=str, default="sample",
                         choices=["sample", "opera", "vcd", "gnn", "sid", "gnn_opera",
                                 "gnn_vcd", "gnn_sid", "confidence", "confidence_gnn"],
                         help="Type of generation to perform")
    gen_group.add_argument("--prompt", type=str,
                         default="Please describe this image in detail.",
                         help="Prompt for caption generation")
    gen_group.add_argument("--max_new_tokens", type=int, default=256,
                         help="Maximum number of new tokens to generate")
    gen_group.add_argument("--seed", type=int, default=42,
                         help="Random seed for reproducibility")
    
    # Layer manipulation
    layer_group = parser.add_argument_group("Layer Manipulation")
    layer_group.add_argument("--layer", type=int, default=12,
                           help="Layer to perform the operation")
    layer_group.add_argument("--num_layers", type=int, default=1,
                           help="Number of layers to modify")
    layer_group.add_argument("--layer_weights", type=float, nargs='+', default=[1.0],
                           help="Weights for consecutive layers")
    layer_group.add_argument("--operation", type=str, default="subtract",
                           help="Operation to perform")
    layer_group.add_argument("--weight", type=float, default=1.0,
                           help="Weight for projection subtraction")
    
    # VCD configuration
    vcd_group = parser.add_argument_group("VCD Configuration")
    vcd_group.add_argument("--noise_step", type=int, default=500,
                         help="Number of noise steps for VCD")
    vcd_group.add_argument("--cd_alpha", type=float, default=0.5,
                         help="Alpha parameter for contrastive decoding")
    vcd_group.add_argument("--cd_beta", type=float, default=0.1,
                         help="Beta parameter for contrastive decoding")
    
    # GNN configuration
    gnn_group = parser.add_argument_group("GNN Configuration")
    gnn_group.add_argument("--cluster_results_path", type=str,
                         help="Path to cluster results (required for GNN generation)")
    gnn_group.add_argument("--num_clusters", type=int, default=2,
                         help="Number of top clusters to use")
    gnn_group.add_argument("--find_non_input_indices", action="store_true",
                         help="Whether to find indices that don't appear in the input image")
    
    # Confidence configuration
    conf_group = parser.add_argument_group("Confidence Configuration")
    conf_group.add_argument("--coco_objects_path", type=str,
                          default="data/coco_unique_objs.txt",
                          help="Path to COCO objects file")
    conf_group.add_argument("--confidence_threshold", type=float, default=0.5,
                          help="Threshold for confidence-based generation")
    conf_group.add_argument("--text_edit_layer", type=int, default=21,
                          help="Layer to perform the text edit")
    
    # Fast token merging
    fast_group = parser.add_argument_group("Fast Token Merging")
    fast_group.add_argument("--fast", action="store_true",
                          help="Use fast settings")
    fast_group.add_argument("--fast-v-inplace", action="store_true",
                          help="Use in-place operations for fast token merging")
    fast_group.add_argument("--fast-v-attention-rank", type=int, default=100,
                          help="Attention rank for fast token merging")
    fast_group.add_argument("--fast-v-attention-rank-add", type=int, default=100,
                          help="Additional attention rank for fast token merging")
    fast_group.add_argument("--fast-v-agg-layer", type=int, default=2,
                          help="Aggregation layer for fast token merging")
    fast_group.add_argument("--fast-v-sys-length", type=int, default=None,
                          help="System prompt length for fast token merging")
    fast_group.add_argument("--fast-v-image-token-length", type=int, default=None,
                          help="Image token length for fast token merging")
    
    # I/O configuration
    io_group = parser.add_argument_group("Input/Output Configuration")
    io_group.add_argument("--root_dir", type=str,
                        default=None,
                        help="Root directory for results")
    io_group.add_argument("--output_dir_name", type=str, default="Base",
                        help="Output directory name")
    io_group.add_argument("--image_dir", type=str,
                        default=None,
                        help="Image directory")
                        
    
    return parser.parse_args()

def get_generation_config(
    model_type: str,
    gen_type: str,
    args: argparse.Namespace,
    processor: ProcessorType,
    inputs: Dict[str, torch.Tensor],
    model: ModelType,
    key_position: Optional[Dict[str, torch.Tensor]] = None
) -> GenerationConfig:
    """Get appropriate generation config based on model type and generation mode.
    
    Args:
        model_type: Type of model being used
        gen_type: Type of generation to perform
        args: Command line arguments
        processor: Model processor
        inputs: Model inputs
        model: Model instance
        key_position: Optional key positions for attention
        
    Returns:
        GenerationConfig object configured for the specified generation type
    """
    if gen_type in ["vcd", "sid"]:
        return _get_vcd_config(model_type, gen_type, args, processor, inputs, model, key_position)
    elif gen_type == "opera":
        return _get_opera_config(model_type, args, processor, model, key_position)
    else:  # Default sampling or gnn
        return _get_default_config(model_type, args, processor, model)

def _get_vcd_config(
    model_type: str,
    gen_type: str,
    args: argparse.Namespace,
    processor: ProcessorType,
    inputs: Dict[str, torch.Tensor],
    model: ModelType,
    key_position: Optional[Dict[str, torch.Tensor]] = None
) -> GenerationConfig:
    """Get VCD-specific generation config."""
    # Prepare noisy image for VCD
    image_tensor = _prepare_image_tensor(inputs["pixel_values"], model_type, processor)
    
    if gen_type == "vcd":
        image_cd = add_diffusion_noise(image_tensor, args.noise_step)
    else:  # sid
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

def _get_opera_config(
    model_type: str,
    args: argparse.Namespace,
    processor: ProcessorType,
    model: ModelType,
    key_position: Optional[Dict[str, torch.Tensor]] = None
) -> GenerationConfig:
    """Get OPERA-specific generation config."""
    config = {
        "max_length": 600,
        "output_attentions": True,
        "num_beams": 5,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "opera_decoding": True,
        "key_position": key_position,
        "scale_factor": 50,
        "threshold": 15,
        "num_attn_candidates": 5,
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

def _get_default_config(
    model_type: str,
    args: argparse.Namespace,
    processor: ProcessorType,
    model: ModelType
) -> GenerationConfig:
    """Get default generation config."""
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

def _prepare_image_tensor(
    image_input: Union[str, torch.Tensor],
    model_type: str,
    processor: ProcessorType
) -> torch.Tensor:
    """Prepare image tensor for processing.
    
    Args:
        image_input: Image path or tensor
        model_type: Type of model being used
        processor: Model processor
        
    Returns:
        Processed image tensor
    """
    if isinstance(image_input, str):
        image = Image.open(image_input)
        image = image.convert("RGB")
        
        if model_type == "chameleon":
            dummy_inputs = processor(text="<image>", images=image, return_tensors="pt")
            return dummy_inputs["pixel_values"]
        elif model_type == "Emu3":
            return processor.image_processor([image], return_tensors="pt")["pixel_values"]
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
            return prepare_inputs["pixel_values"]
    else:
        return image_input

def main() -> None:
    """Main entry point for the MME generation script.
    
    This function:
    1. Parses command line arguments
    2. Loads the model and processor
    3. Processes input images and queries
    4. Generates responses using the specified generation method
    5. Saves results to output files
    """
    try:
        args = parse_arguments()
        set_seed(args.seed)  # Set random seed for reproducibility
        
        # Load model and processor
        logger.info(f"Loading model and processor from {args.model_id}")
        model, processor = load_model_and_processor(
            model_path=args.model_id,
            model_type=args.model_type,
            cache_dir=args.model_cache_dir
        )
        
        # Load COCO objects if needed for confidence-based generation
        coco_objects = None
        if args.gen_type in ["confidence", "confidence_gnn"]:
            coco_objects = load_coco_objects(args.coco_objects_path)
            logger.info(f"Loaded {len(coco_objects)} COCO object classes")
        
        # Setup directories
        output_dir = Path(args.root_dir) / args.output_dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        query_dir = Path(args.root_dir) / "Query"
        
        # Process each query file
        for query_file in query_dir.glob("*.txt"):
            output_file = output_dir / query_file.name
            process_query_file(
                query_file=query_file,
                output_file=output_file,
                model=model,
                processor=processor,
                args=args,
                coco_objects=coco_objects
            )
            
        logger.info("Finished generating captions")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)
        raise

def process_query_file(
    query_file: Path,
    output_file: Path,
    model: ModelType,
    processor: ProcessorType,
    args: argparse.Namespace,
    coco_objects: Optional[List[str]] = None
) -> None:
    """Process a single query file and generate responses.
    
    Args:
        query_file: Path to input query file
        output_file: Path to output file
        model: Model instance
        processor: Model processor
        args: Command line arguments
        coco_objects: Optional list of COCO objects for confidence-based generation
    """
    logger.info(f"Processing query file: {query_file}")
    
    with open(query_file, 'r') as fin, open(output_file, 'w') as fout:
        lines = fin.read().splitlines()
        filename = query_file.stem
        
        for line in tqdm.tqdm(lines, desc=f"Generating captions for {filename}"):
            try:
                img, query, gt = line.strip().split('\t')
                image_path = Path(args.image_dir) / filename / img
                
                if not image_path.exists():
                    raise FileNotFoundError(f"Image not found: {image_path}")
                
                response = generate_response(
                    image_path=image_path,
                    query=query,
                    model=model,
                    processor=processor,
                    args=args,
                    coco_objects=coco_objects
                )
                
                response = format_response(response)
                print(img, query, gt, response, sep='\t', file=fout)
                fout.flush()  # Force write to disk
                
            except Exception as e:
                logger.error(f"Error processing line: {line}\nError: {str(e)}")
                continue

def generate_response(
    image_path: Path,
    query: str,
    model: ModelType,
    processor: ProcessorType,
    args: argparse.Namespace,
    coco_objects: Optional[List[str]] = None
) -> str:
    """Generate a response for a single image and query.
    
    Args:
        image_path: Path to input image
        query: Input query text
        model: Model instance
        processor: Model processor
        args: Command line arguments
        coco_objects: Optional list of COCO objects for confidence-based generation
        
    Returns:
        Generated response text
    """
    # Load and process image
    image = load_image(str(image_path))
    
    if args.gen_type in ["confidence", "confidence_gnn"]:
        return generate_confidence_based_response(
            image=image,
            query=query,
            model=model,
            processor=processor,
            args=args,
            coco_objects=coco_objects
        )
    elif args.gen_type in ["gnn", "gnn_opera", "gnn_vcd", "gnn_sid"]:
        return generate_gnn_based_response(
            image_path=image_path,
            query=query,
            model=model,
            processor=processor,
            args=args
        )
    else:
        return generate_standard_response(
            image=image,
            query=query,
            model=model,
            processor=processor,
            args=args
        )

def generate_confidence_based_response(
    image: Image.Image,
    query: str,
    model: ModelType,
    processor: ProcessorType,
    args: argparse.Namespace,
    coco_objects: List[str]
) -> str:
    """Generate response using confidence-based approach.
    
    Args:
        image: Input image
        query: Input query
        model: Model instance
        processor: Model processor
        args: Command line arguments
        coco_objects: List of COCO objects
        
    Returns:
        Generated response text
    """
    # Generate baseline caption
    operation = "normal" if args.gen_type == "confidence" else "subtract"
    logger.info(f"Running {operation} generation for baseline caption")
    
    baseline_caption = run_latent_generation_with_gnn(
        model_type=args.model_type,
        model=model,
        processor=processor,
        image_1_path=image,
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
    
    # Extract COCO objects from caption
    detected_objects = extract_coco_objects_from_text(baseline_caption, coco_objects)
    logger.info(f"Detected objects: {detected_objects}")
    
    if not detected_objects:
        logger.info("No COCO objects detected in caption, using baseline caption")
        return baseline_caption
    
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
        cluster_results_path=args.cluster_results_path,
        num_clusters=args.num_clusters,
        use_mean=args.use_mean
    )
    
    if not low_confidence_objects:
        logger.info("No low confidence objects detected, using baseline caption")
        return baseline_caption
    
    logger.info(f"Generating without low confidence objects: {low_confidence_objects}")
    
    if args.gen_type == "confidence":
        return run_latent_generation(
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
    else:  # confidence_gnn
        return run_latent_generation_with_gnn(
            model_type=args.model_type,
            model=model,
            processor=processor,
            model_id=args.model_id,
            image_1_path=image,
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

def generate_gnn_based_response(
    image_path: Path,
    query: str,
    model: ModelType,
    processor: ProcessorType,
    args: argparse.Namespace
) -> str:
    """Generate response using GNN-based approach.
    
    Args:
        image_path: Path to input image
        query: Input query
        model: Model instance
        processor: Model processor
        args: Command line arguments
        
    Returns:
        Generated response text
    """
    if not args.cluster_results_path:
        raise ValueError("cluster_results_path must be provided for GNN generation")
        
    if args.model_type == "chameleon":
        query = f"You are a helpful assistant, please answer the question based on the image. {query[:-10]} Please answer yes when you think it is correct."
        
    return run_latent_generation_with_gnn(
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
        use_opera=args.gen_type == "gnn_opera",
        use_vcd=args.gen_type == "gnn_vcd",
        use_sid=args.gen_type == "gnn_sid"
    )

def generate_standard_response(
    image: Image.Image,
    query: str,
    model: ModelType,
    processor: ProcessorType,
    args: argparse.Namespace
) -> str:
    """Generate response using standard approach.
    
    Args:
        image: Input image
        query: Input query
        model: Model instance
        processor: Model processor
        args: Command line arguments
        
    Returns:
        Generated response text
    """
    # Process inputs based on model type
    inputs, key_position = process_model_inputs(
        model_type=args.model_type,
        processor=processor,
        image=image,
        query=query,
        model=model
    )
    
    # Configure model for generation
    if args.gen_type == "sid":
        configure_model_for_sid(model, args)
    
    # Get generation config
    generation_config = get_generation_config(
        model_type=args.model_type,
        gen_type=args.gen_type,
        args=args,
        processor=processor,
        inputs=inputs,
        model=model,
        key_position=key_position
    )
    
    # Generate response
    with torch.inference_mode():
        if args.model_type == "chameleon":
            return generate_chameleon_response(
                model=model,
                processor=processor,
                inputs=inputs,
                generation_config=generation_config,
                args=args
            )
        elif args.model_type == "Emu3":
            return generate_emu3_response(
                model=model,
                processor=processor,
                inputs=inputs,
                generation_config=generation_config,
                args=args
            )
        else:  # janus
            return generate_janus_response(
                model=model,
                processor=processor,
                inputs=inputs,
                generation_config=generation_config,
                args=args
            )

def process_model_inputs(
    model_type: str,
    processor: ProcessorType,
    image: Image.Image,
    query: str,
    model: ModelType
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Process inputs for different model types.
    
    Args:
        model_type: Type of model being used
        processor: Model processor
        image: Input image
        query: Input query
        model: Model instance
        
    Returns:
        Tuple of (processed inputs, key positions)
    """
    if model_type == "chameleon":
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
        
    elif model_type == "Emu3":
        inputs, image_start_list, image_end_list = processor(
            text=query,
            image=image,
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
        
    else:  # janus
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
    
    return inputs, key_position

def configure_model_for_sid(model: ModelType, args: argparse.Namespace) -> None:
    """Configure model for SID generation.
    
    Args:
        model: Model instance
        args: Command line arguments
    """
    if isinstance(model, ChameleonForConditionalGeneration):
        model.model.config.use_fast_v = True
        model.model.config.fast_v_inplace = args.fast_v_inplace
        model.model.config.fast_v_sys_length = args.fast_v_sys_length
        model.model.config.fast_v_image_token_length = args.fast_v_image_token_length
        model.model.config.fast_v_attention_rank = args.fast_v_attention_rank
        model.model.config.fast_v_attention_rank_add = args.fast_v_attention_rank_add
        model.model.config.fast_v_agg_layer = args.fast_v_agg_layer
        model.model.reset_fastv()
    else:
        model.language_model.model.config.use_fast_v = True
        model.language_model.model.config.fast_v_inplace = args.fast_v_inplace
        model.language_model.model.config.fast_v_sys_length = args.fast_v_sys_length
        model.language_model.model.config.fast_v_image_token_length = args.fast_v_image_token_length
        model.language_model.model.config.fast_v_attention_rank = args.fast_v_attention_rank
        model.language_model.model.config.fast_v_attention_rank_add = args.fast_v_attention_rank_add
        model.language_model.model.config.fast_v_agg_layer = args.fast_v_agg_layer
        model.language_model.model.reset_fastv()

def generate_chameleon_response(
    model: ModelType,
    processor: ProcessorType,
    inputs: Dict[str, torch.Tensor],
    generation_config: GenerationConfig,
    args: argparse.Namespace
) -> str:
    """Generate response using Chameleon model.
    
    Args:
        model: Model instance
        processor: Model processor
        inputs: Model inputs
        generation_config: Generation configuration
        args: Command line arguments
        
    Returns:
        Generated response text
    """
    if args.gen_type == "vcd":
        image_cd = tensor_to_pil(generation_config.vcd_inputs.squeeze(0))
        inputs_cd = processor(text=args.prompt + "<image>", images=image_cd, return_tensors="pt")
        inputs_cd_ids = inputs_cd.input_ids.to(model.device, dtype=torch.long)
        generation_config.vcd_inputs = inputs_cd_ids
    elif args.gen_type == "sid":
        generation_config.vcd_inputs = inputs["input_ids"]
        
    output = model.generate(
        **inputs,
        generation_config=generation_config,
    )
    
    if isinstance(output, dict):
        output = output["sequences"]
    
    return processor.decode(
        output[0][len(inputs["input_ids"][0]):], 
        skip_special_tokens=True
    )

def generate_emu3_response(
    model: ModelType,
    processor: ProcessorType,
    inputs: Dict[str, torch.Tensor],
    generation_config: GenerationConfig,
    args: argparse.Namespace
) -> str:
    """Generate response using Emu3 model.
    
    Args:
        model: Model instance
        processor: Model processor
        inputs: Model inputs
        generation_config: Generation configuration
        args: Command line arguments
        
    Returns:
        Generated response text
    """
    if args.gen_type == "vcd":
        generation_config.vcd_inputs = inputs_cd["input_ids"]
    elif args.gen_type == "sid":
        generation_config.vcd_inputs = inputs["input_ids"]
        
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
    return processor.batch_decode(outputs, skip_special_tokens=True)[0]

def generate_janus_response(
    model: ModelType,
    processor: ProcessorType,
    inputs: torch.Tensor,
    generation_config: GenerationConfig,
    args: argparse.Namespace
) -> str:
    """Generate response using Janus model.
    
    Args:
        model: Model instance
        processor: Model processor
        inputs: Model inputs
        generation_config: Generation configuration
        args: Command line arguments
        
    Returns:
        Generated response text
    """
    detached_inputs = inputs.detach().clone()
    detached_attention_mask = prepare_inputs.attention_mask.detach().clone()
    
    if args.gen_type == "vcd":
        image_cd = tensor_to_pil(generation_config.vcd_inputs.squeeze(0))
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{args.prompt}",
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
        generation_config.vcd_inputs = inputs_cd.detach().clone()
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
        
    return processor.tokenizer.decode(
        generated_sequence[0].cpu().tolist(), 
        skip_special_tokens=True
    )

if __name__ == "__main__":
    main() 
