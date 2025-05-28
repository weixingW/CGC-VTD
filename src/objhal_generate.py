import os
import glob
import re
import torch
import argparse
import sys
sys.path.append(".")
sys.path.append("..")
from transformers import (
    ChameleonForConditionalGeneration,
    ChameleonProcessor,
    set_seed,
    GenerationConfig,
)

import logging
import json
from typing import Dict, List, Optional, Tuple
import tqdm
from pathlib import Path
from latent_gen import run_latent_generation_with_gnn, run_latent_generation, load_models
from model_utils import load_model_and_processor
from vcd_utils.vcd_sample import evolve_vcd_sampling
from vcd_utils.vcd_add_noise import add_diffusion_noise
from torchvision import transforms
from PIL import Image
from internal_confidence import calculate_internal_confidence, load_coco_objects
from eval.chair import CHAIR



logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger()

def load_image(image_path):
    if image_path.startswith("http"):
        image = Image.open(requests.get(image_path, stream=True).raw)
    else:
        image = Image.open(image_path)
    return image

def extract(line):
    NEG_WORDS = ["No", "not", "no", "NO"]
    line = line.replace('.', '')
    line = line.replace(',', '')
    words = line.split(' ')
    if any(word in NEG_WORDS for word in words) or any(word.endswith("n't") for word in words):
        return "No"
    else:
        return "Yes"

def load_coco_objects(file_path):
    """Load COCO object classes from a text file."""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def extract_coco_objects_from_text(text, coco_objects):
    """Extract COCO objects mentioned in the text."""
    evaluator = CHAIR()
    words, node_words, idxs, double_words = evaluator.caption_to_words(text)

    return words




def tensor_to_pil(tensor):
    """Convert a tensor to PIL Image in RGB mode."""
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




def load_objhal_data(question_file: str) -> List[Dict]:
    """Load ObjHal data from JSONL file."""
    if not os.path.exists(question_file):
        raise FileNotFoundError(f"Question file not found: {question_file}")
        
    questions = [json.loads(q) for q in open(os.path.expanduser(question_file), "r")]
    logger.info(f"Loaded {len(questions)} questions from {question_file}")
    return questions

def get_chunk(questions, num_chunks, chunk_idx):
    """Split questions into chunks for parallel processing."""
    if num_chunks == 1:
        return questions
    
    chunk_size = len(questions) // num_chunks
    start_idx = chunk_idx * chunk_size
    end_idx = start_idx + chunk_size if chunk_idx < num_chunks - 1 else len(questions)
    return questions[start_idx:end_idx]

def save_objhal_response(question_idx, image_id, prompt, response, model_id, output_file: str):
    """Save ObjHal response to JSONL file."""
    
    with open(output_file, 'a') as f:
        f.write(json.dumps({
            "question_id": question_idx,
            "image_id": image_id,
            "prompt": prompt,
            "text": response,
            "model_id": model_id
        }) + "\n")

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate text with latent image/text embedding manipulation for ObjHal evaluation."
    )
    parser.add_argument(
        "--query_file",
        type=str,
        required=True,
        help="Path to ObjHal question file in JSONL format",
    )
    parser.add_argument("--image_dir", type=str, default="", help="Image directory (if images are not embedded)")
    parser.add_argument("--model_id", type=str, default="leloy/Anole-7b-v0.1-hf", help="Model ID")
    parser.add_argument("--operation", type=str, default="subtract", help="Operation to perform")
    parser.add_argument("--weight", type=float, default=0.2, help="Weight for projection subtraction")
    parser.add_argument("--use_mean", action="store_true", help="Use mean pooling for text embeddings")
    parser.add_argument("--fast", action="store_true", help="Use fast settings")
    parser.add_argument("--model_cache_dir", type=str, default=None, help="Model cache directory")
    parser.add_argument("--answers_file", type=str, required=True, help="Output file for answers")
    parser.add_argument("--prompt", type=str, default="", help="Additional prompt for generation")
    parser.add_argument("--layer", type=int, default=27, help="Layer to perform the operation")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of layers to modify")
    parser.add_argument("--num_clusters", type=int, default=2, help="Number of top clusters to use")
    parser.add_argument("--layer_weights", type=float, nargs='+', default=[1.0], help="Weights for consecutive layers")
    parser.add_argument("--gen_type", type=str, default="sample", 
                       choices=["sample", "opera", "vcd", "gnn", "sid", "gnn_opera", "gnn_vcd", "gnn_sid", "confidence", "confidence_gnn"],
                       help="Generation type: 'sample', 'opera', 'vcd', 'gnn', 'sid', 'gnn_opera', 'gnn_vcd', 'gnn_sid', 'confidence', or 'confidence_gnn'")
    parser.add_argument("--noise_step", type=int, default=500,
                       help="Number of noise steps for VCD")
    parser.add_argument("--cd_alpha", type=float, default=0.5,
                       help="Alpha parameter for contrastive decoding")
    parser.add_argument("--cd_beta", type=float, default=0.1,
                       help="Beta parameter for contrastive decoding")
    parser.add_argument("--cluster_results_path", type=str, 
                       help="Path to cluster results (required for GNN generation)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--find_non_input_indices", action="store_true", 
                       help="Whether to find indices that don't appear in the input image")
    parser.add_argument("--model_type", type=str, default="chameleon", help="Model type")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum number of new tokens")
    parser.add_argument("--num_chunks", type=int, default=1, help="Number of chunks to split the dataset")
    parser.add_argument("--chunk_idx", type=int, default=0, help="Index of the chunk to process")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p for sampling")
    parser.add_argument("--num_beams", type=int, default=5, help="Number of beams for beam search")

    # fast token merging
    parser.add_argument("--fast-v-inplace", default=False)
    parser.add_argument("--fast-v-attention-rank", type=int, default=100)
    parser.add_argument("--fast-v-attention-rank-add", type=int, default=100)
    parser.add_argument("--fast-v-agg-layer", type=int, default=2)
    # auto-generation
    parser.add_argument("--fast-v-sys-length", default=None, type=int, help='the length of system prompt')
    parser.add_argument("--fast-v-image-token-length", default=None, type=int, help='the length of image token')

    parser.add_argument("--coco_objects_path", type=str, default="data/coco_unique_objs.txt", 
                       help="Path to COCO objects file")
    parser.add_argument("--confidence_threshold", type=float, default=0.5,
                       help="Threshold for confidence-based generation")
    parser.add_argument("--text_edit_layer", type=int, default=21,
                       help="Layer to perform the text edit")
    parser.add_argument("--use_targets", type=bool, default=False,
                       help="Whether to use targets for GNN generation")
    parser.add_argument("--use_cls", type=bool, default=False,
                       help="Whether to use cls token for GNN generation")

    return parser.parse_args()

def get_generation_config(model_type, gen_type, args, processor, inputs, model, key_position=None):
    """Get appropriate generation config based on model type and generation mode."""
    
    if gen_type == "vcd" or gen_type == "sid":
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
            image = inputs["pixel_values"].convert("RGB")
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

        if gen_type == "vcd":
            image_cd = add_diffusion_noise(image_tensor, args.noise_step)
        elif gen_type == "sid":
            image_cd = image_tensor
        image_cd = image_cd.to(model.device, dtype=torch.bfloat16)
        
        config = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": True,
            #"temperature": 0.7,
            #"top_p": 0.9,
            "vcd_sample": True,
            "cd_alpha": args.cd_alpha,
            "cd_beta": args.cd_beta,
            "vcd_inputs": image_cd,
            "output_attentions": False,
            "output_hidden_states": False,
            "return_dict_in_generate": True,
            #"pad_token_id": model.config.pad_token_id if model.config.pad_token_id is not None else processor.tokenizer.pad_token_id,
            #"eos_token_id": model.config.eos_token_id if model.config.eos_token_id is not None else processor.tokenizer.eos_token_id
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
            "max_length": 600,
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
    """Calculate confidence for detected objects and return those with low confidence.
    Uses the calculate_internal_confidence function from internal_confidence.py.
    """
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

def main():
    args = parse_arguments()
    logger.info(f"Loading model and processor from {args.model_id}")
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Load model and processor based on model type
    model, processor = load_model_and_processor(
        model_path=args.model_id,
        model_type=args.model_type,
        cache_dir=args.model_cache_dir
    )
    
    # Load ObjHal questions
    questions = load_objhal_data(args.query_file)
    
    # Split questions into chunks if needed
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    
    # Load COCO objects if needed for confidence-based generation
    coco_objects = None
    if args.gen_type == "confidence" or args.gen_type == "confidence_gnn":
        coco_objects = load_coco_objects(args.coco_objects_path)
        logger.info(f"Loaded {len(coco_objects)} COCO object classes")
    
    # Create output directory if it doesn't exist
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    
    # Open the output file for writing
    with open(answers_file, 'w') as ans_file:
        for question_idx, question in enumerate(tqdm.tqdm(questions, desc="Processing questions")):
            # Get question text
            question_text = question["question"]
            
            # Get image from the question - either embedded or from path
            if 'image' in question:
                # Handle base64 encoded image
                import base64
                import io
                image_bytes = base64.b64decode(question["image"])
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                image_id = question.get('image_id', question.get('image_url', f"embedded_image_{question_idx}"))
            elif 'image_path' in question:
                # Handle image path
                image_path = question['image_path']
                image = Image.open(image_path).convert('RGB')
                image_id = question.get('image_id', question.get('image_url', os.path.basename(image_path)))
            elif 'image_id' in question:
                # Handle image_id with external directory
                image_id = question['image_id']
                # Construct image path from image_id and image_dir
                image_path = os.path.join(args.image_dir, f"{image_id}.jpg")
                
                # Try alternative extensions if the file doesn't exist
                if not os.path.exists(image_path):
                    for ext in ['.png', '.jpeg', '.JPEG', '.PNG']:
                        alt_path = os.path.join(args.image_dir, f"{image_id}{ext}")
                        if os.path.exists(alt_path):
                            image_path = alt_path
                            break
                
                if not os.path.exists(image_path):
                    raise ValueError(f"Image file not found for image_id: {image_id}")
                    
                image = Image.open(image_path).convert('RGB')
            else:
                raise ValueError(f"No image source found for question {question_idx}")
            
            # For confidence-based generation
            if args.gen_type == "confidence" or args.gen_type == "confidence_gnn":
                logger.info(f"Running confidence-based generation for question {question_idx}")
                if 'image' in question:
                    temp_image_path = f"/tmp/temp_image_{question_idx}.jpg"
                    image.save(temp_image_path)
                    image_path = temp_image_path

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
                    prompt=question_text,
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
                    if args.gen_type == "confidence":
                        enable_layer_editing = False
                    else:
                        enable_layer_editing = True
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
                        response = baseline_caption
                    else:
                        logger.info(f"Generating without low confidence objects: {low_confidence_objects}")


                        if args.gen_type == "confidence":
                            # Use run_latent_generation to generate without low confidence objects
                            response = run_latent_generation(
                                model_type=args.model_type,
                                model=model,
                                processor=processor,
                                image_1 = image,
                                text_input=low_confidence_objects,
                                prompt=question_text,
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
                                prompt=question_text,
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
            
            # For GNN-based generation types
            elif args.gen_type == "gnn" or args.gen_type.startswith("gnn_"):
                if not args.cluster_results_path:
                    raise ValueError("cluster_results_path must be provided for GNN generation")
                    
                # For GNN, we need to save the image temporarily if it's embedded
                if 'image' in question:
                    temp_image_path = f"/tmp/temp_image_{question_idx}.jpg"
                    image.save(temp_image_path)
                    image_path = temp_image_path
                
                response = run_latent_generation_with_gnn(
                    model_type=args.model_type,
                    model=model,
                    processor=processor,
                    model_id=args.model_id,
                    image_1_path=image_path,
                    prompt=question_text,
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
                    use_sid=args.gen_type == "gnn_sid",
                    use_targets=args.use_targets,
                    use_cls=args.use_cls
                )
                
                # Clean up temporary file if created
                if 'image' in question and os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
            
            # For other generation types
            else:
                # Process inputs based on model type
                if args.model_type == "chameleon":
                    inputs = processor(
                        text=question_text + "<image>",
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
                        text=question_text,
                        image=image,
                        mode="U",
                        return_tensors="pt",
                        padding="longest",
                    )
                    
                    image_tensor = processor.image_processor([image], return_tensors="pt")["pixel_values"]
                    image_cd = add_diffusion_noise(image_tensor, args.noise_step)
                    image_cd = image_cd.to(model.device, dtype=torch.bfloat16)
                    image_cd = tensor_to_pil(image_cd)
                    inputs_cd, _, _ = processor(
                        text=question_text,
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
                            "content": f"<image_placeholder>\n{question_text}",
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
                    inputs={"pixel_values": image, "text": question_text},
                    model=model,
                    key_position=key_position
                )
                
                # Update generation config with temperature and top_p
                if hasattr(generation_config, "temperature"):
                    generation_config.temperature = args.temperature
                if hasattr(generation_config, "top_p"):
                    generation_config.top_p = args.top_p
                if hasattr(generation_config, "num_beams"):
                    generation_config.num_beams = args.num_beams
                
                # Generate response
                with torch.inference_mode():
                    if args.model_type == "chameleon":
                        if args.gen_type == "vcd":
                            image_cd = tensor_to_pil(generation_config.vcd_inputs.squeeze(0))
                            inputs_cd = processor(text=question_text + "<image>", images=image_cd, return_tensors="pt")
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
                        if args.gen_type == "vcd":
                            generation_config.vcd_inputs = inputs_cd["input_ids"].to(model.device)
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
                        response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
                        
                    elif args.model_type == "janus":
                        detached_inputs = inputs.detach().clone()
                        detached_attention_mask = prepare_inputs.attention_mask.detach().clone()
                        if args.gen_type == "vcd":
                            # Detach and clone the tensors before passing to generate
                            image_cd = tensor_to_pil(generation_config.vcd_inputs.squeeze(0))
                            conversation = [
                                {
                                    "role": "<|User|>",
                                    "content": f"<image_placeholder>\n{question_text}",
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
                        
                        # Handle outputs whether they're a dictionary or tensor
                        if isinstance(outputs, dict):
                            generated_sequence = outputs.sequences
                        else:
                            generated_sequence = outputs
                            
                        response = processor.tokenizer.decode(
                            generated_sequence[0].cpu().tolist(), 
                            skip_special_tokens=True
                        )
            
            logger.info(f"Response: {response}")
            
            # Write response directly to the file in JSON format
            ans_file.write(json.dumps({
                "question_id": question_idx,
                "image_id": image_id,
                "prompt": question_text,
                "text": response,
                "model_id": args.model_id
            }) + "\n")
            ans_file.flush()
            
            # Clear CUDA cache between iterations to prevent memory leaks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    logger.info(f"Finished processing {len(questions)} questions. Results saved to {args.answers_file}")

if __name__ == "__main__":
    main() 
