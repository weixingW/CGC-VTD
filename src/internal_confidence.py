"""
Internal confidence calculation for vision-language models.
Modified from https://github.com/nickjiang2378/vl-interp/tree/main
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm
from transformers import GenerationConfig
import argparse

# Add local imports
sys.path.append(".")
sys.path.append("..")

from vcd_utils.vcd_add_noise import add_diffusion_noise
from model_utils import load_model_and_processor
from latent_gen import load_clustering_results, get_text_embeddings
from eval_utils import subtract_projection

# Constants
DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_CD_ALPHA = 0.5
DEFAULT_CD_BETA = 0.1
DEFAULT_FAST_V_ATTENTION_RANK = 100
DEFAULT_FAST_V_AGG_LAYER = 2

def get_cd_logits(
    next_token_logits: torch.Tensor,
    next_token_logits_cd: torch.Tensor,
    cd_alpha: float = DEFAULT_CD_ALPHA,
    cd_beta: float = DEFAULT_CD_BETA
) -> torch.Tensor:
    """
    Get the logits for contrastive decoding.
    
    Args:
        next_token_logits: Original token logits
        next_token_logits_cd: Contrastive decoding token logits
        cd_alpha: Alpha parameter for CD
        cd_beta: Beta parameter for CD
        
    Returns:
        Contrastive decoding logits
    """
    cutoff1 = torch.log(torch.tensor(cd_beta))
    cutoff = cutoff1 + next_token_logits.max(dim=-1, keepdim=True).values
    diffs = (1 + cd_alpha) * next_token_logits - cd_alpha * next_token_logits_cd
    return diffs

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a tensor to PIL Image in RGB mode.
    
    Args:
        tensor: Input tensor
        
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

def get_generation_config(
    model_type: str,
    gen_type: str,
    processor: Any,
    inputs: Dict[str, Any],
    model: Any,
    key_position: Optional[Dict[str, torch.Tensor]] = None
) -> GenerationConfig:
    """
    Get appropriate generation config based on model type and generation mode.
    
    Args:
        model_type: Type of model (e.g. "chameleon", "Emu3", "janus", "llava_next")
        gen_type: Type of generation (e.g. "vcd", "sid", "opera", "gnn")
        processor: Model processor
        inputs: Input dictionary containing pixel values and text
        model: Model instance
        key_position: Optional dictionary containing key positions for generation
        
    Returns:
        GenerationConfig object configured for the specified model and generation type
        
    Raises:
        ValueError: If model_type or gen_type is not supported
    """
    if gen_type in ["vcd", "sid"]:
        inputs["pixel_values"] = str(inputs["pixel_values"])
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
            image_cd = add_diffusion_noise(image_tensor, 500)
        elif gen_type == "sid":
            image_cd = image_tensor
        image_cd = image_cd.to(model.device, dtype=torch.bfloat16)
        
        config = {
            "max_new_tokens": DEFAULT_MAX_TOKENS,
            "do_sample": True,
            "vcd_sample": True,
            "cd_alpha": DEFAULT_CD_ALPHA,
            "cd_beta": DEFAULT_CD_BETA,
            "vcd_inputs": image_cd,
            "output_attentions": False,
            "output_hidden_states": True,
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
            "num_beams": 2,
            "max_new_tokens": DEFAULT_MAX_TOKENS,
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
            "max_new_tokens": DEFAULT_MAX_TOKENS,
            "do_sample": True,
            "temperature": DEFAULT_TEMPERATURE,
            "top_p": DEFAULT_TOP_P,
            "output_hidden_states": True,
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

def internal_confidence(
    model_type: str,
    tokenizer: Any,
    softmax_probs: np.ndarray,
    class_: str,
    top_probs: List[float],
    top_tokens: List[str]
) -> float:
    """
    Calculate the maximum probability of a class in the softmax probabilities across all layers and positions.
    
    Args:
        model_type: Type of model (e.g. "chameleon", "Emu3", "janus", "llava_next")
        tokenizer: Model tokenizer
        softmax_probs: Softmax probabilities array of shape (num_layers, batch_size, sequence_length, vocab_size)
                       or (batch_size, sequence_length, vocab_size)
        class_: Class name to calculate confidence for
        top_probs: List of top probabilities
        top_tokens: List of top tokens
        
    Returns:
        Maximum probability of the class across all layers and positions
        
    Raises:
        ValueError: If softmax_probs has unexpected dimensions
    """
    if model_type == "chameleon":
        class_token_indices = tokenizer.encode(class_)[1:]
        class_token_indices = [token_id - 16384 for token_id in class_token_indices]
    elif model_type == "Emu3":
        # Check if class_ is in any of the decoded tokens
        class_lower = class_.lower()
        for i, token in enumerate(top_tokens):
            if class_lower in token.lower():
                return top_probs[i]
        
        # Class not found in top tokens, return very small confidence
        return 1e-6
        
    elif model_type == "janus":
        class_token_indices = tokenizer.encode(class_)[1:]
    elif model_type == "llava_next":
        class_token_indices = tokenizer.encode(class_)[1:]

    # Handle different array dimensions
    if softmax_probs.ndim == 4:  # 4D array
        return softmax_probs[:,:,:, class_token_indices].max()
    elif softmax_probs.ndim == 3:  # 3D array
        return softmax_probs[:,:, class_token_indices].max()
    elif softmax_probs.ndim == 2:  # 2D array
        return softmax_probs[:, class_token_indices].max()
    else:
        raise ValueError(f"Unexpected softmax_probs array dimension: {softmax_probs.ndim}")

def load_coco_objects(file_path: str) -> List[str]:
    """
    Load COCO object classes from a text file.
    
    Args:
        file_path: Path to the text file containing COCO object classes
        
    Returns:
        List of COCO object class names
    """
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def get_coco_ground_truth(image_id: str, coco_ann_path: str) -> List[str]:
    """
    Get ground truth object labels for a specific COCO image ID.
    
    Args:
        image_id: COCO image ID
        coco_ann_path: Path to COCO annotation file
        
    Returns:
        List of ground truth object labels
        
    Raises:
        Exception: If there is an error loading COCO annotations
    """
    try:
        coco = COCO(coco_ann_path)
    except Exception as e:
        print(f"Error loading COCO annotations: {e}")
        return []
    
    # Get annotation IDs for the image
    image_id = int(image_id)
    ann_ids = coco.getAnnIds(imgIds=[image_id])
    if not ann_ids:
        print(f"No annotations found for image ID: {image_id}")
        return []
    
    # Load annotations
    anns = coco.loadAnns(ann_ids)
    
    # Get category IDs and convert to category names
    category_ids = [ann['category_id'] for ann in anns]
    category_names = [coco.loadCats(cat_id)[0]['name'] for cat_id in category_ids]
    
    # Return unique category names
    return list(set(category_names))

def get_image_token_positions(model_type, model, processor, image_path=None,image=None):
    """Get the start and end positions of image tokens."""
    if image_path is not None:
        image = Image.open(image_path).convert("RGB")
    else:
        image = image
    
    if model_type == "chameleon":
        
        # Use a dummy prompt for Chameleon
        inputs = processor(text="<image>", images=image, return_tensors="pt")
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
        image_token_pos = (inputs["input_ids"] == image_token_id).nonzero()[0, 1].item()
        
        return {
            "image_start": image_token_pos,
            "image_end": image_token_pos + 1024,  # Chameleon uses 1024 image tokens
            "response_start": inputs["input_ids"].shape[-1]
        }
    
    elif model_type == "Emu3":
        
        inputs, image_start_list, image_end_list = processor(
            text="",
            image=image,
            mode="U",
            return_tensors="pt",
            padding="longest"
        )
        
        return {
            "image_start": image_start_list[0],
            "image_end": image_end_list[0],
            "response_start": inputs["input_ids"].shape[-1]
        }
    
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
        
        img_start_id = processor.image_start_id
        img_end_id = processor.image_end_id
        img_start_pos = (prepare_inputs["input_ids"] == img_start_id).nonzero()[0, 1].item()
        img_end_pos = (prepare_inputs["input_ids"] == img_end_id).nonzero()[0, 1].item()
        
        return {
            "image_start": img_start_pos+1,
            "image_end": img_end_pos,
            "response_start": prepare_inputs["input_ids"].shape[-1]
        }

    elif model_type == "llava_next":
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Write a detailed description."},
                ],
            },
        ]
        
        formatted_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=formatted_prompt, return_tensors="pt")
        
        # In LLAVA-NEXT, image tokens start after the prompt tokens
        prompt_length = inputs["input_ids"].shape[1]
        
        # LLAVA-NEXT embeds image tokens internally, so we need to identify
        # where the image tokens would appear in the hidden states
        return {
            "image_start": 5,  # Typical start position after prompt tokens
            "image_end": prompt_length - 14,  # Typical end position before generation tokens
            "response_start": prompt_length
        }

def calculate_internal_confidence(model_type, model, processor,  coco_objects, 
                                 enable_layer_editing=False, edit_layer=15, edit_operation="subtract", 
                                 edit_weight=1.0, edit_text=None, find_non_input_indices=False, 
                                 cluster_results_path=None, num_clusters=2, use_mean=True,image_path=None,image=None,
                                 layer_weights=None, num_layers=None, least_important=False, gen_type="gnn", query=None):
    """Calculate internal confidence for all COCO objects."""
    assert query is not None, "Query must be provided"
    # Disable layer editing for LLaVA-Next as it's not compatible
    if model_type == "llava_next" and enable_layer_editing:
        print("Warning: Layer editing is not supported for LLaVA-NEXT. Disabling layer editing.")
        enable_layer_editing = False
    
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype  # Get model's dtype
    if image_path is not None:
        image = Image.open(image_path).convert("RGB")
    else:
        image = image
    
    # Get token positions
    positions = get_image_token_positions(model_type, model, processor, image_path=image_path,image=image)
    positions["image_end"] = positions["image_end"]
    positions["image_start"] = positions["image_start"]
    
    # Setup for layer editing if enabled
    if gen_type == "gnn":
        # Get embeddings for layer editing
        if edit_text is not None:
            # Use provided text for editing
            text_embeddings = get_text_embeddings(model_type, model, edit_text, processor, edit_layer)
        elif cluster_results_path is not None:
            # Get cluster embeddings similar to latent_gen.py
            print(f"Loading clustering results from {cluster_results_path}")
            results = load_clustering_results(cluster_results_path)
            
            # Get quantized indices based on model type
            if model_type == "chameleon":
                # Process image and get codebook usage
                dummy_prompt = "<image>"
                inputs = processor(text=dummy_prompt, images=image, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(device, dtype=dtype)
                
                # Get quantized representation
                with torch.no_grad():
                    encoder_output = model.model.vqmodel.encoder(pixel_values)
                    hidden_states = model.model.vqmodel.quant_conv(encoder_output)
                    quant, _, indices = model.model.vqmodel.quantize(hidden_states)
                
                # Get indices and convert to numpy
                indices = indices.cpu().numpy().flatten()
                
            elif model_type == "janus":
                # Janus-specific codebook extraction
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
                ).to(device)
                pixel_values = prepare_inputs.pixel_values.squeeze(0).to(device, dtype=dtype)
                
                with torch.no_grad():
                    vqmodel = model.gen_vision_model
                    encoder_output = vqmodel.encoder(pixel_values)
                    hidden_states = vqmodel.quant_conv(encoder_output)
                    _, _, info = vqmodel.quantize(hidden_states)
                    indices = info[2]
                
                indices = indices.cpu().numpy().flatten()
            elif model_type == "Emu3":
                pixel_values = processor.image_processor([image], return_tensors="pt")["pixel_values"]
                pixel_values = pixel_values.to(device=processor.vision_tokenizer.device, dtype=processor.vision_tokenizer.dtype)
                
                ndim = pixel_values.ndim
                if ndim == 4:
                    t = processor.vision_tokenizer.config.temporal_downsample_factor
                    b, c, h, w = pixel_values.shape
                    pixel_values = pixel_values.unsqueeze(1).repeat(1, t, 1, 1, 1)
                elif ndim == 5:
                    b, t, c, h, w = pixel_values.shape
                    
                with torch.no_grad():
                    encoder_output = processor.vision_tokenizer.encoder(pixel_values)
                    encoder_output = encoder_output.permute(0, 2, 1, 3, 4)
                    hidden_state = processor.vision_tokenizer.quant_conv(encoder_output)
                    hidden_state = hidden_state.permute(0, 2, 1, 3, 4)
                    indices = processor.vision_tokenizer.quantize(hidden_state)
                if ndim == 4:
                    indices = indices.squeeze(1)
                
                indices = indices.flatten().cpu().numpy()
            
            # Get cluster analysis
            cluster_labels = results['kmeans']['labels']
            image_cluster_labels = cluster_labels[indices]
            
            # Find the most frequent clusters
            unique_clusters, counts = np.unique(image_cluster_labels, return_counts=True)
            if least_important:
                top_clusters = unique_clusters[np.argsort(counts)][-num_clusters:]
            else:
                top_clusters = unique_clusters[np.argsort(-counts)][:num_clusters]
            
            # Get codebook indices corresponding to these clusters
            top_cluster_indices = []
            for cluster in top_clusters:
                cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster]
                cluster_indices_set = set(cluster_indices)
                input_indices_set = set(indices)
                
                if find_non_input_indices:
                    # Get indices that belong to this cluster but do NOT exist in the input image
                    filtered_indices = list(cluster_indices_set.difference(input_indices_set))
                    top_cluster_indices.extend(filtered_indices)
                else:
                    # Get indices that belong to this cluster and exist in the input image
                    filtered_indices = list(cluster_indices_set.intersection(input_indices_set))
                    top_cluster_indices.extend(filtered_indices)
            
            if not top_cluster_indices:
                print("No valid cluster indices found. Disabling layer editing.")
                enable_layer_editing = False
            else:
                # Get embeddings for these indices
                print(f"Using top cluster tokens as text inputs: {top_cluster_indices[:5]}...")
                if model_type == "chameleon":
                    text_inputs = [processor.tokenizer.decode([idx]) for idx in top_cluster_indices]
                    text_embeddings = get_text_embeddings(model_type, model, text_inputs, processor, edit_layer)
                else:
                    text_embeddings = get_text_embeddings(model_type, model, top_cluster_indices, processor, edit_layer)
        else:
            print("No text or cluster path provided for layer editing. Disabling.")
            enable_layer_editing = False
    
    cd_probs = None
    # Process image based on model type
    if model_type == "chameleon":
        # Prepare inputs for generation
        inputs = processor(text=f"<image>{query} In the image, there is a", images=image, return_tensors="pt")
        inputs["input_ids"] = inputs['input_ids'][:,:-1]
        inputs["attention_mask"] = inputs['attention_mask'][:,:-1]
        
        
        inputs = {
            "input_ids": inputs["input_ids"].to(device, dtype=torch.long),
            "attention_mask": inputs["attention_mask"].to(device, dtype=torch.long),
            "pixel_values": inputs["pixel_values"].to(device, dtype=dtype)
        }
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
        image_token_pos = (inputs["input_ids"] == image_token_id).nonzero()[0, 1].item()
        key_position = {
            "image_start": torch.tensor(image_token_pos).to(model.device),
            "image_end": torch.tensor(image_token_pos + 1024).to(model.device),
            "response_start": torch.tensor(inputs["input_ids"].shape[-1]).to(model.device),
        }
        if gen_type == "sid":
            model.model.config.use_fast_v = True
            model.model.config.fast_v_inplace = False
            model.model.config.fast_v_sys_length = None
            model.model.config.fast_v_image_token_length = None
            model.model.config.fast_v_attention_rank = 100
            model.model.config.fast_v_attention_rank_add = 100
            model.model.config.fast_v_agg_layer = 2
            
        else:
            model.model.config.use_fast_v = False
        model.model.reset_fastv()
        
        # Setup hook for layer editing if enabled
        if gen_type == "gnn":
            start_idx = positions["image_start"]
            end_idx = positions["image_end"]
            
            def edit_hook_fn(module, input):
                hidden_states = input[0]
                img_embeddings = hidden_states[:, start_idx:end_idx, :]
                
                modified = img_embeddings.clone()
                for i in range(text_embeddings.shape[0]):
                    curr_embedding = text_embeddings[i:i+1]
                    curr_embedding = curr_embedding.to(modified.device)
                    
                    if edit_operation == "subtract":
                        modified = subtract_projection(modified, curr_embedding, weight=edit_weight, use_mean=use_mean)
                    elif edit_operation == "add":
                        modified = subtract_projection(modified, curr_embedding, weight=-edit_weight, use_mean=use_mean)
                
                hidden_states[:, start_idx:end_idx, :] = modified
                return tuple([hidden_states] + list(input[1:]))
            
            # Register the hook
            hook_handle = model.model.layers[edit_layer].register_forward_pre_hook(edit_hook_fn)
        
        generation_config = get_generation_config(
            model_type="chameleon",
            gen_type=gen_type,
            processor=processor,
            inputs={"pixel_values": image_path, "text": query},
            model=model,
            key_position=key_position
        )
        if gen_type == "vcd":
            image_cd = tensor_to_pil(generation_config.vcd_inputs.squeeze(0))
            inputs_cd = processor(text=f"<image>{query} In the image, there is a", images=image_cd, return_tensors="pt")
            inputs_cd_ids = inputs_cd.input_ids[:,:-1].to(device, dtype=torch.long)
            generation_config.vcd_inputs = inputs_cd_ids
        elif gen_type == "sid":
            generation_config.vcd_inputs = inputs["input_ids"]
            
            

        # Generate with the model to get hidden states
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                generation_config=generation_config
            )
        
        # print the generated text
        response = processor.decode(outputs.sequences[0], skip_special_tokens=True)
        print(f"Chameleon caption: {response}")
            
        # Remove hook if it was registered
        if gen_type == "gnn" and 'hook_handle' in locals():
            hook_handle.remove()
            
        # Get hidden states from generation output
        hidden_states = torch.stack(outputs.hidden_states[0])  # Get the hidden states from first generation step
        if gen_type == "vcd" or gen_type == "sid":
            cd_hidden_states = torch.stack(outputs.cd_hidden_states[0])
        else:
            cd_hidden_states = hidden_states
        image_hidden_states = hidden_states[:, :, positions["image_start"]:positions["image_end"], :]
        next_token_hidden_states = hidden_states[-1:, :, -1:, :]
        cd_next_token_hidden_states = cd_hidden_states[-1:, :, -1:, :]
        
        # Get logits from hidden states using model's lm_head
        logits = model.lm_head(image_hidden_states).to("cpu").float()
        logits_next_token = model.lm_head(next_token_hidden_states).to("cpu").float()
        logits_next_token_cd = model.lm_head(cd_next_token_hidden_states).to("cpu").float()
        cd_logits = get_cd_logits(logits_next_token, logits_next_token_cd)

        logits_scores = torch.nn.functional.log_softmax(logits[...,16384:], dim=-1)
        logits_next_token_scores = torch.nn.functional.log_softmax(logits_next_token[...,16384:], dim=-1)
        cd_logits_scores = torch.nn.functional.log_softmax(cd_logits[...,16384:], dim=-1)
        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(logits_scores, dim=-1)
        probs_next_token = torch.nn.functional.softmax(logits_next_token_scores, dim=-1)
        probs_next_token_cd = torch.nn.functional.softmax(cd_logits_scores, dim=-1)

        
    elif model_type == "Emu3":
        # Prepare inputs for generation
        inputs, image_start_list, image_end_list = processor(
            text=query,
            image=image,
            mode="U",
            return_tensors="pt",
            padding="longest"
        )
        inputs = {
            "input_ids": inputs["input_ids"].to(device, dtype=torch.long),
            "attention_mask": inputs["attention_mask"].to(device, dtype=torch.long),
        }
        key_position = {
            "image_start": torch.tensor(image_start_list[0]).to(model.device),
            "image_end": torch.tensor(image_end_list[0]).to(model.device),
            "response_start": torch.tensor(inputs["input_ids"].shape[-1]).to(model.device),
        }
        if gen_type == "sid":
            model.model.config.use_fast_v = True
            model.model.config.fast_v_inplace = False
            model.model.config.fast_v_sys_length = None
            model.model.config.fast_v_image_token_length = None
            model.model.config.fast_v_attention_rank = 100
            model.model.config.fast_v_attention_rank_add = 100
            model.model.config.fast_v_agg_layer = 2
        else:
            model.model.config.use_fast_v = False
        model.model.reset_fastv()
        generation_config = get_generation_config(
                model_type="Emu3",
                gen_type=gen_type,
                processor=processor,
                inputs={"pixel_values": image_path, "text": query},
                model=model,
                key_position=key_position)
        
            
        # Setup hook for layer editing if enabled
        if gen_type == "gnn":
            start_idx = positions["image_start"]
            end_idx = positions["image_end"]
            
            def edit_hook_fn(module, input):
                hidden_states = input[0]
                img_embeddings = hidden_states[:, start_idx:end_idx, :]
                
                modified = img_embeddings.clone()
                for i in range(text_embeddings.shape[0]):
                    curr_embedding = text_embeddings[i:i+1]
                    curr_embedding = curr_embedding.to(modified.device)
                    
                    if edit_operation == "subtract":
                        modified = subtract_projection(modified, curr_embedding, weight=edit_weight, use_mean=use_mean)
                    elif edit_operation == "add":
                        modified = subtract_projection(modified, curr_embedding, weight=-edit_weight, use_mean=use_mean)
                
                hidden_states[:, start_idx:end_idx, :] = modified
                return tuple([hidden_states] + list(input[1:]))
            
            # Register the hook
            hook_handle = model.model.layers[edit_layer].register_forward_pre_hook(edit_hook_fn)
        if gen_type == "vcd":
            image_cd = tensor_to_pil(generation_config.vcd_inputs.squeeze(0))
            inputs_cd, _, _ = processor(
                text=query,
                image=image_cd,
                mode="U",
                return_tensors="pt",
                padding="longest"
            )
            inputs_cd_ids = inputs_cd["input_ids"].to(device, dtype=torch.long)
            generation_config.vcd_inputs = inputs_cd_ids
            assert inputs_cd_ids.shape == inputs["input_ids"].shape, "Input IDs shape mismatch"
        elif gen_type == "sid":
            generation_config.vcd_inputs = inputs["input_ids"]
            
        # Generate with the model to get hidden states
        with torch.inference_mode():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                generation_config=generation_config
            )
            
        # print the generated text
        response = processor.decode(outputs.sequences[0], skip_special_tokens=True)
        print(f"Emu3 caption: {response}")
        
        # Remove hook if it was registered
        if gen_type == "gnn" and 'hook_handle' in locals():
            hook_handle.remove()
            
        # Get hidden states from generation output
        hidden_states = torch.stack(outputs.hidden_states[0])  # Get the hidden states from first generation step
        cd_hidden_states = torch.stack(outputs.cd_hidden_states[0])
        image_hidden_states = hidden_states[:, :, positions["image_start"]:positions["image_end"], :]
        next_token_hidden_states = hidden_states[-1:, :, -1:, :]
        cd_next_token_hidden_states = cd_hidden_states[-1:, :, -1:, :]
        # Process in chunks along sequence dimension
        chunk_size = 256  # Can be adjusted based on memory constraints
        seq_len = image_hidden_states.shape[2]
        chunks = []
        
        for i in range(0, seq_len, chunk_size):
            chunk = image_hidden_states[..., i:i+chunk_size, :]
            chunk_logits = model.lm_head(chunk).to("cpu").float()
            chunk_scores = torch.nn.functional.log_softmax(chunk_logits[...,:151643], dim=-1)
            chunks.append(chunk_scores)
        
        logits_next_token = model.lm_head(next_token_hidden_states).to("cpu").float()
        logits_next_token_cd = model.lm_head(cd_next_token_hidden_states).to("cpu").float()
        cd_logits = get_cd_logits(logits_next_token, logits_next_token_cd)
        logits_next_token_scores = torch.nn.functional.log_softmax(logits_next_token[...,:151643], dim=-1)
        cd_logits_scores = torch.nn.functional.log_softmax(cd_logits[...,:151643], dim=-1)
            
        # Concatenate chunks back together
        logits_scores = torch.cat(chunks, dim=2)
        
        # Apply softmax
        probs = torch.nn.functional.softmax(logits_scores, dim=-1)
        probs_next_token = torch.nn.functional.softmax(logits_next_token_scores, dim=-1)
        probs_next_token_cd = torch.nn.functional.softmax(cd_logits_scores, dim=-1)
        
    
    elif model_type == "janus":
        # Prepare inputs for generation
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{query}",
                "images": [image],
            },
            {"role": "<|Assistant|>", "content": "In the image, there is a"},
        ]
        
        prepare_inputs = processor(
            conversations=conversation,
            images=[image],
            force_batchify=True
        ).to(device)
        prepare_inputs.input_ids = prepare_inputs.input_ids[:,:-1]
        prepare_inputs.attention_mask = prepare_inputs.attention_mask[:,:-1]
        prepare_inputs.images_seq_mask = prepare_inputs.images_seq_mask[:,:-1]
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
        if gen_type == "sid":
            model.language_model.model.config.use_fast_v = True
            model.language_model.model.config.fast_v_inplace = False
            model.language_model.model.config.fast_v_sys_length = None
            model.language_model.model.config.fast_v_image_token_length = None
            model.language_model.model.config.fast_v_attention_rank = 100
            model.language_model.model.config.fast_v_attention_rank_add = 100
            model.language_model.model.config.fast_v_agg_layer = 2
        else:
            model.language_model.model.config.use_fast_v = False
        model.language_model.model.reset_fastv()
        generation_config = get_generation_config(
            model_type="janus",
            gen_type=gen_type,
            processor=processor,
            inputs={"pixel_values": image_path, "text": query},
            model=model,
            key_position=key_position
        )
        
        
        # Setup hook for layer editing if enabled
        if gen_type == "gnn":
            start_idx = positions["image_start"]
            end_idx = positions["image_end"]
            
            def edit_hook_fn(module, input):
                hidden_states = input[0]
                img_embeddings = hidden_states[:, start_idx:end_idx, :]
                
                modified = img_embeddings.clone()
                for i in range(text_embeddings.shape[0]):
                    curr_embedding = text_embeddings[i:i+1]
                    curr_embedding = curr_embedding.to(modified.device)
                    
                    if edit_operation == "subtract":
                        modified = subtract_projection(modified, curr_embedding, weight=edit_weight, use_mean=use_mean)
                    elif edit_operation == "add":
                        modified = subtract_projection(modified, curr_embedding, weight=-edit_weight, use_mean=use_mean)
                
                hidden_states[:, start_idx:end_idx, :] = modified
                return tuple([hidden_states] + list(input[1:]))
            
            # Register the hook
            hook_handle = model.language_model.model.layers[edit_layer].register_forward_pre_hook(edit_hook_fn)
        if gen_type == "vcd":
            
            image_cd = tensor_to_pil(generation_config.vcd_inputs.squeeze(0))
            conversation = [
                {
                    "role": "<|User|>",
                    "content": f"<image_placeholder>\n{query}",
                    "images": [image_cd],
                },
                {"role": "<|Assistant|>", "content": "In the image, there is a"},
            ]
            prepare_inputs_cd = processor(
                conversations=conversation,
                images=[image_cd],
                return_tensors="pt"
            ).to(model.device)
            prepare_inputs_cd.input_ids = prepare_inputs_cd.input_ids[:,:-1]
            prepare_inputs_cd.attention_mask = prepare_inputs_cd.attention_mask[:,:-1]
            prepare_inputs_cd.images_seq_mask = prepare_inputs_cd.images_seq_mask[:,:-1]
            inputs_cd = model.prepare_inputs_embeds(**prepare_inputs_cd)
            inputs_cd = inputs_cd.detach().clone()
            generation_config.vcd_inputs=inputs_cd
        if gen_type == "sid":
            generation_config.vcd_inputs = inputs.detach().clone()
        
        # Generate with the model to get hidden states
        detached_inputs = inputs.detach().clone()
        detached_attention_mask = prepare_inputs.attention_mask.detach().clone()
        
        with torch.inference_mode():
            outputs = model.language_model.generate(
                inputs_embeds=detached_inputs,
                attention_mask=detached_attention_mask,
                generation_config=generation_config
            )
            
        # print the generated text
        if isinstance(outputs, dict):
            generated_sequence = outputs.sequences
        else:
            generated_sequence = outputs
            
        response = processor.tokenizer.decode(
            generated_sequence[0].cpu().tolist(), 
            skip_special_tokens=True
        )
        print(f"Janus caption: {response}")
        # Remove hook if it was registered
        if gen_type == "gnn" and 'hook_handle' in locals():
            hook_handle.remove()
            
        # Get hidden states from generation output
        hidden_states = torch.stack(outputs.hidden_states[0])  # Get the hidden states from first generation step
        if gen_type == "vcd" or gen_type == "sid":
            cd_hidden_states = torch.stack(outputs.cd_hidden_states[0])
        else:
            cd_hidden_states = hidden_states
        image_hidden_states = hidden_states[:, :, positions["image_start"]:positions["image_end"], :]
        next_token_hidden_states = hidden_states[-1:, :, -1:, :]
        cd_next_token_hidden_states = cd_hidden_states[-1:, :, -1:, :] 
        # Get logits
        logits = model.language_model.lm_head(image_hidden_states).to("cpu").float()
        logits_next_token = model.language_model.lm_head(next_token_hidden_states).to("cpu").float()
        logits_next_token_cd = model.language_model.lm_head(cd_next_token_hidden_states).to("cpu").float()
        cd_logits = get_cd_logits(logits_next_token, logits_next_token_cd)
        logits_scores = torch.nn.functional.log_softmax(logits, dim=-1)
        logits_next_token_scores = torch.nn.functional.log_softmax(logits_next_token, dim=-1)
        cd_logits_scores = torch.nn.functional.log_softmax(cd_logits, dim=-1)
        # Apply softmax
        probs = torch.nn.functional.softmax(logits_scores, dim=-1)
        probs_next_token = torch.nn.functional.softmax(logits_next_token_scores, dim=-1)
        probs_next_token_cd = torch.nn.functional.softmax(cd_logits_scores, dim=-1)

    elif model_type == "llava_next":
        cd_probs = None
        from methods.llava_next_utils import retrieve_logit_lens_llava_next
        
        # Create state dictionary for llava_next utilities
        state = {
            "model": model,
            "processor": processor
        }
        
        
        
        # Get logit lens analysis which gives us softmax probabilities
        if gen_type == "vcd":
            response, image_token_region, next_token_region, next_token_region_cd = retrieve_logit_lens_llava_next(state, image, query, use_vcd=True)
        else:
            response, image_token_region, next_token_region,_ = retrieve_logit_lens_llava_next(state, image, query, use_vcd=False)
        print(f"LLAVA-NEXT caption: {response}")
        
        # Convert to torch tensor
        probs = torch.from_numpy(image_token_region)
        probs_next_token = torch.from_numpy(next_token_region)
        probs = probs.permute(2, 1, 0) #(seq_len,num_layer,vocab) -> (vocab,num_layer,seq_len)
        probs_next_token = probs_next_token.permute(2, 1, 0)[:,-1:,:] 
        if gen_type == "vcd":
            probs_next_token_cd = torch.from_numpy(next_token_region_cd)
            probs_next_token_cd = probs_next_token_cd.permute(2, 1, 0)[:,-1:,:] 
        else:
            probs_next_token_cd = probs_next_token
        # Remove hook if it was registered
        if gen_type == "gnn" and 'hook_handle' in locals():
            hook_handle.remove()
    
    # Convert to CPU and numpy for easier processing
    probs = probs.detach().numpy()
    probs_next_token = probs_next_token.detach().numpy()
    
    
    # Get top 100 tokens with highest probabilities across all layers and positions
    tokenizer = processor.tokenizer
    
    # Find top 100 tokens with highest probabilities
    if model_type == "chameleon":
        # For chameleon, we're already looking at the relevant vocab section
        flattened_probs = probs.reshape(-1, probs.shape[-1])
        flattened_probs_next_token = probs_next_token.reshape(-1, probs_next_token.shape[-1])
        flattened_probs_next_token_cd = probs_next_token_cd.reshape(-1, probs_next_token_cd.shape[-1]).detach().numpy()

        top_indices = np.argsort(flattened_probs.max(axis=0))[-1000:][::-1]
        top_indices_next_token = np.argsort(flattened_probs_next_token.max(axis=0))[-1000:][::-1]
        top_indices_next_token_cd = np.argsort(flattened_probs_next_token_cd.max(axis=0))[-1000:][::-1]

        if cd_probs is not None:
            flattened_cd_probs = cd_probs.reshape(-1, cd_probs.shape[-1])
            top_indices_cd = np.argsort(flattened_cd_probs.max(axis=0))[-1000:][::-1]
            token_ids_cd = [idx + 16384 for idx in top_indices_cd]
        # Need to add offset for the actual token IDs
        token_ids = [idx + 16384 for idx in top_indices]
        token_ids_next_token = [idx + 16384 for idx in top_indices_next_token]
        token_ids_next_token_cd = [idx + 16384 for idx in top_indices_next_token_cd]
    elif model_type == "Emu3":
        flattened_probs = probs.reshape(-1, probs.shape[-1])
        flattened_probs_next_token = probs_next_token.reshape(-1, probs_next_token.shape[-1])
        flattened_probs_next_token_cd = probs_next_token_cd.reshape(-1, probs_next_token_cd.shape[-1]).detach().numpy()
        top_indices = np.argsort(flattened_probs.max(axis=0))[-1000:][::-1]
        top_indices_next_token = np.argsort(flattened_probs_next_token.max(axis=0))[-1000:][::-1]
        top_indices_next_token_cd = np.argsort(flattened_probs_next_token_cd.max(axis=0))[-1000:][::-1]
        # Need to add offset for the actual token IDs
        token_ids = top_indices
        token_ids_next_token = top_indices_next_token
        token_ids_next_token_cd = top_indices_next_token_cd
        if cd_probs is not None:
            flattened_cd_probs = cd_probs.reshape(-1, cd_probs.shape[-1])
            top_indices_cd = np.argsort(flattened_cd_probs.max(axis=0))[-1000:][::-1]
            token_ids_cd = top_indices_cd
    elif model_type == "janus":
        flattened_probs = probs.reshape(-1, probs.shape[-1])
        flattened_probs_next_token = probs_next_token.reshape(-1, probs_next_token.shape[-1])
        flattened_probs_next_token_cd = probs_next_token_cd.reshape(-1, probs_next_token_cd.shape[-1]).detach().numpy()
        top_indices = np.argsort(flattened_probs.max(axis=0))[-1000:][::-1]
        top_indices_next_token = np.argsort(flattened_probs_next_token.max(axis=0))[-1000:][::-1]
        top_indices_next_token_cd = np.argsort(flattened_probs_next_token_cd.max(axis=0))[-1000:][::-1]
        token_ids = top_indices
        token_ids_next_token = top_indices_next_token
        token_ids_next_token_cd = top_indices_next_token_cd
        if cd_probs is not None:
            flattened_cd_probs = cd_probs.reshape(-1, cd_probs.shape[-1])
            top_indices_cd = np.argsort(flattened_cd_probs.max(axis=0))[-1000:][::-1]
            token_ids_cd = top_indices_cd
    elif model_type == "llava_next":
        flattened_probs = probs.reshape(-1, probs.shape[-1])
        top_indices = np.argsort(flattened_probs.max(axis=0))[-1000:][::-1]
        token_ids = top_indices

        flattened_probs_next_token = probs_next_token.reshape(-1, probs_next_token.shape[-1])
        top_indices_next_token = np.argsort(flattened_probs_next_token.max(axis=0))[-1000:][::-1]
        token_ids_next_token = top_indices_next_token

        flattened_probs_next_token_cd = probs_next_token_cd.reshape(-1, probs_next_token_cd.shape[-1])
        top_indices_next_token_cd = np.argsort(flattened_probs_next_token_cd.squeeze())[-1000:].flip(0)
        token_ids_next_token_cd = top_indices_next_token_cd
    
    # Get probabilities for top tokens - handle different array dimensions
    top_probs = []
    for idx in top_indices:
        if idx >= probs.shape[-1]:
            top_probs.append(0.0)
            continue
            
        if probs.ndim == 4:  # 4D array (layers, batch, seq, vocab)
            top_probs.append(float(probs[:,:,:,idx].max()))
        elif probs.ndim == 3:  # 3D array (batch, seq, vocab) - for LLAVA-NEXT
            top_probs.append(float(probs[:,:,idx].max()))
        else:
            raise ValueError(f"Unexpected probs array dimension: {probs.ndim}")
    top_probs_next_token = []
    for idx in top_indices_next_token:
        if idx >= probs_next_token.shape[-1]:
            top_probs_next_token.append(0.0)
            continue
        if probs_next_token.ndim == 4:  # 4D array (layers, batch, seq, vocab)
            top_probs_next_token.append(float(probs_next_token[:,:,:,idx].max()))
        elif probs_next_token.ndim == 3:  # 3D array (batch, seq, vocab) - for LLAVA-NEXT
            top_probs_next_token.append(float(probs_next_token[:,:,idx].max()))
        else:
            raise ValueError(f"Unexpected probs_next_token array dimension: {probs_next_token.ndim}")
    top_probs_cd = []
    if cd_probs is not None:
        
        for idx in top_indices_cd:
            if idx >= cd_probs.shape[-1]:
                top_probs_cd.append(0.0)
                continue
            top_probs_cd.append(float(cd_probs[:,idx].max()))

    top_probs_next_token_cd = []
    
    for idx in top_indices_next_token_cd:
        
        if idx >= probs_next_token_cd.shape[-1]:
            top_probs_next_token_cd.append(0.0)
            continue
        if probs_next_token_cd.ndim == 4:  # 4D array (layers, batch, seq, vocab)
            top_probs_next_token_cd.append(float(probs_next_token_cd[:,:,:,idx].max()))
        elif probs_next_token_cd.ndim == 3:  # 3D array (batch, seq, vocab) - for LLAVA-NEXT
            top_probs_next_token_cd.append(float(probs_next_token_cd[:,:,idx].max()))
        else:
            raise ValueError(f"Unexpected probs_next_token_cd array dimension: {probs_next_token_cd.ndim}")
    # Decode tokens
    top_tokens = []
    for token_id in token_ids:
        try:
            token = tokenizer.decode([token_id])
            top_tokens.append(token)
        except:
            top_tokens.append(f"<ID:{token_id}>")
    
    top_tokens_next_token = []
    for token_id in token_ids_next_token:
        try:
            token = tokenizer.decode([token_id])
            top_tokens_next_token.append(token)
        except:
            top_tokens_next_token.append(f"<ID:{token_id}>")
    top_tokens_cd = []
    if cd_probs is not None:
        for token_id in token_ids_cd:
            try:
                token = tokenizer.decode([token_id])
                top_tokens_cd.append(token)
            except:
                top_tokens_cd.append(f"<ID:{token_id}>")
    top_tokens_next_token_cd = []
    for token_id in token_ids_next_token_cd:
        try:
            token = tokenizer.decode([token_id])
            top_tokens_next_token_cd.append(token)
        except:
            top_tokens_next_token_cd.append(f"<ID:{token_id}>")
    # Calculate confidence for each COCO object
    confidence_scores = {}
    confidence_scores_next_token = {}
    confidence_scores_cd = {}
    confidence_scores_next_token_cd = {}
    for obj in tqdm(coco_objects, desc=f"Calculating confidence for {model_type}"):
        confidence = internal_confidence(model_type, tokenizer, probs, obj, top_probs, top_tokens)
        confidence_next_token = internal_confidence(model_type, tokenizer, probs_next_token, obj, top_probs_next_token, top_tokens_next_token)
        confidence_next_token_cd = internal_confidence(model_type, tokenizer, probs_next_token_cd, obj, top_probs_next_token_cd, top_tokens_next_token_cd)
        confidence_scores[obj] = float(confidence)
        confidence_scores_next_token[obj] = float(confidence_next_token)
        confidence_scores_next_token_cd[obj] = float(confidence_next_token_cd)
        if cd_probs is not None:
            confidence_cd = internal_confidence(model_type, tokenizer, cd_probs, obj, top_probs_cd, top_tokens_cd)
            confidence_scores_cd[obj] = float(confidence_cd)
    
    
    return response,confidence_scores, top_tokens, top_probs, confidence_scores_next_token, top_tokens_next_token, top_probs_next_token, confidence_scores_next_token_cd, top_tokens_next_token_cd, top_probs_next_token_cd

def plot_confidence_comparison(
    confidence_results: Dict[str, Dict[str, float]],
    output_dir: str
) -> pd.DataFrame:
    """
    Plot confidence comparison across models.
    
    Args:
        confidence_results: Dictionary mapping model names to their confidence scores
        output_dir: Directory to save the plot and data
        
    Returns:
        DataFrame containing the confidence scores
    """
    # Create a DataFrame for easier plotting
    df = pd.DataFrame(confidence_results)
    
    # Sort by average confidence across models
    avg_confidence = df.mean(axis=1)
    sorted_indices = avg_confidence.sort_values(ascending=False).index
    df = df.loc[sorted_indices]
    
    # Plot settings
    plt.figure(figsize=(15, 10))
    df.plot(kind='bar', figsize=(15, 8))
    plt.title('Confidence Comparison Across Models')
    plt.xlabel('COCO Objects')
    plt.ylabel('Confidence Score')
    plt.xticks(rotation=90)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, 'confidence_comparison.png'))
    
    # Also save the raw data
    df.to_csv(os.path.join(output_dir, 'confidence_scores.csv'))
    
    return df

def plot_individual_confidence(
    confidence_results: Dict[str, Dict[str, float]],
    output_dir: str,
    ground_truth: Optional[List[str]] = None,
    type: str = "confidence"
) -> None:
    """
    Create individual bar plots for each model's confidence scores.
    
    Args:
        confidence_results: Dictionary mapping model names to their confidence scores
        output_dir: Directory to save the plots and data
        ground_truth: Optional list of ground truth object labels
        type: Type of confidence being plotted (e.g. "confidence", "next_token")
    """
    os.makedirs(os.path.join(output_dir, 'individual_plots'), exist_ok=True)
    
    for model_name, scores in confidence_results.items():
        fig_name = f'{model_name}_{type}.png'
        csv_name = f'{model_name}_{type}.csv'
        
        # Convert to dataframe and sort by confidence
        df = pd.DataFrame(list(scores.items()), columns=['Object', 'Confidence'])
        df = df.sort_values('Confidence', ascending=False)
        
        # Plot settings
        plt.figure(figsize=(12, 8))
        plt.bar(df['Object'], df['Confidence'], color='skyblue')
        plt.title(f'{type} Scores for {model_name}')
        plt.xlabel('COCO Objects')
        plt.ylabel(f'{type} Score')
        plt.xticks(rotation=90)
        plt.tight_layout()
        
        # Add horizontal grid lines for better readability
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Highlight ground truth objects if provided
        if ground_truth:
            # Create a mask for ground truth objects
            gt_mask = df['Object'].isin(ground_truth)
            gt_df = df[gt_mask]
            
            if not gt_df.empty:
                # Highlight ground truth objects
                plt.bar(gt_df['Object'], gt_df['Confidence'], color='orange')
                
                # Add labels for ground truth objects
                for i, row in gt_df.iterrows():
                    plt.annotate(f"{row['Confidence']:.4f}",
                                xy=(row['Object'], row['Confidence']),
                                xytext=(0, 5),
                                textcoords='offset points',
                                ha='center',
                                fontweight='bold')
                
                # Add legend
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='skyblue', label='Other Objects'),
                    Patch(facecolor='orange', label='Ground Truth Objects')
                ]
                plt.legend(handles=legend_elements, loc='upper right')
        
        # Save the plot
        plot_path = os.path.join(output_dir, 'individual_plots', fig_name)
        plt.savefig(plot_path)
        plt.close()
        
        # Also save sorted data as CSV
        csv_path = os.path.join(output_dir, 'individual_plots', csv_name)
        df.to_csv(csv_path, index=False)
        
    print(f"Individual confidence plots saved to {os.path.join(output_dir, 'individual_plots')}")

def plot_top_tokens(
    model_tokens_probs: Dict[str, Tuple[List[str], List[float]]],
    output_dir: str,
    type: str = "confidence"
) -> None:
    """
    Plot top tokens for each model.
    
    Args:
        model_tokens_probs: Dictionary mapping model names to tuples of (tokens, probabilities)
        output_dir: Directory to save the plots and data
        type: Type of confidence being plotted (e.g. "confidence", "next_token")
    """
    os.makedirs(os.path.join(output_dir, 'top_tokens'), exist_ok=True)
    
    for model_name, (tokens, probs) in model_tokens_probs.items():
        fig_name = f'{model_name}_top_{type}.png'
        csv_name = f'{model_name}_top_{type}.csv'
       
        # Plot top tokens
        plt.figure(figsize=(12, 8))
        plt.bar(range(len(tokens)), probs, color='skyblue')
        plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
        plt.title(f'Top 100 Tokens with Highest Confidence for {model_name}')
        plt.xlabel('Tokens')
        plt.ylabel('Confidence')
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(output_dir, 'top_tokens', fig_name)
        plt.savefig(plot_path)
        plt.close()
        
        # Save data as CSV
        df = pd.DataFrame({'Token': tokens, 'Confidence': probs})
        csv_path = os.path.join(output_dir, 'top_tokens', csv_name)
        df.to_csv(csv_path, index=False)
    
    print(f"Top token plots saved to {os.path.join(output_dir, 'top_tokens')}")

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Calculate internal confidence for COCO objects')
    
    # Input/output paths
    parser.add_argument('--image_path', type=str, 
                       default=None,
                       help='Path to image directory')
    parser.add_argument('--image_id', type=str, 
                       help='COCO image ID to get ground truth labels')
    parser.add_argument('--coco_ann_path', type=str,
                       default=None,
                       help='Path to COCO annotation file')
    parser.add_argument('--output_dir', type=str,
                       default=None,
                       help='Output directory')
    parser.add_argument('--coco_objects_path', type=str,
                       default=None,
                       help='Path to COCO objects file')
    parser.add_argument('--model_cache_dir', type=str,
                       default=None,
                       help='Model cache directory')
    
    # Model configurations
    parser.add_argument('--chameleon_model_id', type=str,
                       default='leloy/Anole-7b-v0.1-hf',
                       help='Chameleon model ID')
    parser.add_argument('--emu3_model_id', type=str,
                       default='BAAI/Emu3-Chat',
                       help='Emu3 model ID')
    parser.add_argument('--janus_model_id', type=str,
                       default='deepseek-ai/Janus-Pro-7B',
                       help='Janus model ID')
    parser.add_argument('--llava_next_model_id', type=str,
                       default='llava-hf/llava-v1.6-vicuna-7b-hf',
                       help='LLAVA-NEXT model ID')
    
    # Layer editing parameters
    parser.add_argument('--enable_layer_editing', action='store_true',
                       help='Enable layer editing during confidence calculation')
    parser.add_argument('--edit_layer', type=int, default=21,
                       help='Layer to apply editing')
    parser.add_argument('--edit_operation', type=str,
                       choices=['add', 'subtract'], default='subtract',
                       help='Operation for layer editing')
    parser.add_argument('--edit_weight', type=float, default=0.2,
                       help='Weight for the editing operation')
    parser.add_argument('--edit_text', type=str, nargs='+',
                       help='Text to use for editing (optional)')
    parser.add_argument('--find_non_input_indices', action='store_true',
                       help='Find indices not in input image for editing')
    parser.add_argument('--num_clusters', type=int, default=2,
                       help='Number of clusters to use for editing')
    parser.add_argument('--use_mean', type=bool, default=True,
                       help='Use mean pooling for projection')
    parser.add_argument('--layer_weights', type=float, nargs='+',
                       default=[1.0],
                       help='Weights for consecutive layers')
    parser.add_argument('--num_layers', type=int, default=1,
                       help='Number of layers to modify')
    parser.add_argument('--least_important', type=bool, default=False,
                       help='Use least important clusters for editing')
    parser.add_argument('--query', type=str,
                       default="Please describe the image.",
                       help='Query for LLaVA-NEXT')
    parser.add_argument('--gen_type', type=str, default="gnn",
                       help='Generation type')
    
    return parser.parse_args()

def main() -> None:
    """
    Main function to calculate and plot internal confidence scores.
    """
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load COCO objects
    coco_objects = load_coco_objects(args.coco_objects_path)
    print(f"Loaded {len(coco_objects)} COCO object classes")
    image_path = Path(args.image_path) / f"COCO_val2014_{args.image_id}.jpg"
    
    # Get ground truth objects if image ID is provided
    ground_truth = []
    if args.image_id:
        ground_truth = get_coco_ground_truth(args.image_id, args.coco_ann_path)
        print(f"\nGround truth objects for image ID {args.image_id}:")
        for obj in ground_truth:
            print(f"  - {obj}")
        print()  # Empty line for better readability
    
    # Model configurations
    model_configs = [
        {"type": "chameleon", "id": args.chameleon_model_id,
         "gnn_cluster_results_path": "clustering_results/chameleon_coco_4/clustering_results.pkl"},
        {"type": "janus", "id": args.janus_model_id,
         "gnn_cluster_results_path": "clustering_results/janus_coco_4/clustering_results.pkl"},
    ]
    
    # Initialize results dictionaries
    confidence_results = {}
    confidence_results_next_token = {}
    model_tokens_probs = {}
    model_tokens_probs_next_token = {}
    model_tokens_probs_next_token_cd = {}
    confidence_results_next_token_cd = {}

    # Calculate confidence for each model
    for config in model_configs:
        model_type = config["type"]
        model_id = config["id"]
        print(f"Processing {model_type} model: {model_id}")
        
        # Load model and processor
        if model_type == "llava_next":
            from methods.llava_next_utils import load_llava_next_state
            state = load_llava_next_state(model_path=model_id)
            model = state["model"]
            processor = state["processor"]
        else:
            model, processor = load_model_and_processor(
                model_type=model_type,
                model_path=model_id,
                cache_dir=args.model_cache_dir
            )
        
        # Calculate confidence with layer editing option
        response, confidence_scores, top_tokens, top_probs, \
        confidence_scores_next_token, top_tokens_next_token, top_probs_next_token, \
        confidence_scores_next_token_cd, top_tokens_next_token_cd, top_probs_next_token_cd = calculate_internal_confidence(
            model_type=model_type,
            model=model,
            processor=processor,
            image_path=image_path,
            coco_objects=coco_objects,
            enable_layer_editing=args.enable_layer_editing,
            edit_layer=args.edit_layer,
            edit_operation=args.edit_operation,
            edit_weight=args.edit_weight,
            edit_text=args.edit_text,
            find_non_input_indices=args.find_non_input_indices,
            cluster_results_path=config["gnn_cluster_results_path"],
            num_clusters=args.num_clusters,
            use_mean=args.use_mean,
            layer_weights=args.layer_weights,
            num_layers=args.num_layers,
            least_important=args.least_important,
            query=args.query,
            gen_type=args.gen_type
        )
        
        # Store results
        confidence_results[model_type] = confidence_scores
        confidence_results_next_token[model_type] = confidence_scores_next_token
        confidence_results_next_token_cd[model_type] = confidence_scores_next_token_cd
        model_tokens_probs[model_type] = (top_tokens[:50], top_probs[:50])
        model_tokens_probs_next_token[model_type] = (top_tokens_next_token[:50], top_probs_next_token[:50])
        model_tokens_probs_next_token_cd[model_type] = (top_tokens_next_token_cd[:50], top_probs_next_token_cd[:50])
        
        # Clean up to free memory
        del model, processor
        torch.cuda.empty_cache()
    
    # Plot results
    plot_top_tokens(model_tokens_probs, args.output_dir, type="confidence")
    plot_top_tokens(model_tokens_probs_next_token, args.output_dir, type="next_token")
    plot_top_tokens(model_tokens_probs_next_token_cd, args.output_dir, type="next_token_cd")
    plot_individual_confidence(confidence_results, args.output_dir, ground_truth, type="confidence")
    plot_individual_confidence(confidence_results_next_token, args.output_dir, ground_truth, type="next_token")
    plot_individual_confidence(confidence_results_next_token_cd, args.output_dir, ground_truth, type="next_token_cd")
    
    # Compare with ground truth if available
    if ground_truth:
        print("\nComparison with ground truth objects:")
        for model_type in confidence_results:
            scores = confidence_results[model_type]
            gt_scores = {obj: scores.get(obj, 0.0) for obj in ground_truth}
            print(f"\n{model_type} scores for ground truth objects:")
            for obj, score in sorted(gt_scores.items(), key=lambda x: x[1], reverse=True):
                print(f"  {obj}: {score:.4f}")
    
    # Print top 10 objects with highest confidence for each model
    print("\nTop 10 objects with highest confidence:")
    for model_type in confidence_results:
        scores = confidence_results[model_type]
        top_objects = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"\n{model_type}:")
        for obj, score in top_objects:
            print(f"  {obj}: {score:.4f}")

if __name__ == "__main__":
    main()
