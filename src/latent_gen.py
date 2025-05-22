import argparse
import logging
import os
import random
from typing import List, Literal, Optional, Tuple, Union
from PIL import Image
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cgc'))
import torch
from transformers import GenerationConfig
from transformers import (
    ChameleonForConditionalGeneration,
    ChameleonProcessor,
    set_seed,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoImageProcessor,
    AutoModel,
)
import json
import pickle
sys.path.append(".")
sys.path.append("..")
from eval_utils import load_image, subtract_projection
from cgc.cgc_utils import CodebookClusterer
from janus.models import MultiModalityCausalLM, VLChatProcessor
from emu3.mllm.processing_emu3 import Emu3Processor
from model_utils import load_model_and_processor

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger()

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

def load_clustering_results(results_path):
    """Load clustering results from pickle file"""
    with open(results_path, "rb") as f:
        results = pickle.load(f)
    return results

def get_image_embeddings(
    model_type: Literal["chameleon", "janus", "Emu3"],
    model: Union[ChameleonForConditionalGeneration, MultiModalityCausalLM, AutoModelForCausalLM],
    image: torch.Tensor,
    processor: Union[ChameleonProcessor, VLChatProcessor, Emu3Processor],
    layer: int,
) -> Tuple[torch.Tensor, int, int]:
    """Get image embeddings at a specific layer.
    
    Args:
        model: The Chameleon or Janus model
        image: PIL Image or path to image
        processor: The Chameleon or Janus processor
        layer: Which layer to extract embeddings from
        
    Returns:
        Tuple of:
        - Image embeddings at specified layer
        - Start index of image tokens
        - End index of image tokens
    """
    if model_type == "chameleon":
        # Process image with processor using dummy prompt
        dummy_prompt = "<image>"
        inputs = processor(text=dummy_prompt, images=image, return_tensors="pt")
        
        # Convert inputs to proper types
        inputs = {
            "input_ids": inputs["input_ids"].to(model.device, dtype=torch.long),
            "attention_mask": inputs["attention_mask"].to(model.device, dtype=torch.long),
            "pixel_values": inputs["pixel_values"].to(model.device, dtype=model.dtype)
        }
        
        # Find image token position
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
        image_token_pos = (inputs["input_ids"] == image_token_id).nonzero()[0, 1].item()
        
        # Image tokens come by replacing the <image> token
        image_start_idx = image_token_pos 
        # End index is determined by the model's image token sequence length 1024 for chameleon
        image_end_idx = image_start_idx + 1024
        
        # Storage for embeddings
        embeddings = None
        
        def hook_fn(module, input, output):
            nonlocal embeddings
            embeddings = output[0].detach().clone()
            
        # Register hook on specified layer
        handle = model.model.layers[layer].register_forward_hook(hook_fn)
        
        # Forward pass to get embeddings
        with torch.no_grad():
            model(**inputs)
            
        # Remove hook
        handle.remove()
        
    elif model_type == "janus":
        # Prepare conversation format for Janus
        conversation = [
            {
                "role": "<|User|>",
                "content": "<image_placeholder>",
                "images": [image],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        
        # Process inputs
        pil_images = [image] if isinstance(image, Image.Image) else [load_image(image)]
        prepare_inputs = processor(
            conversations=conversation, 
            images=pil_images, 
            force_batchify=True
        )
        
        # Convert tensors to device and dtype
        prepare_inputs.input_ids = prepare_inputs.input_ids.to(model.device)
        prepare_inputs.attention_mask = prepare_inputs.attention_mask.to(model.device)
        prepare_inputs.pixel_values = prepare_inputs.pixel_values.to(model.device, dtype=model.dtype)
        if hasattr(prepare_inputs, 'images_seq_mask'):
            prepare_inputs.images_seq_mask = prepare_inputs.images_seq_mask.to(model.device)
        if hasattr(prepare_inputs, 'images_emb_mask'):
            prepare_inputs.images_emb_mask = prepare_inputs.images_emb_mask.to(model.device)
        
        # Get embeddings
        embeddings = None
        
        def hook_fn(module, input, output):
            nonlocal embeddings
            embeddings = output[0].detach().clone()
            
        # Register hook on specified layer
        handle = model.language_model.model.layers[layer].register_forward_hook(hook_fn)
        
        # Run image encoder to get the image embeddings
        with torch.no_grad():
            inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
            model.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
            )
            
        # Remove hook
        handle.remove()
        
        
        image_start_id = processor.image_start_id
        image_end_id = processor.image_end_id
        
        # find the index of the image start and end ids in the embeddings
        image_start_idx = (prepare_inputs.input_ids == image_start_id).nonzero(as_tuple=True)[1].item()
        image_end_idx = (prepare_inputs.input_ids == image_end_id).nonzero(as_tuple=True)[1].item()
        
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")
    
    return embeddings[:, image_start_idx:image_end_idx, :]

def get_text_embeddings(
    model_type: Literal["chameleon", "janus", "Emu3"],
    model: Union[ChameleonForConditionalGeneration, MultiModalityCausalLM, AutoModelForCausalLM],
    text: Union[str, List[str], torch.Tensor,List[torch.Tensor]],
    processor: Union[ChameleonProcessor, VLChatProcessor, Emu3Processor],
    layer: int,
) -> torch.Tensor:
    """Get text embeddings at a specific layer.
    
    Args:
        model: The Chameleon model
        text: text tokens or codebook indices
        processor: The Chameleon processor
        layer: Which layer to extract embeddings from
        
    Returns:
        Text embeddings at specified layer, excluding special tokens
    """
    if isinstance(text, str):
        text = [text]
    
    all_embeddings = []
    for t in text:
        if model_type == "chameleon":
            # Process text input
            inputs = processor(text=t, return_tensors="pt")
            inputs = {
                "input_ids": inputs["input_ids"].to(model.device, dtype=torch.long),
                "attention_mask": inputs["attention_mask"].to(model.device, dtype=torch.long),
            }
            
            # Get special token IDs
            bos_token_id = processor.tokenizer.bos_token_id
            eos_token_id = processor.tokenizer.eos_token_id
            pad_token_id = processor.tokenizer.pad_token_id
            special_token_ids = {bos_token_id, eos_token_id, pad_token_id}
            
            # Find positions of actual text tokens (excluding special tokens)
            input_ids = inputs["input_ids"][0]
            text_mask = torch.tensor([id not in special_token_ids for id in input_ids])
            text_positions = text_mask.nonzero().squeeze(-1)
            
            if len(text_positions) == 0:
                logger.warning(f"No non-special tokens found in text: {t}")
                continue
            
            # Storage for embeddings
            embeddings = None
            
            def hook_fn(module, input):
                nonlocal embeddings
                # Only keep embeddings for text tokens
                embeddings = input[0][:, text_positions, :].detach().clone()
                
            # Register hook on specified layer
            handle = model.model.layers[layer].register_forward_pre_hook(hook_fn)
            
            # Forward pass to get embeddings
            with torch.no_grad():
                model(**inputs)
                
            # Remove hook
            handle.remove()
            
            if embeddings is not None:
                # Mean pool across text token positions to get a single embedding vector
                mean_embedding = embeddings.mean(dim=1, keepdim=True)  # Shape: (1, 1, D)
                all_embeddings.append(mean_embedding)
        elif model_type == "janus":
            if isinstance(t, str):
                t = processor.tokenizer.encode(t, add_special_tokens=False)[0]
                inputs_embeds = model.language_model.get_input_embeddings()(torch.tensor([t], device=model.device)).unsqueeze(0)
            else:
                text_embeddings = model.gen_embed(torch.tensor([t], device=model.device))
                inputs_embeds = model.gen_aligner(text_embeddings.unsqueeze(0)) # (B, T, D) (1,1,D)
            attention_mask = torch.ones(1,1).to(model.device)
            
            # Get embeddings
            embeddings = None
            
            def hook_fn(module, input):
                nonlocal embeddings
                embeddings = input[0].detach().clone()

            handle = model.language_model.model.layers[layer].register_forward_pre_hook(hook_fn)

            with torch.no_grad():
                model.language_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                )
            
            handle.remove()
            
            if embeddings is not None:
                # Mean pool across text token positions to get a single embedding vector
                mean_embedding = embeddings.mean(dim=1, keepdim=True)  # Shape: (1, 1, D)
                all_embeddings.append(mean_embedding)
        elif model_type == "Emu3":
            # For Emu3, t is a codebook index that needs to be converted to a visual token
            if isinstance(t, (int, np.integer)):
                # Convert codebook index to visual token string using the pattern
                visual_token = processor.visual_template[0].format(token_id=t)
                t = visual_token
            
            # Process text/visual token input
            inputs = processor.tokenizer(text=t, return_tensors="pt")
            inputs = {
                "input_ids": inputs["input_ids"].to(model.device, dtype=torch.long),
                "attention_mask": inputs["attention_mask"].to(model.device, dtype=torch.long),
            }
            
            # Get special token IDs
            bos_token_id = processor.tokenizer.bos_token_id
            eos_token_id = processor.tokenizer.eos_token_id
            pad_token_id = processor.tokenizer.pad_token_id
            special_token_ids = {bos_token_id, eos_token_id, pad_token_id}
            
            # Find positions of visual tokens (excluding special tokens)
            input_ids = inputs["input_ids"][0]
            text_mask = torch.tensor([id not in special_token_ids for id in input_ids])
            text_positions = text_mask.nonzero().squeeze(-1)
            
            if len(text_positions) == 0:
                logger.warning(f"No non-special tokens found in visual token: {t}")
                continue
            
            # Storage for embeddings
            embeddings = None
            
            def hook_fn(module, input):
                nonlocal embeddings
                # Only keep embeddings for visual tokens
                embeddings = input[0][:, text_positions, :].detach().clone()
                
            # Register hook on specified layer
            handle = model.model.layers[layer].register_forward_pre_hook(hook_fn)
            
            # Forward pass to get embeddings
            with torch.no_grad():
                model(**inputs)
                
            # Remove hook
            handle.remove()
            
            if embeddings is not None:
                # Mean pool across token positions to get a single embedding vector
                mean_embedding = embeddings.mean(dim=1, keepdim=True)  # Shape: (1, 1, D)
                all_embeddings.append(mean_embedding)
    
    if not all_embeddings:
        raise ValueError("No valid embeddings found for any of the input texts")
    
    # Concatenate all text embeddings
    return torch.cat(all_embeddings, dim=0)  # Shape: (num_texts, 1, D)

def manipulate_embeddings(
    embeddings1: torch.Tensor,
    embeddings2: torch.Tensor,
    start_idx: int,
    end_idx: int,
    operation: Literal["add", "subtract"] = "subtract",
    weight: float = 1.0,
    use_mean: bool = True,
) -> torch.Tensor:
    """Manipulate embeddings using projection operations.
    Only manipulates the image token region, preserving special tokens and text tokens.
    
    Args:
        embeddings1: First image embeddings [B, L, D]
        embeddings2: Second image embeddings [B, L, D]
        start_idx: Start index of image tokens
        end_idx: End index of image tokens
        operation: Whether to add or subtract projections
        weight: Weight for the projection operation
        use_mean: Whether to use mean pooling for projection
        
    Returns:
        Modified embeddings with same shape as embeddings1
    """
    # Only manipulate the image token region
    img_embeddings1 = embeddings1[:, start_idx:end_idx, :]
    img_embeddings2 = embeddings2[:, start_idx:end_idx, :]
    
    if operation == "subtract":
        modified_img_embeddings = subtract_projection(img_embeddings1, img_embeddings2, weight=weight, use_mean=use_mean)
    else:  # add
        modified_img_embeddings = subtract_projection(img_embeddings1, img_embeddings2, weight=-weight, use_mean=use_mean)
    
    # Create output tensor with original embeddings
    modified_embeddings = embeddings1.clone()
    # Replace only the image token region
    modified_embeddings[:, start_idx:end_idx, :] = modified_img_embeddings
    
    return modified_embeddings

def load_models(
    model_id: str,
    fast: bool = False,
    model_cache_dir: Optional[str] = None,
) -> Tuple[Union[ChameleonForConditionalGeneration, MultiModalityCausalLM], Union[ChameleonProcessor, VLChatProcessor]]:
    """Load the model and processor once to be reused."""
    logger.info("Loading models...")
    
    if "Janus" in model_id:
        # Load Janus model
        processor = VLChatProcessor.from_pretrained(
            model_id,
            token=os.environ.get("HF_TOKEN"),
            cache_dir=model_cache_dir,
            device_map="cuda:0",
        )
        
        if fast:
            model = MultiModalityCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                device_map="cuda:0",
                trust_remote_code=True,
                token=os.environ.get("HF_TOKEN"),
                cache_dir=model_cache_dir,
            )
            # Convert all model components to bfloat16
            model.vision_model = model.vision_model.to(dtype=torch.bfloat16)
            model.aligner = model.aligner.to(dtype=torch.bfloat16)
            model.gen_vision_model = model.gen_vision_model.to(dtype=torch.bfloat16)
            model.gen_aligner = model.gen_aligner.to(dtype=torch.bfloat16)
            model.gen_head = model.gen_head.to(dtype=torch.bfloat16)
            model.gen_embed = model.gen_embed.to(dtype=torch.bfloat16)
            model.language_model = model.language_model.to(dtype=torch.bfloat16)
            
            # Convert all nested modules to bfloat16
            def convert_to_bfloat16(module):
                for child in module.children():
                    if isinstance(child, torch.nn.Module):
                        child.to(dtype=torch.bfloat16)
                        convert_to_bfloat16(child)
                for param in module.parameters():
                    param.data = param.data.to(torch.bfloat16)
                for buffer in module.buffers():
                    buffer.data = buffer.data.to(torch.bfloat16)
            
            # Apply conversion recursively
            convert_to_bfloat16(model.vision_model)
            convert_to_bfloat16(model.aligner)
            convert_to_bfloat16(model.gen_vision_model)
            convert_to_bfloat16(model.gen_aligner)
            convert_to_bfloat16(model.gen_head)
            convert_to_bfloat16(model.language_model)
            
        else:
            model = MultiModalityCausalLM.from_pretrained(
                model_id,
                device_map="cuda:0",
                trust_remote_code=True,
                token=os.environ.get("HF_TOKEN"),
                cache_dir=model_cache_dir,
            )
            # Convert all model components to bfloat16
            model.vision_model = model.vision_model.to(dtype=torch.bfloat16)
            model.aligner = model.aligner.to(dtype=torch.bfloat16)
            model.gen_vision_model = model.gen_vision_model.to(dtype=torch.bfloat16)
            model.gen_aligner = model.gen_aligner.to(dtype=torch.bfloat16)
            model.gen_head = model.gen_head.to(dtype=torch.bfloat16)
            model.gen_embed = model.gen_embed.to(dtype=torch.bfloat16)
            model.language_model = model.language_model.to(dtype=torch.bfloat16)
            
            # Convert all nested modules to bfloat16
            def convert_to_bfloat16(module):
                for child in module.children():
                    if isinstance(child, torch.nn.Module):
                        child.to(dtype=torch.bfloat16)
                        convert_to_bfloat16(child)
                for param in module.parameters():
                    param.data = param.data.to(torch.bfloat16)
                for buffer in module.buffers():
                    buffer.data = buffer.data.to(torch.bfloat16)
            
            # Apply conversion recursively
            convert_to_bfloat16(model.vision_model)
            convert_to_bfloat16(model.aligner)
            convert_to_bfloat16(model.gen_vision_model)
            convert_to_bfloat16(model.gen_aligner)
            convert_to_bfloat16(model.gen_head)
            convert_to_bfloat16(model.language_model)
    else:
        # Load Chameleon model
        if fast:
            model = ChameleonForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                device_map="auto",
                token=os.environ.get("HF_TOKEN"),
                cache_dir=model_cache_dir,
            )
        else:
            model = ChameleonForConditionalGeneration.from_pretrained(
                model_id,
                device_map="auto",
                token=os.environ.get("HF_TOKEN"),
                cache_dir=model_cache_dir,
            )
        processor = ChameleonProcessor.from_pretrained(
            model_id,
            token=os.environ.get("HF_TOKEN"),
            cache_dir=model_cache_dir,
        )
        
    logger.info("Models loaded successfully")
    return model, processor

def run_latent_generation(
    model_type: Literal["chameleon", "janus", "Emu3"],
    model: Optional[Union[ChameleonForConditionalGeneration, MultiModalityCausalLM, AutoModelForCausalLM]] = None,
    processor: Optional[Union[ChameleonProcessor, VLChatProcessor, Emu3Processor]] = None,
    model_id: str = "leloy/Anole-7b-v0.1-hf",
    image_1_path: str = None,
    image_1: Image.Image = None,
    image_2_path: Optional[str] = None,
    image_2: Image.Image = None,
    text_input: Optional[Union[str, List[str]]] = None,
    prompt: Optional[str] = None,
    layer: int = 20,
    operation: Literal["add", "subtract", "normal"] = "subtract",
    weight: float = 0.2,
    use_mean: bool = True,
    max_new_tokens: int = 40,
    fast: bool = False,
    model_cache_dir: str = None,
    seed: Optional[int] = None,
    generation_kwargs: Optional[dict] = None,
) -> str:
    """Run image generation with latent embedding manipulation."""
    if seed is not None:
        set_seed(seed)

    # Load models if not provided
    if model is None or processor is None:
        model, processor = load_models(model_id, fast, model_cache_dir)

    logger.info("TASK: Latent Image Manipulation")

    # Load base image
    if image_1 is not None:
        image1 = image_1
    elif image_1_path is not None:
        image1 = load_image(image_1_path)
    else:
        raise ValueError("Must specify either image_1 or image_1_path")
    
    # Get projection embeddings (either from image2 or text)
    if image_2_path is not None and text_input is not None:
        raise ValueError("Cannot specify both image_2_path and text_input")
    
    # Get model's dtype and device
    model_dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    
    # Prepare inputs and get image token positions based on model type
    if model_type == "chameleon":
        if prompt is None:
            prompt = "Please describe this image in detail."
        prompt = f"{prompt}<image>"
        logger.info(f"Prompt: {prompt}")
        
        # Get image2 embeddings if provided
        if image_2_path is not None:
            logger.info(f"Using image projection from: {image_2_path}")
            image2 = load_image(image_2_path)
            proj_embeddings = get_image_embeddings(model_type, model, image2, processor, layer)
            del image2
        elif text_input is not None:
            logger.info(f"Using text projection from: {text_input}")
            proj_embeddings = get_text_embeddings(model_type, model, text_input, processor, layer)
        else:
            raise ValueError("Must specify either image_2_path or text_input")
        
        # Prepare inputs for generation
        inputs = processor(
            text=prompt,
            images=image1,
            return_tensors="pt"
        )
        
        # Find image token position
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
        image_token_pos = (inputs["input_ids"] == image_token_id).nonzero()[0, 1].item()
        start_idx = image_token_pos
        end_idx = start_idx + 1024  # Chameleon uses 1024 image tokens
        
        inputs = {
            "input_ids": inputs["input_ids"].to(device, dtype=torch.long),
            "attention_mask": inputs["attention_mask"].to(device, dtype=torch.long),
            "pixel_values": inputs["pixel_values"].to(device, dtype=model_dtype)
        }
        
    elif model_type == "janus":
        if prompt is None:
            prompt = "Please describe this image in detail."
        logger.info(f"Prompt: {prompt}")
        
        # Get image2 embeddings if provided
        if image_2_path is not None:
            logger.info(f"Using image projection from: {image_2_path}")
            image2 = load_image(image_2_path)
            proj_embeddings = get_image_embeddings(model_type, model, image2, processor, layer)
            del image2
        elif text_input is not None:
            logger.info(f"Using text projection from: {text_input}")
            proj_embeddings = get_text_embeddings(model_type, model, text_input, processor, layer)
        else:
            raise ValueError("Must specify either image_2_path or text_input")
        
        # Prepare conversation format for Janus
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{prompt}",
                "images": [image1],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        
        # Process inputs
        prepare_inputs = processor(
            conversations=conversation, 
            images=[image1], 
            force_batchify=True
        ).to(device)
        inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
        
        # Get image token positions
        start_id = processor.image_start_id
        end_id = processor.image_end_id
        start_idx = (prepare_inputs.input_ids == start_id).nonzero(as_tuple=True)[1].item()+1
        end_idx = (prepare_inputs.input_ids == end_id).nonzero(as_tuple=True)[1].item()
        
    elif model_type == "Emu3":
        if prompt is None:
            prompt = "Please describe this image in detail."
        logger.info(f"Prompt: {prompt}")
        
        # Get image2 embeddings if provided
        if image_2_path is not None:
            logger.info(f"Using image projection from: {image_2_path}")
            image2 = load_image(image_2_path)
            proj_embeddings = get_image_embeddings(model_type, model, image2, processor, layer)
            del image2
        elif text_input is not None:
            logger.info(f"Using text projection from: {text_input}")
            proj_embeddings = get_text_embeddings(model_type, model, text_input, processor, layer)
        else:
            raise ValueError("Must specify either image_2_path or text_input")
        
        # Process inputs for Emu3
        inputs, image_start_list, image_end_list = processor(
            text=prompt,
            image=image1,
            mode="U",  # Understanding mode for image input
            return_tensors="pt"
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get image token positions
        start_idx = image_start_list[0]  # First image start position
        end_idx = image_end_list[0]  # First image end position
        
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Register the hook to modify embeddings during forward pass
    def hook_fn(module, input):
        # Safety checks for input format
        if not isinstance(input, tuple) or len(input) == 0:
            logger.debug(f"Layer {layer} hook: input is not a tuple or is empty")
            return input
            
        hidden_states = input[0]
        
        if not isinstance(hidden_states, torch.Tensor):
            logger.debug(f"Layer {layer} hook: hidden_states is not a tensor")
            return input
            
        if hidden_states.dim() < 2:
            logger.debug(f"Layer {layer} hook: hidden_states dimension is < 2, shape: {hidden_states.shape}")
            return input
        
        # Check tensor shape to ensure it matches expectations
        if hidden_states.shape[1] <= max(start_idx, end_idx):
            logger.debug(f"Layer {layer} hook: hidden_states shape {hidden_states.shape} doesn't match indices {start_idx}:{end_idx}")
            return input
        
        try:
            # Get image region
            img_embeddings = hidden_states[:, start_idx:end_idx, :]
            
            # Apply projection operation sequentially for each embedding
            modified = img_embeddings.clone()
            for i in range(proj_embeddings.shape[0]):
                curr_embedding = proj_embeddings[i:i+1]  # Take one embedding at a time
                curr_embedding = curr_embedding.to(modified.device)
                
                # Apply projection operation
                if operation == "subtract":
                    modified = subtract_projection(modified, curr_embedding, weight=weight, use_mean=use_mean)
                elif operation == "add":
                    modified = subtract_projection(modified, curr_embedding, weight=-weight, use_mean=use_mean)
                else:  # "normal" - no modification
                    pass
            
            # Replace image region with modified embeddings
            hidden_states[:, start_idx:end_idx, :] = modified
            
            # Return modified hidden states
            return tuple([hidden_states] + list(input[1:]))
            
        except Exception as e:
            logger.warning(f"Error in hidden state hook at layer {layer}: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            return input
    
    # Register hook on the appropriate layer
    if model_type == "chameleon" or model_type == "Emu3":
        handle = model.model.layers[layer].register_forward_pre_hook(hook_fn)
    else:  # Janus model
        handle = model.language_model.model.layers[layer].register_forward_pre_hook(hook_fn)
        
    # Set up generation config
    generation_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
    }
    if generation_kwargs:
        generation_config.update(generation_kwargs)
    
    try:
        # Run generation with robust error handling
        with torch.inference_mode():
            if model_type == "chameleon":
                output_token_ids_batch = model.generate(
                    **inputs,
                    **generation_config
                )
                
                # Handle different return formats
                if isinstance(output_token_ids_batch, dict):
                    output_token_ids_batch = output_token_ids_batch.sequences
                    
                response_token_ids = [
                    output_token_ids[len(input_ids):]
                    for input_ids, output_token_ids in zip(
                        inputs["input_ids"], output_token_ids_batch
                    )
                ]
                response = processor.decode(response_token_ids[0], skip_special_tokens=True)
                
            elif model_type == "Emu3":
                outputs = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    **generation_config
                )
                
                # Handle different return formats
                if isinstance(outputs, dict):
                    outputs = outputs.sequences
                    
                # Skip input tokens to get only the generated response
                generated_text = outputs[:, inputs["input_ids"].shape[-1]:]
                response = processor.batch_decode(generated_text, skip_special_tokens=True)[0]
                
            else:  # Janus model
                outputs = model.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    pad_token_id=processor.tokenizer.eos_token_id,
                    bos_token_id=processor.tokenizer.bos_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    **generation_config
                )
                
                # Handle different return formats
                if isinstance(outputs, dict):
                    outputs = outputs.sequences
                    
                response = processor.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
                
    except Exception as e:
        logger.error(f"Error during generation: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        response = f"Error generating response: {str(e)}"
        
    finally:
        # Remove the hook
        try:
            handle.remove()
        except Exception as e:
            logger.warning(f"Error removing hook: {str(e)}")
        
        logger.info("Finished generation.")
    
    logger.info(f"Response: {response}")
    
    # Final cleanup
    try:
        del generation_config
        if model_type == "chameleon":
            if 'output_token_ids_batch' in locals():
                del output_token_ids_batch
            if 'response_token_ids' in locals():
                del response_token_ids
        else:
            if 'outputs' in locals():
                del outputs
        torch.cuda.empty_cache()
    except Exception as e:
        logger.warning(f"Error during final cleanup: {str(e)}")
    
    return response

def process_amber_batch(
    original_img_dir: str,
    new_img_dir: str,
    model_id: str = "leloy/Anole-7b-v0.1-hf",
    prompt: Optional[str] = None,
    layer: int = 20,
    operation: Literal["add", "subtract"] = "subtract",
    weight: float = 1.0,
    use_mean: bool = True,
    max_new_tokens: int = 40,
    fast: bool = False,
    model_cache_dir: Optional[str] = None,
    outputs_dir: str = ".",
    seed: Optional[int] = None,
) -> None:
    """Process amber image directories for batch latent generation between two image sets."""
    # Get list of images from both directories
    original_images = {
        int(f.split('_')[1].split('.')[0]): f 
        for f in os.listdir(original_img_dir) 
        if f.startswith('AMBER_') and f.endswith('.jpg')
    }
    new_images = {
        int(f.split('_')[1]): f 
        for f in os.listdir(new_img_dir) 
        if f.startswith('AMBER_') and f.endswith('_caption.jpg')
    }
    
    logger.info(f"Found {len(original_images)} images in original set")
    logger.info(f"Found {len(new_images)} images in new set")
    
    # Create outputs directory if it doesn't exist
    full_outputs_dir = os.path.abspath(outputs_dir)
    if not os.path.exists(full_outputs_dir):
        logging.info(f"Creating directory: {full_outputs_dir}")
        os.makedirs(full_outputs_dir)

    # Output JSON filename based on operation type
    output_file = os.path.join(full_outputs_dir, f"amber_out_latent_{operation}_image2image.json")
    
    # Load existing results if any
    existing_results = []
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            existing_results = json.load(f)
    existing_ids = {entry["id"] for entry in existing_results}

    # Load models once
    model, processor = load_models(model_id, fast, model_cache_dir)

    # Process entries that exist in both directories
    for image_id in sorted(set(original_images.keys()) & set(new_images.keys())):
        # Skip if already processed
        if image_id in existing_ids:
            logger.info(f"Skipping {image_id} as it already exists in output file")
            continue
            
        logger.info(f"Processing entry {image_id}")
        
        try:
            # Construct paths for both original and new images
            image_1_path = os.path.join(original_img_dir, original_images[image_id])
            image_2_path = os.path.join(new_img_dir, new_images[image_id])
            
            if not os.path.exists(image_1_path):
                logger.warning(f"Skipping {image_id} as original image not found: {image_1_path}")
                continue
                
            if not os.path.exists(image_2_path):
                logger.warning(f"Skipping {image_id} as new image not found: {image_2_path}")
                continue

            response = run_latent_generation(
                model=model,
                processor=processor,
                model_id=model_id,
                image_1_path=image_1_path,
                image_2_path=image_2_path,
                prompt=prompt,
                layer=layer,
                operation=operation,
                weight=weight,
                use_mean=use_mean,
                max_new_tokens=max_new_tokens,
                fast=fast,
                model_cache_dir=model_cache_dir,
                seed=seed,
            )
            
            # Save result
            existing_results.append({
                "id": image_id,
                "response": response
            })
            
            # Write results after each successful generation
            with open(output_file, 'w') as f:
                json.dump(existing_results, f, indent=2)
            logger.info(f"Updated results saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error processing entry {image_id}: {str(e)}")
            continue

def process_text_batch(
    img_dir: str,
    json_path: str,
    model_id: str = "leloy/Anole-7b-v0.1-hf",
    prompt: Optional[str] = None,
    layer: int = 20,
    operation: Literal["add", "subtract", "normal"] = "subtract",
    weight: float = 1.0,
    use_mean: bool = True,
    max_new_tokens: int = 40,
    fast: bool = False,
    model_cache_dir: Optional[str] = None,
    outputs_dir: str = ".",
    seed: Optional[int] = None,
) -> None:
    """Process text batch for batch latent generation between image and text."""
    original_images = {
        int(f.split('_')[1].split('.')[0]): f 
        for f in os.listdir(img_dir) 
        if f.startswith('AMBER_') and f.endswith('.jpg')
    }

    with open(json_path, 'r') as f:
        data = json.load(f)

    full_outputs_dir = os.path.abspath(outputs_dir)
    if not os.path.exists(full_outputs_dir):
        logging.info(f"Creating directory: {full_outputs_dir}")
        os.makedirs(full_outputs_dir)

    # Output JSON filename based on operation type
    output_file = os.path.join(full_outputs_dir, f"amber_out_latent_{operation}_imagetext.json")

    # Load existing results if any
    existing_results = []
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            existing_results = json.load(f)
    existing_ids = {entry["id"] for entry in existing_results}

    # Load models once
    model, processor = load_models(model_id, fast, model_cache_dir)

    for entry in data:
        image_id = entry["id"]
        if image_id not in original_images:
            logger.warning(f"Skipping {image_id} as original image not found: {image_id}")
            continue
        
        if image_id in existing_ids:
            logger.info(f"Skipping {image_id} as it already exists in output file")
            continue

        logger.info(f"Processing entry {image_id}")
        try:
            image_path = os.path.join(img_dir, original_images[image_id])
            if (operation == "subtract" and len(entry["hallucinated object"]) ==0) or (operation == "add" and len(entry["missed object"]) == 0):
                logger.info(f"Skipping {image_id} as no objects to manipulate")
                continue
            response = run_latent_generation(
                model=model,
                processor=processor,
                model_id=model_id,
                image_1_path=image_path,
                text_input=entry["missed object"] if operation == "add" else entry["hallucinated object"],
                prompt=prompt,
                layer=layer,
                operation=operation,
                weight=weight,
                use_mean=use_mean,
                max_new_tokens=max_new_tokens,
                fast=fast,
                model_cache_dir=model_cache_dir,
                seed=seed,
            )
            # Save result
            existing_results.append({
                "id": image_id,
                "response": response
            })
            
            # Write results after each successful generation
            with open(output_file, 'w') as f:
                json.dump(existing_results, f, indent=2)
            logger.info(f"Updated results saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error processing entry {image_id}: {str(e)}")
            continue

def run_latent_generation_with_gnn(
    model_type: Literal["chameleon", "janus", "Emu3"],
    model: Optional[Union[ChameleonForConditionalGeneration, MultiModalityCausalLM, AutoModelForCausalLM]] = None,
    processor: Optional[Union[ChameleonProcessor, VLChatProcessor, Emu3Processor]] = None,
    model_id: str = "leloy/Anole-7b-v0.1-hf",
    image_1_path: str = None,
    prompt: Optional[str] = None,
    layer: int = 10,
    operation: Literal["add", "subtract", "normal", "random_subtract"] = "subtract",
    weight: float = 1.0,
    use_mean: bool = True,
    max_new_tokens: int = 40,
    fast: bool = False,
    model_cache_dir: str = None,
    seed: Optional[int] = None,
    generation_kwargs: Optional[dict] = None,
    cluster_results_path: Optional[str] = None,
    find_non_input_indices: bool = False,
    num_layers: int = 2,
    layer_weights: List[float] = [1, 0.5],
    num_clusters: int = 2,
    RGBimage: Image.Image = None,
    use_opera: bool = False,
    opera_scale_factor: float = 50,
    opera_threshold: float = 15,
    opera_num_attn_candidates: int = 2,
    opera_penalty_weights: float = 1,
    use_vcd: bool = False,
    use_sid: bool = False,
    noise_step: int = 500,
    cd_alpha: float = 0.5,
    cd_beta: float = 0.1,
    text_input: Optional[Union[str, List[str]]] = None,
    text_edit_layer: int = 21,
    use_targets: bool = False,
    use_cls: bool = False,
) -> str:
    """Run image generation with latent embedding manipulation using GNN clustering."""
    if seed is not None:
        set_seed(seed)

    # Load models if not provided
    if model is None or processor is None:
        model, processor = load_models(model_id, fast, model_cache_dir)
    
    if not hasattr(model, 'apply_hook_flag'):
        model.apply_hook_flag = True  # Default to False
    if model_type == "janus":
        if not hasattr(model.language_model, 'apply_hook_flag'):
            model.language_model.apply_hook_flag = True

    logger.info("TASK: Latent Image Manipulation with GNN Clustering")

    # Load base image
    if RGBimage is None:   
        image1 = load_image(image_1_path)
        image1 = image1.convert("RGB")
    else:
        image1 = RGBimage
    
    
    # Get model's dtype and device
    model_dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    
    if model_type == "chameleon":
        if prompt is None:
            prompt = "Please describe this image in detail."
        prompt = f"{prompt}<image>"
        logger.info(f"Prompt: {prompt}")
        
        # Process image and get codebook usage
        dummy_prompt = "<image>"
        inputs = processor(text=dummy_prompt, images=image1, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device, dtype=model_dtype)
        
        # Get quantized representation
        with torch.no_grad():
            encoder_output = model.model.vqmodel.encoder(pixel_values)
            hidden_states = model.model.vqmodel.quant_conv(encoder_output)
            quant, _, indices = model.model.vqmodel.quantize(hidden_states)
        
        # Get indices and convert to numpy
        indices = indices.cpu().numpy().flatten()
        
        # Load pre-computed clustering results
        if cluster_results_path is None:
            raise ValueError("cluster_results_path must be provided")
        
        logger.info(f"Loading clustering results from {cluster_results_path}")
        results = load_clustering_results(cluster_results_path)
        
        # Get cluster labels from results (using K-means results by default)
        cluster_labels = results['kmeans']['labels']
        
        # Map indices to cluster labels
        image_cluster_labels = cluster_labels[indices]
        
        # Find the two most frequent clusters
        unique_clusters, counts = np.unique(image_cluster_labels, return_counts=True)
        top_clusters = unique_clusters[np.argsort(-counts)][:num_clusters]
        
        
        # Get codebook indices corresponding to these clusters
        top_cluster_indices = []
        
        if operation == "random_subtract":
            # For random_subtract, we'll randomly select indices from the codebook
            # First, determine how many indices we would have selected with the normal method
            for cluster in top_clusters:
                cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster]
                cluster_indices_set = set(cluster_indices)
                input_indices_set = set(indices)
                
                if find_non_input_indices:
                    filtered_indices = list(cluster_indices_set.difference(input_indices_set))
                    if filtered_indices:
                        top_cluster_indices.extend(filtered_indices[:4])
                else:
                    filtered_indices = list(cluster_indices_set.intersection(input_indices_set))
                    top_cluster_indices.extend(filtered_indices)
            
            # Get the count of indices we would have used
            num_indices = len(top_cluster_indices)
            
            # Now randomly select the same number of indices from the entire codebook
            codebook_size = len(cluster_labels)
            random_indices = random.sample(range(codebook_size), min(num_indices, codebook_size))
            top_cluster_indices = random_indices
            logger.info(f"Randomly selected {len(top_cluster_indices)} indices for manipulation")
        else:
            targets = []
            # Original method for other operations
            for cluster in top_clusters:
                cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster]
                cluster_indices_set = set(cluster_indices)
                input_indices_set = set(indices)
                
                if find_non_input_indices:
                    # Get indices that belong to this cluster but do NOT exist in the input image
                    filtered_indices = list(cluster_indices_set.difference(input_indices_set))
                    #if filtered_indices:  # Only add if there are indices not in input
                        # Limit the number of indices to avoid too many manipulations
                        #top_cluster_indices.extend(filtered_indices[:10])  # Take up to 5 indices per cluster
                    top_cluster_indices.extend(filtered_indices)
                else:
                    # Get all indices that belong to this cluster and exist in the input image
                    filtered_indices = list(cluster_indices_set.intersection(input_indices_set))
                    top_cluster_indices.extend(filtered_indices)  # Add all intersecting indices
            positions = [pos for pos, val in enumerate(indices.tolist()) if val in cluster_indices]
            targets.extend(positions)

        targets = sorted(list(set(targets)))
        if not use_targets:
            targets = None
        
        # Handle empty top_cluster_indices
        if not top_cluster_indices:
            logger.warning("No valid cluster indices found. Running normal generation without manipulation.")
            operation = "normal"  # Set operation to normal to skip manipulation
            # Use a dummy embedding that won't affect the output when operation is "normal"
            text_embeddings = torch.zeros((1, 1, model.config.hidden_size), device=device, dtype=model_dtype)
            
        else:
            logger.info(f"Top cluster indices: {top_cluster_indices}")
            # Get embeddings for these indices using model-specific method
            text_inputs = [processor.tokenizer.decode([idx]) for idx in top_cluster_indices]
            logger.info(f"Using top cluster tokens as text inputs: {text_inputs}")
            text_embeddings = get_text_embeddings(model_type, model, text_inputs, processor, layer)
            
        
        # Prepare inputs for generation
        inputs = processor(
            text=prompt,
            images=image1,
            return_tensors="pt"
        )
        inputs = {
            "input_ids": inputs["input_ids"].to(device, dtype=torch.long),
            "attention_mask": inputs["attention_mask"].to(device, dtype=torch.long),
            "pixel_values": inputs["pixel_values"].to(device, dtype=model_dtype)
        }
        
        # Get image token position
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
        image_token_pos = (inputs["input_ids"] == image_token_id).nonzero()[0, 1].item()
        start_idx = image_token_pos
        end_idx = start_idx + 1024 +1  # Chameleon uses 1024 image tokens
        
        # Set up key_position for OPERA generation
        key_position = {
            "image_start": torch.tensor(start_idx).to(device),
            "image_end": torch.tensor(end_idx).to(device),
            "response_start": torch.tensor(inputs["input_ids"].shape[-1]).to(device),
        }
        
    elif model_type == "janus":
        if prompt is None:
            prompt = "Please describe this image in detail."
        logger.info(f"Prompt: {prompt}")
        
        # Process image and get codebook usage
        conversation = [
            {
                "role": "<|User|>",
                "content": "<image_placeholder>",
                "images": [image1],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        prepare_inputs = processor(
            conversations=conversation, 
            images=[image1], 
            force_batchify=True
        ).to(device)
        inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
        pixel_values = prepare_inputs.pixel_values.squeeze(0).to(device, dtype=model_dtype)
        
        # Get quantized representation from Janus vision model
        with torch.no_grad():
            vqmodel = model.gen_vision_model
            encoder_output = vqmodel.encoder(pixel_values)
            hidden_states = vqmodel.quant_conv(encoder_output)
            _, _, info = vqmodel.quantize(hidden_states)
            indices = info[2]
        
        # Get indices and convert to numpy
        indices = indices.cpu().numpy().flatten()
        
        # Load pre-computed clustering results
        if cluster_results_path is None:
            raise ValueError("cluster_results_path must be provided")
        
        logger.info(f"Loading clustering results from {cluster_results_path}")
        results = load_clustering_results(cluster_results_path)
        
        
        # Get cluster labels from results (using K-means results by default)
        cluster_labels = results['kmeans']['labels']

        
        # Map indices to cluster labels
        image_cluster_labels = cluster_labels[indices]
        
        # Find the two most frequent clusters
        unique_clusters, counts = np.unique(image_cluster_labels, return_counts=True)
        top_clusters = unique_clusters[np.argsort(-counts)][:num_clusters]
        
        # Get codebook indices corresponding to these clusters
        top_cluster_indices = []
        
        if operation == "random_subtract":
            # For random_subtract, we'll randomly select indices from the codebook
            # First, determine how many indices we would have selected with the normal method
            for cluster in top_clusters:
                cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster]
                cluster_indices_set = set(cluster_indices)
                input_indices_set = set(indices)
                
                if find_non_input_indices:
                    filtered_indices = list(cluster_indices_set.difference(input_indices_set))
                    if filtered_indices:
                        top_cluster_indices.extend(filtered_indices[:4])
                else:
                    filtered_indices = list(cluster_indices_set.intersection(input_indices_set))
                    top_cluster_indices.extend(filtered_indices)
            
            # Get the count of indices we would have used
            num_indices = len(top_cluster_indices)
            
            # Now randomly select the same number of indices from the entire codebook
            codebook_size = len(cluster_labels)
            random_indices = random.sample(range(codebook_size), min(num_indices, codebook_size))
            top_cluster_indices = random_indices
            logger.info(f"Randomly selected {len(top_cluster_indices)} indices for manipulation")
        else:
            # Original method for other operations
            targets = []
            for cluster in top_clusters:
                cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster]
                cluster_indices_set = set(cluster_indices)
                input_indices_set = set(indices)
                
                if find_non_input_indices:
                    # Get indices that belong to this cluster but do NOT exist in the input image
                    filtered_indices = list(cluster_indices_set.difference(input_indices_set))
                    #if filtered_indices:  # Only add if there are indices not in input
                        # Limit the number of indices to avoid too many manipulations
                        #top_cluster_indices.extend(filtered_indices[:4])  # Take up to 5 indices per cluster
                    top_cluster_indices.extend(filtered_indices)
                else:
                    # Get all indices that belong to this cluster and exist in the input image
                    filtered_indices = list(cluster_indices_set.intersection(input_indices_set))
                    top_cluster_indices.extend(filtered_indices)  # Add all intersecting indices
                
                positions = [pos for pos, val in enumerate(indices.tolist()) if val in cluster_indices]
                targets.extend(positions)

            targets = sorted(list(set(targets)))
            if not use_targets:
                targets = None

        
        # Handle empty top_cluster_indices
        if not top_cluster_indices:
            logger.warning("No valid cluster indices found. Running normal generation without manipulation.")
            operation = "normal"  # Set operation to normal to skip manipulation
            # Use a dummy embedding that won't affect the output when operation is "normal"
            text_embeddings = torch.zeros((1, model.config.gen_aligner_config.params.n_embed), device=device, dtype=model_dtype)
        else:
            
            # convert top_cluster_indices of code book to token ids
            logger.info(f"Using top cluster tokens as text inputs: {top_cluster_indices}")
            text_embeddings = get_text_embeddings(model_type, model, top_cluster_indices, processor, layer)
        
        # Prepare inputs for generation
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{prompt}",
                "images": [image1],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        prepare_inputs = processor(
            conversations=conversation, 
            images=[image1], 
            force_batchify=True
        ).to(device)
        inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
        
        # Get image token positions
        
        start_id = processor.image_start_id
        end_id = processor.image_end_id
        
        start_idx = (prepare_inputs.input_ids == start_id).nonzero()[0,1].item()+1
        end_idx = (prepare_inputs.input_ids == end_id).nonzero()[0,1].item()+1
        
        # Set up key_position for OPERA generation
        key_position = {
            "image_start": torch.tensor(start_idx).to(device),
            "image_end": torch.tensor(end_idx).to(device),
            "response_start": torch.tensor(inputs_embeds.shape[1]).to(device),
        }
        
    elif model_type == "Emu3":
        if prompt is None:
            prompt = "Please describe this image in detail."
        logger.info(f"Prompt: {prompt}")
        
        # release cuda memory
        torch.cuda.empty_cache()
        
        # Process image and get codebook usage
        image1 = load_image(image_1_path)
        pixel_values = processor.image_processor([image1], return_tensors="pt")["pixel_values"]
        # Move to same device and dtype as vision tokenizer
        pixel_values = pixel_values.to(
            device=processor.vision_tokenizer.device,
            dtype=processor.vision_tokenizer.dtype
        )
        
        # Get quantized representation from Emu3 vision model
        ndim = pixel_values.ndim
        if ndim == 4:
            t = processor.vision_tokenizer.config.temporal_downsample_factor
            b, c, h, w = pixel_values.shape
            pixel_values = pixel_values.unsqueeze(1).repeat(1, t, 1, 1, 1)
        elif ndim == 5:
            b, t, c, h, w = pixel_values.shape
            
        with torch.no_grad():
            encoder_output = processor.vision_tokenizer.encoder(pixel_values)
            # b t c h w -> b c t h w
            encoder_output = encoder_output.permute(0, 2, 1, 3, 4)
            hidden_state = processor.vision_tokenizer.quant_conv(encoder_output)
            # b c t h w -> b t c h w
            hidden_state = hidden_state.permute(0, 2, 1, 3, 4)
            indices = processor.vision_tokenizer.quantize(hidden_state)
        if ndim == 4:
            indices = indices.squeeze(1)
        
        indices = indices.flatten().cpu().numpy()
        
        # Load pre-computed clustering results
        if cluster_results_path is None:
            raise ValueError("cluster_results_path must be provided")
        
        logger.info(f"Loading clustering results from {cluster_results_path}")
        results = load_clustering_results(cluster_results_path)
        
        # Get cluster labels from results (using K-means results by default)
        cluster_labels = results['kmeans']['labels']
        
        # Map indices to cluster labels
        image_cluster_labels = cluster_labels[indices]
        
        # Find the two most frequent clusters
        unique_clusters, counts = np.unique(image_cluster_labels, return_counts=True)
        top_clusters = unique_clusters[np.argsort(-counts)][:num_clusters]
        
        # Get codebook indices corresponding to these clusters
        top_cluster_indices = []
        
        if operation == "random_subtract":
            # For random_subtract, we'll randomly select indices from the codebook
            # First, determine how many indices we would have selected with the normal method
            for cluster in top_clusters:
                cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster]
                cluster_indices_set = set(cluster_indices)
                input_indices_set = set(indices)
                
                if find_non_input_indices:
                    filtered_indices = list(cluster_indices_set.difference(input_indices_set))
                    if filtered_indices:
                        top_cluster_indices.extend(filtered_indices[:4])
                else:
                    filtered_indices = list(cluster_indices_set.intersection(input_indices_set))
                    top_cluster_indices.extend(filtered_indices)
            
            # Get the count of indices we would have used
            num_indices = len(top_cluster_indices)
            
            # Now randomly select the same number of indices from the entire codebook
            codebook_size = len(cluster_labels)
            random_indices = random.sample(range(codebook_size), min(num_indices, codebook_size))
            top_cluster_indices = random_indices
            logger.info(f"Randomly selected {len(top_cluster_indices)} indices for manipulation")
        else:
            # Original method for other operations
            for cluster in top_clusters:
                cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster]
                cluster_indices_set = set(cluster_indices)
                input_indices_set = set(indices)
                
                if find_non_input_indices:
                    # Get indices that belong to this cluster but do NOT exist in the input image
                    filtered_indices = list(cluster_indices_set.difference(input_indices_set))
                    if filtered_indices:  # Only add if there are indices not in input
                        # Limit the number of indices to avoid too many manipulations
                        #top_cluster_indices.extend(filtered_indices[:10])  # Take up to 10 indices per cluster
                        top_cluster_indices.extend(filtered_indices)
                else:
                    # Get all indices that belong to this cluster and exist in the input image
                    filtered_indices = list(cluster_indices_set.intersection(input_indices_set))
                    top_cluster_indices.extend(filtered_indices)  # Add all intersecting indices
        
        # Handle empty top_cluster_indices
        if not top_cluster_indices:
            logger.warning("No valid cluster indices found. Running normal generation without manipulation.")
            operation = "normal"  # Set operation to normal to skip manipulation
            # Use a dummy embedding that won't affect the output when operation is "normal"
            text_embeddings = torch.zeros((1, 1, model.config.hidden_size), device=device, dtype=model_dtype)
        else:
            logger.info(f"Using top cluster tokens as text inputs: {top_cluster_indices}")
            text_embeddings = get_text_embeddings(model_type, model, top_cluster_indices, processor, layer)
        
        # Prepare inputs for generation
        inputs, image_start_list, image_end_list = processor(
            text=prompt,
            image=image1,
            mode="U",  # Understanding mode for image input
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get image token positions
        start_idx = image_start_list[0]  # First image start position
        end_idx = image_end_list[0]  # First image end position
        
        # Set up key_position for OPERA generation
        key_position = {
            "image_start": torch.tensor(start_idx).to(device),
            "image_end": torch.tensor(end_idx).to(device),
            "response_start": torch.tensor(inputs["input_ids"].shape[1]).to(device),
        }
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    if text_input is not None:
        proj_text_embeddings = get_text_embeddings(model_type, model, text_input, processor, text_edit_layer)
        def create_text_edit_hook(layer_idx, layer_weight):
            def hook_fn(module, input):
                
                # Safety checks with more detailed diagnostics
                if not isinstance(input, tuple) or len(input) == 0:
                    logger.debug(f"Layer {layer_idx} hook: input is not a tuple or is empty")
                    return input
                    
                hidden_states = input[0]
                
                if not isinstance(hidden_states, torch.Tensor):
                    logger.debug(f"Layer {layer_idx} hook: hidden_states is not a tensor")
                    return input
                    
                if hidden_states.dim() < 2:
                    logger.debug(f"Layer {layer_idx} hook: hidden_states dimension is < 2, shape: {hidden_states.shape}")
                    return input
                
                # Check tensor shape to ensure it matches expectations
                if hidden_states.shape[1] <= max(start_idx, end_idx):
                    logger.debug(f"Layer {layer_idx} hook: hidden_states shape {hidden_states.shape} doesn't match indices {start_idx}:{end_idx}")
                    return input
                if model.apply_hook_flag:
                    try:
                        # Log the shape for debugging
                        logger.debug(f"Layer {layer_idx} hook: Processing hidden_states with shape {hidden_states.shape}")
                        
                        img_embeddings = hidden_states[:, start_idx:end_idx, :]
                        
                        modified = img_embeddings.clone()
                        for i in range(proj_text_embeddings.shape[0]):
                            curr_embedding = proj_text_embeddings[i:i+1]
                            curr_embedding = curr_embedding.to(modified.device)
                            
                            
                            # Higher weight for object removal at early layers
                            modified = subtract_projection(modified, curr_embedding, 
                                                        weight=weight * layer_weight, 
                                                        use_mean=use_mean)
                            
                        
                        hidden_states[:, start_idx:end_idx, :] = modified
                        logger.info(f"Modified hidden states shape: {hidden_states.shape}")
                        return tuple([hidden_states] + list(input[1:]))
                    except Exception as e:
                        logger.warning(f"Error in hidden state hook at layer {layer_idx}: {str(e)}")
                        import traceback
                        logger.debug(traceback.format_exc())
                        return input
                else:
                    return input
                
            return hook_fn

    # Create the primary hidden state modification hook with more robust implementation
    def create_hidden_state_hook(layer_idx, layer_weight):
        def hook_fn(module, input):
            
            # Safety checks with more detailed diagnostics
            if not isinstance(input, tuple) or len(input) == 0:
                logger.debug(f"Layer {layer_idx} hook: input is not a tuple or is empty")
                return input
                
            hidden_states = input[0]
            
            if not isinstance(hidden_states, torch.Tensor):
                logger.debug(f"Layer {layer_idx} hook: hidden_states is not a tensor")
                return input
                
            if hidden_states.dim() < 2:
                logger.debug(f"Layer {layer_idx} hook: hidden_states dimension is < 2, shape: {hidden_states.shape}")
                return input
            
            # Check tensor shape to ensure it matches expectations
            if hidden_states.shape[1] <= max(start_idx, end_idx):
                logger.debug(f"Layer {layer_idx} hook: hidden_states shape {hidden_states.shape} doesn't match indices {start_idx}:{end_idx}")
                return input
            if model.apply_hook_flag:
                try:
                    # Log the shape for debugging
                    logger.debug(f"Layer {layer_idx} hook: Processing hidden_states with shape {hidden_states.shape}")
                    
                    img_embeddings = hidden_states[:, start_idx:end_idx, :]
                    
                    modified = img_embeddings.clone()
                    for i in range(text_embeddings.shape[0]):
                        curr_embedding = text_embeddings[i:i+1]
                        curr_embedding = curr_embedding.to(modified.device)
                        
                        # Add normalization for more focus on direction rather than magnitude
                        if operation == "subtract" or operation == "random_subtract":
                            # Higher weight for object removal at early layers
                            modified = subtract_projection(modified, curr_embedding, 
                                                        weight=weight * layer_weight, 
                                                        use_mean=use_mean, targets=targets, use_cls=use_cls )
                        elif operation == "add":
                            modified = subtract_projection(modified, curr_embedding, 
                                                        weight=-weight * layer_weight, 
                                                        use_mean=use_mean)
                        else:
                            modified = modified
                    
                    hidden_states[:, start_idx:end_idx, :] = modified
                    logger.info(f"Modified hidden states shape: {hidden_states.shape}")
                    return tuple([hidden_states] + list(input[1:]))
                except Exception as e:
                    logger.warning(f"Error in hidden state hook at layer {layer_idx}: {str(e)}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    return input
            else:
                return input
            
        return hook_fn
    
    # Register hooks as before, but with better error handling
    num_layers = num_layers
    layer_weights = layer_weights
    handles = []
    
    # Register hooks with proper error handling
    for i in range(num_layers):
        current_layer = layer + i
        current_layer_text = text_edit_layer + i
        
        try:
            if model_type == "chameleon" or model_type == "Emu3":
                if current_layer < len(model.model.layers):
                    handle = model.model.layers[current_layer].register_forward_pre_hook(
                        create_hidden_state_hook(current_layer, layer_weights[i])
                    )
                    handles.append(handle)
            else:  # Janus model
                if current_layer < len(model.language_model.model.layers):
                    handle = model.language_model.model.layers[current_layer].register_forward_pre_hook(
                        create_hidden_state_hook(current_layer, layer_weights[i])
                    )
                    handles.append(handle)
            logger.info(f"Registered hook for layer {current_layer} for {model_type} model")
        except Exception as e:
            logger.error(f"Failed to register hook for layer {current_layer}: {str(e)}")
            raise e
        if text_input is not None:
            try:
                if model_type == "chameleon" or model_type == "Emu3":
                    if current_layer_text < len(model.model.layers):
                        handle = model.model.layers[current_layer_text].register_forward_pre_hook(
                            create_text_edit_hook(current_layer_text, layer_weights[i])
                        )
                        handles.append(handle)
                else:  # Janus model
                    if current_layer_text < len(model.language_model.model.layers):
                        handle = model.language_model.model.layers[current_layer_text].register_forward_pre_hook(
                            create_text_edit_hook(current_layer_text, layer_weights[i])
                        )
                        handles.append(handle)
                logger.info(f"Registered hook for layer {current_layer_text} for {model_type} model")
            except Exception as e:
                logger.error(f"Failed to register hook for layer {current_layer_text}: {str(e)}")
                raise e
    # Setup for VCD generation
    try:
        # Import VCD utilities if needed
        if use_vcd or use_sid:
            from vcd_utils.vcd_add_noise import add_diffusion_noise
            
            # Process image and get image tensor for adding noise
            if model_type == "chameleon":
                # Process with dummy prompt since processor requires text
                dummy_inputs = processor(text="<image>", images=image1, return_tensors="pt")
                image_tensor = dummy_inputs["pixel_values"].to(device, dtype=model_dtype)
                # Create noisy image for contrastive decoding
                image_cd = add_diffusion_noise(image_tensor, noise_step)
                
            elif model_type == "Emu3":
                image_tensor = processor.image_processor([image1], return_tensors="pt")["pixel_values"]
                image_tensor = image_tensor.to(device, dtype=model_dtype)
                # Create noisy image for contrastive decoding
                image_cd = add_diffusion_noise(image_tensor, noise_step)
                
            elif model_type == "janus":
                conversation = [
                    {
                        "role": "<|User|>",
                        "content": f"<image_placeholder>\n{prompt}",
                        "images": [image1],
                    },
                    {"role": "<|Assistant|>", "content": ""},
                ]
                prepare_inputs = processor(
                    conversations=conversation,
                    images=[image1],
                    force_batchify=True
                ).to(model.device)
                image_tensor = prepare_inputs["pixel_values"].to(device, dtype=model_dtype)
                # Create noisy image for contrastive decoding
                image_cd = add_diffusion_noise(image_tensor, noise_step)
        
        # Setup SID configuration if needed
        if use_sid:
            if model_type == "chameleon":
                model.model.config.use_fast_v = True
                model.model.config.fast_v_inplace = False
                model.model.config.fast_v_attention_rank = 100
                model.model.config.fast_v_attention_rank_add = 100
                model.model.config.fast_v_agg_layer = 2
                model.model.reset_fastv()
            elif model_type == "janus":
                model.language_model.model.config.use_fast_v = True
                model.language_model.model.config.fast_v_inplace = False
                model.language_model.model.config.fast_v_attention_rank = 100
                model.language_model.model.config.fast_v_attention_rank_add = 100
                model.language_model.model.config.fast_v_agg_layer = 2
                model.language_model.model.reset_fastv()
            elif model_type == "Emu3":
                model.model.config.use_fast_v = True
                model.model.config.fast_v_inplace = False
                model.model.config.fast_v_attention_rank = 100
                model.model.config.fast_v_attention_rank_add = 100
                model.model.config.fast_v_agg_layer = 2
                model.model.reset_fastv()
                
        # Create generation config
        if use_opera:
            # OPERA generation config
            generation_config = {
                "max_length": 600,
                "output_attentions": True,
                "num_beams": 3,
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "opera_decoding": True,
                "key_position": key_position,
                "scale_factor": opera_scale_factor,
                "threshold": opera_threshold,
                "num_attn_candidates": opera_num_attn_candidates,
                "penalty_weights": opera_penalty_weights,
                "return_dict_in_generate": True,
                "pad_token_id": model.config.pad_token_id if hasattr(model, 'config') and hasattr(model.config, 'pad_token_id') and model.config.pad_token_id is not None else processor.tokenizer.pad_token_id,
                "eos_token_id": model.config.eos_token_id if hasattr(model, 'config') and hasattr(model.config, 'eos_token_id') and model.config.eos_token_id is not None else processor.tokenizer.eos_token_id
            }
            
            if model_type == "Emu3":
                generation_config.update({
                    "pad_token_id": processor.tokenizer.pad_token_id,
                    "bos_token_id": processor.tokenizer.bos_token_id,
                    "eos_token_id": processor.tokenizer.eos_token_id,
                })
            
            
        elif use_vcd or use_sid:
            # VCD or SID generation config
            generation_config = {
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "vcd_sample": True,
                "cd_alpha": cd_alpha,
                "cd_beta": cd_beta,
                "vcd_inputs": image_cd,
                "output_attentions": False,
                "output_hidden_states": False,
                "return_dict_in_generate": True,
                "key_position": key_position,
                "pad_token_id": model.config.pad_token_id if hasattr(model, 'config') and hasattr(model.config, 'pad_token_id') and model.config.pad_token_id is not None else processor.tokenizer.pad_token_id,
                "eos_token_id": model.config.eos_token_id if hasattr(model, 'config') and hasattr(model.config, 'eos_token_id') and model.config.eos_token_id is not None else processor.tokenizer.eos_token_id
            }
            
            if use_sid:
                generation_config["use_sid"] = True
            
            if model_type == "Emu3":
                generation_config.update({
                    "pad_token_id": processor.tokenizer.pad_token_id,
                    "bos_token_id": processor.tokenizer.bos_token_id,
                    "eos_token_id": processor.tokenizer.eos_token_id,
                })
        else:
            # Standard generation config
            generation_config = {
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
            }
            if generation_kwargs:
                generation_config.update(generation_kwargs)
                
        generation_config = GenerationConfig(**generation_config)

        # Handle VCD-specific preparation for models
        if (use_vcd or use_sid) and model_type == "janus":
            # Convert tensor to PIL image
            
            image_cd_pil = tensor_to_pil(image_cd.squeeze(0))
            
            conversation_cd = [
                {
                    "role": "<|User|>",
                    "content": f"<image_placeholder>\n{prompt}",
                    "images": [image_cd_pil],
                },
                {"role": "<|Assistant|>", "content": ""},
            ]
            prepare_inputs_cd = processor(
                conversations=conversation_cd,
                images=[image_cd_pil],
                force_batchify=True
            ).to(model.device)
            inputs_cd = model.prepare_inputs_embeds(**prepare_inputs_cd)
            
            generation_config.vcd_inputs = inputs_cd.detach().clone()
            if use_sid:
                generation_config.vcd_inputs = inputs_embeds.detach().clone()
            
        elif (use_vcd or use_sid) and model_type == "Emu3":
                
            image_cd_pil = tensor_to_pil(image_cd)
            
            inputs_cd, _, _ = processor(
                text=prompt,
                image=image_cd_pil,
                mode="U",
                return_tensors="pt",
                padding="longest",
            )
            
            inputs_cd = {k: v.to(device) for k, v in inputs_cd.items()}
            
            generation_config.vcd_inputs = inputs_cd["input_ids"]
            if use_sid:
                generation_config.vcd_inputs = inputs["input_ids"]

        # Run the generation with robust error handling        
        with torch.inference_mode():
            if model_type == "chameleon":
                if use_vcd:
                    image_cd = tensor_to_pil(generation_config.vcd_inputs.squeeze(0))
                    inputs_cd = processor(text=prompt + "<image>", images=image_cd, return_tensors="pt")
                    inputs_cd_ids = inputs_cd.input_ids.to(device, dtype=torch.long)
                    generation_config.vcd_inputs = inputs_cd_ids
                elif use_sid:
                    generation_config.vcd_inputs = inputs["input_ids"]
                output_token_ids_batch = model.generate(
                    **inputs,
                    generation_config=generation_config if isinstance(generation_config, GenerationConfig) else None,
                    **({} if isinstance(generation_config, GenerationConfig) else generation_config)
                )
                
                if isinstance(output_token_ids_batch, dict):
                    output_token_ids_batch = output_token_ids_batch.sequences
                    
                response_token_ids = [
                    output_token_ids[len(input_ids):]
                    for input_ids, output_token_ids in zip(
                        inputs["input_ids"], output_token_ids_batch
                    )
                ]
                response = processor.decode(response_token_ids[0], skip_special_tokens=True)
            elif model_type == "Emu3":
                outputs = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    generation_config=generation_config if isinstance(generation_config, GenerationConfig) else None,
                    **({} if isinstance(generation_config, GenerationConfig) else generation_config)
                )
                    
                if isinstance(outputs, dict):
                    generated_sequence = outputs.sequences
                else:
                    generated_sequence = outputs
                outputs = generated_sequence[:, inputs["input_ids"].shape[-1]:]
                response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
            else:  # Janus model
                outputs = model.language_model.generate(
                    inputs_embeds=inputs_embeds.detach().clone(),
                    attention_mask=prepare_inputs.attention_mask.detach().clone(),
                    generation_config=generation_config if isinstance(generation_config, GenerationConfig) else None,
                    pad_token_id=processor.tokenizer.eos_token_id,
                    bos_token_id=processor.tokenizer.bos_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    **({} if isinstance(generation_config, GenerationConfig) else generation_config)
                )
                    
                if isinstance(outputs, dict):
                    outputs = outputs.sequences
                    
                response = processor.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
                
    except Exception as e:
        logger.error(f"Error during generation: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        response = f"Error generating response: {str(e)}"
        
    finally:
        # Clean up hooks and reset model settings
        for handle in handles:
            try:
                handle.remove()
            except Exception as e:
                logger.warning(f"Error removing hook: {str(e)}")
                
        # Reset fast_v settings if needed
        if use_sid:
            try:
                if model_type == "chameleon":
                    model.model.config.use_fast_v = False
                    model.model.reset_fastv()
                elif model_type == "janus":
                    model.language_model.model.config.use_fast_v = False
                    model.language_model.model.reset_fastv()
                elif model_type == "Emu3":
                    model.model.config.use_fast_v = False
                    model.model.reset_fastv()
            except Exception as e:
                logger.warning(f"Error resetting fast_v: {str(e)}")
        
        logger.info("Finished cleanup")
    
    logger.info(f"Generated response: {response}")
    
    # Final cleanup
    try:
        del generation_config
        if model_type == "chameleon":
            if 'output_token_ids_batch' in locals():
                del output_token_ids_batch
            if 'response_token_ids' in locals():
                del response_token_ids
        else:
            if 'outputs' in locals():
                del outputs
        torch.cuda.empty_cache()
    except Exception as e:
        logger.warning(f"Error during final cleanup: {str(e)}")
    
    return response

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate text with latent image/text embedding manipulation."
    )
    parser.add_argument(
        "-m",
        "--model_id",
        type=str,
        required=False,
        default="leloy/Anole-7b-v0.1-hf",
        help="The model ID to use for generation.",
    )
    parser.add_argument(
        "-i1",
        "--image_1_path",
        type=str,
        required=False,
        default=None,
        help="The path to the first (base) image.",
    )
    parser.add_argument(
        "-i2",
        "--image_2_path", 
        type=str,
        required=False,
        default=None,
        help="The path to the second image to mix embeddings from.",
    )
    parser.add_argument(
        "-t",
        "--text_input",
        type=str,
        nargs="+",  # Accept multiple text inputs
        required=False,
        default=None,
        help="Text or list of texts to use for projection instead of image_2",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        required=False,
        default=None,
        help="The prompt for generation.",
    )
    parser.add_argument(
        "-l",
        "--layer",
        type=int,
        required=False,
        default=15,
        help="Which layer to manipulate embeddings at.",
    )
    parser.add_argument(
        "--operation",
        type=str,
        choices=["add", "subtract", "normal", "random_subtract"],
        required=False,
        default="subtract",
        help="Whether to add or subtract projections, or use random indices for subtraction",
    )
    parser.add_argument(
        "-w",
        "--weight",
        type=float,
        required=False,
        default=2.0,
        help="Weight for the projection operation.",
    )
    parser.add_argument(
        "--use_mean",
        type=bool,
        required=False,
        default=True,
        help="Whether to use mean pooling for projection",
    )
    parser.add_argument(
        "-n",
        "--max_new_tokens",
        type=int,
        required=False,
        default=256,
        help="The maximum number of tokens to generate.",
    )
    parser.add_argument(
        "-f",
        "--fast",
        type=int,
        required=False,
        default=False,
        help="Whether to convert the model to bfloat16 & use Flash Attention 2",
    )
    parser.add_argument(
        "-c",
        "--model_cache_dir",
        type=str,
        required=False,
        default=None,
        help="The directory to cache the model in.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        required=False,
        default=42,
        help="The seed to use for generation.",
    )
    parser.add_argument(
        "--original_img_dir",
        type=str,
        required=False,
        help="Directory containing original AMBER images (AMBER_X.jpg)",
    )
    parser.add_argument(
        "--new_img_dir",
        type=str,
        required=False,
        help="Directory containing new AMBER images (AMBER_X_caption.jpg)",
    )
    parser.add_argument(
        "--json_path",
        type=str,
        required=False,
        help="Path to JSON file containing text data for manipulation",
    )
    parser.add_argument(
        "--gnn_model_path",
        type=str,
        required=False,
        help="Path to trained GNN model and embeddings for GNN-based generation",
    )
    parser.add_argument(
        "--use_gnn",
        action="store_true",
        help="Whether to use GNN-based generation method",
    )
    parser.add_argument(
        "--cluster_results_path",
        type=str,
        required=False,
        help="Path to pre-computed clustering results pickle file",
    )
    parser.add_argument(
        "--find_non_input_indices",
        action="store_true",
        help="Whether to find indices that don't appear in the input image",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=False,
        default="chameleon",
        help="The type of model to use for generation.",
    )
    args: argparse.Namespace = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    logger.info(f"Running with args: {args}")
    
    if args.original_img_dir and args.new_img_dir:
        process_amber_batch(
            original_img_dir=args.original_img_dir,
            new_img_dir=args.new_img_dir,
            model_id=args.model_id,
            prompt=args.prompt,
            layer=args.layer,
            operation=args.operation,
            weight=args.weight,
            use_mean=args.use_mean,
            max_new_tokens=args.max_new_tokens,
            fast=args.fast,
            model_cache_dir=args.model_cache_dir,
            seed=args.seed,
        )
    elif args.original_img_dir and args.json_path:
        process_text_batch(
            img_dir=args.original_img_dir,
            json_path=args.json_path,
            model_id=args.model_id,
            prompt=args.prompt,
            layer=args.layer,
            operation=args.operation,
            weight=args.weight,
            use_mean=args.use_mean,
            max_new_tokens=args.max_new_tokens,
            fast=args.fast,
            model_cache_dir=args.model_cache_dir,
            seed=args.seed,
        )
    else:
        """
        model, processor = load_models(
            model_id=args.model_id,
            fast=args.fast,
            model_cache_dir=args.model_cache_dir,
        )
        """
        model, processor = load_model_and_processor(
            model_path=args.model_id,
            model_type=args.model_type
        )
        if args.use_gnn:
            if not args.cluster_results_path:
                raise ValueError("cluster_results_path must be provided when using GNN-based generation")
            response = run_latent_generation_with_gnn(
                model_type=args.model_type,
                model=model,
                processor=processor,
                model_id=args.model_id,
                image_1_path=args.image_1_path,
                prompt=args.prompt,
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
                use_opera=args.use_opera,
                opera_scale_factor=args.opera_scale_factor,
                opera_threshold=args.opera_threshold,
                opera_num_attn_candidates=args.opera_num_attn_candidates,
                opera_penalty_weights=args.opera_penalty_weights,
                use_vcd=args.use_vcd,
                use_sid=args.use_sid,
                noise_step=args.noise_step,
                cd_alpha=args.cd_alpha,
                cd_beta=args.cd_beta,
            )
        else:
            response = run_latent_generation(
                model_type=args.model_type,
                model=model,
                processor=processor,
                model_id=args.model_id,
                image_1_path=args.image_1_path,
                image_2_path=args.image_2_path,
                text_input=args.text_input,
                prompt=args.prompt,
                layer=args.layer,
                operation=args.operation,
                weight=args.weight,
                use_mean=args.use_mean,
                max_new_tokens=args.max_new_tokens,
                fast=args.fast,
                model_cache_dir=args.model_cache_dir,
                seed=args.seed,
            )
