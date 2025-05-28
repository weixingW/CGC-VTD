
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import ChameleonForConditionalGeneration, ChameleonProcessor, AutoModelForCausalLM, AutoTokenizer, AutoImageProcessor, AutoModel
from torchvision import transforms
import sys
sys.path.append(".")
sys.path.append("..")

from janus.models import MultiModalityCausalLM, VLChatProcessor
from emu3.mllm.processing_emu3 import Emu3Processor
from emu3.mllm.modeling_emu3 import Emu3ForCausalLM

def load_model_and_processor(model_path, model_type="chameleon",cache_dir=None):
    """
    Load the model and processor.
    
    Args:
        model_path (str): Path to the pretrained model
        model_type (str): Type of model to use ("chameleon", "llamagen", or "janus", or "Emu3")
        cache_dir (str): Path to cache directory
        
    Returns:
        tuple: (model, processor)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_type == "chameleon":
        model = ChameleonForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16, 
            device_map=device,
            output_hidden_states=True,
            cache_dir=cache_dir,
            attn_implementation="eager"
        )
        processor = ChameleonProcessor.from_pretrained(model_path)
        # Get vocabulary dimension for Chameleon
        vocab_size = model.model.vqmodel.quantize.embedding.weight.shape[0]
        embed_dim = model.model.vqmodel.quantize.embedding.weight.shape[1]
        print("\nVocabulary Dimension:")
        print(f"Model Type: Chameleon")
        print(f"Codebook Size: {vocab_size}")
        print(f"Embedding Dimension: {embed_dim}")
        print(f"Total Parameters in Codebook: {vocab_size * embed_dim:,}\n")
    elif model_type == "janus":
        # Load Janus model and processor
        vl_chat_processor = VLChatProcessor.from_pretrained(model_path,
                                                            cache_dir=cache_dir,
                                                            device_map=device)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=device,
            output_hidden_states=True,
            cache_dir=cache_dir,
            attn_implementation="eager"
        ).to(device).eval()
        
        # Get vocabulary dimension for Janus
        vocab_size = model.gen_vision_model.quantize.embedding.weight.shape[0]
        embed_dim = model.gen_vision_model.quantize.embedding.weight.shape[1]
        print("\nVocabulary Dimension:")
        print(f"Model Type: Janus")
        print(f"Codebook Size: {vocab_size}")
        print(f"Embedding Dimension: {embed_dim}")
        print(f"Total Parameters in Codebook: {vocab_size * embed_dim:,}\n")
        
        processor = vl_chat_processor
    elif model_type == "Emu3":

        model = Emu3ForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            cache_dir=cache_dir,
            attn_implementation="eager"
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            padding_side="left"
        )
        image_processor = AutoImageProcessor.from_pretrained(
            "BAAI/Emu3-VisionTokenizer", 
            trust_remote_code=True
        )
        # Load vision tokenizer on GPU with fp16
        image_tokenizer = AutoModel.from_pretrained(
            "BAAI/Emu3-VisionTokenizer",
            device_map=device,  # Force GPU
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).eval()
        
        # Move all components to GPU
        image_tokenizer = image_tokenizer.to(device)
        
        processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)
        vocab_size = image_tokenizer.config.codebook_size
        embed_dim = image_tokenizer.config.embed_dim
        print("\nVocabulary Dimension:")
        print(f"Model Type: Emu3")
        print(f"Codebook Size: {vocab_size}")
        print(f"Embedding Dimension: {embed_dim}")
        print(f"Total Parameters in Codebook: {vocab_size * embed_dim:,}\n")


    
    return model, processor