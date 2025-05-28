import requests
from PIL import Image
import torch
def load_image(image_path: str):
    if image_path.startswith("http"):
        image = Image.open(requests.get(image_path, stream=True).raw)
    else:
        image = Image.open(image_path)
    return image


def enhanced_projection(image_embeddings, text_embedding):
    """Calculate projection with normalization for better directional focus.
    
    Args:
        image_embeddings: Tensor of shape (1,N,D)
        text_embedding: Tensor of shape (1,1,D)
        
    Returns:
        Projection coefficients of shape (N,)
    """
    # Normalize embeddings to focus on direction rather than magnitude
    image_norm = torch.norm(image_embeddings, dim=2, keepdim=True)
    text_norm = torch.norm(text_embedding, dim=2, keepdim=True)
    
    image_normalized = image_embeddings / (image_norm + 1e-8)
    text_normalized = text_embedding / (text_norm + 1e-8)
    
    # Calculate projection using normalized embeddings
    numerator = (image_normalized @ text_normalized.transpose(-1, -2))[0, :, 0]
    denominator = (text_normalized @ text_normalized.transpose(-1, -2))[0, 0, 0]
    
    # Add small epsilon for numerical stability
    return numerator / (denominator + 1e-8)


def subtract_projection(image_embeddings, semantic_embedding, weight=1, device=None, use_mean=True, targets=None, use_cls=False):
    """Subtract text projection from image embeddings.
    
    Args:
        image_embeddings: Tensor of shape (1,N,D)
        semantic_embedding: Tensor of shape (1,X,D) - will be mean pooled to (1,1,D) or can use the last token embedding
        weight: Scaling factor for projection subtraction
        device: Optional device to move tensors to
        targets: list of indices to subtract projection from
        use_cls: if True, use the cls token embedding as the semantic embedding
        
    Returns:
        Modified image embeddings with same shape as input
    """
    # Move to device if specified
    if device is not None:
        image_embeddings = image_embeddings.to(device)
        semantic_embedding = semantic_embedding.to(device)
    
    # Mean pool semantic embedding
    if use_mean:
        
        if len(semantic_embedding.shape) == 2:
            semantic_embedding = semantic_embedding.unsqueeze(0)
        # Get top 1% most important embeddings based on L2 norm
        norms = torch.norm(semantic_embedding, dim=2)  # (1,N) where N is sequence length
        k = max(1, int(0.01 * semantic_embedding.shape[1]))  # 1% of sequence length
        _, top_indices = torch.topk(norms, k, dim=1)  # (1,k)
        top_embeddings = torch.gather(semantic_embedding, 1,
                                    top_indices.unsqueeze(-1).expand(-1, -1, semantic_embedding.shape[-1]))
        semantic_embedding = top_embeddings.mean(dim=1, keepdim=True)  # (1,1,D)
    else:
        semantic_embedding = semantic_embedding[:, -1:, :]  # (1,1,D)
    
    image_embeddings = image_embeddings.clone()
    proj = enhanced_projection(image_embeddings, semantic_embedding)  # (N,)
    
    if use_cls:
        image_embeddings[:,-1,:] -= weight * proj[-1] * semantic_embedding.squeeze(1)
    else:
        for i in range(image_embeddings.shape[1]-1):
            if targets:
                if i in targets:
                    image_embeddings[:,i] -= weight * proj[i] * semantic_embedding.squeeze(1)
            else:
                image_embeddings[:, i] -= weight * proj[i] * semantic_embedding.squeeze(1)
        
    
    return image_embeddings