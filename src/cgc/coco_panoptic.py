import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import json
import skimage.io as io
from pycocotools.coco import COCO
import cv2
from panopticapi.utils import id2rgb, rgb2id


def load_coco_panoptic_data(panoptic_json, image_dir, panoptic_dir):
    """
    Load COCO panoptic segmentation dataset
    
    Args:
        panoptic_json: Path to the panoptic annotations JSON file
        image_dir: Directory containing the images
        panoptic_dir: Directory containing the panoptic segmentation PNGs
        
    Returns:
        panoptic_coco: Loaded panoptic annotations
        categories_dict: Dictionary mapping category IDs to category info
    """
    # Load panoptic annotations
    with open(panoptic_json, 'r') as f:
        panoptic_coco = json.load(f)
    
    # Create a lookup table for categories
    categories_dict = {}
    for category in panoptic_coco['categories']:
        categories_dict[category['id']] = category
    
    print(f"Loaded panoptic COCO dataset with {len(panoptic_coco['annotations'])} images and {len(categories_dict)} categories")
    
    return panoptic_coco, categories_dict


def get_panoptic_image_and_masks(panoptic_coco, categories_dict, img_id, image_dir, panoptic_dir):
    """
    Get image and its corresponding panoptic segmentation masks
    
    Args:
        panoptic_coco: Loaded panoptic annotations
        categories_dict: Dictionary mapping category IDs to category info
        img_id: Image ID to process
        image_dir: Directory containing the images
        panoptic_dir: Directory containing the panoptic segmentation PNGs
        
    Returns:
        img: The loaded image
        segments_info: List of segment infos
        panoptic_seg: The panoptic segmentation (colored)
        panoptic_seg_id: The panoptic segmentation (IDs)
    """
    # Find the image info and annotation
    img_info = None
    annotation = None
    
    for image in panoptic_coco['images']:
        if image['id'] == img_id:
            img_info = image
            break
    
    for ann in panoptic_coco['annotations']:
        if ann['image_id'] == img_id:
            annotation = ann
            break
    
    if img_info is None or annotation is None:
        raise ValueError(f"Image ID {img_id} not found in the panoptic dataset")
    
    # Load the original image
    img_path = os.path.join(image_dir, img_info['file_name'])
    img = io.imread(img_path)
    if len(img.shape) == 2:  # Convert grayscale to RGB
        img = np.stack((img,) * 3, axis=-1)
    
    # Load the panoptic segmentation PNG
    panoptic_path = os.path.join(panoptic_dir, annotation['file_name'])
    panoptic_seg = np.array(Image.open(panoptic_path))
    
    # Convert RGB encoding to segment IDs
    panoptic_seg_id = rgb2id(panoptic_seg)
    
    # Get segment info
    segments_info = annotation['segments_info']
    
    return img, segments_info, panoptic_seg, panoptic_seg_id


def visualize_panoptic_segmentation(img, panoptic_seg, segments_info, categories_dict, save_path=None):
    """
    Visualize an image with its panoptic segmentation
    
    Args:
        img: The original image
        panoptic_seg: The panoptic segmentation (colored)
        segments_info: List of segment infos from the annotation
        categories_dict: Dictionary mapping category IDs to category info
        save_path: Optional path to save the visualization
    """
    plt.figure(figsize=(15, 8))
    
    # Create a subplot for the original image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')
    
    # Create a subplot for the panoptic segmentation
    plt.subplot(1, 2, 2)
    plt.imshow(panoptic_seg)
    plt.title("Panoptic Segmentation")
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path)
    
    plt.tight_layout()
    plt.show()
    
    # Print information about the segments
    print(f"Image contains {len(segments_info)} segments:")
    for segment in segments_info:
        category = categories_dict[segment['category_id']]
        isthing = category['isthing']
        category_name = category['name']
        segment_id = segment['id']
        area = segment['area']
        
        print(f"ID: {segment_id}, Category: {category_name}, {'Thing' if isthing else 'Stuff'}, Area: {area} pixels")


def extract_individual_masks(panoptic_seg_id, segments_info, categories_dict):
    """
    Extract individual masks from a panoptic segmentation
    
    Args:
        panoptic_seg_id: The panoptic segmentation with segment IDs
        segments_info: List of segment infos from the annotation
        categories_dict: Dictionary mapping category IDs to category info
        
    Returns:
        thing_masks: Dictionary mapping thing category names to list of binary masks
        stuff_masks: Dictionary mapping stuff category names to list of binary masks
    """
    thing_masks = {}
    stuff_masks = {}
    
    for segment in segments_info:
        category_id = segment['category_id']
        segment_id = segment['id']
        category = categories_dict[category_id]
        category_name = category['name']
        
        # Extract the binary mask for this segment
        binary_mask = (panoptic_seg_id == segment_id).astype(np.uint8)
        
        # Separate things and stuff
        if category['isthing']:
            if category_name not in thing_masks:
                thing_masks[category_name] = []
            thing_masks[category_name].append(binary_mask)
        else:
            if category_name not in stuff_masks:
                stuff_masks[category_name] = []
            stuff_masks[category_name].append(binary_mask)
    
    return thing_masks, stuff_masks


def visualize_category_masks(img, masks_dict, title, is_thing=True, alpha=0.5, save_path=None):
    """
    Visualize masks for specific categories
    
    Args:
        img: The original image
        masks_dict: Dictionary mapping category names to list of binary masks
        title: Title for the plot
        is_thing: Whether visualizing 'thing' or 'stuff' categories
        alpha: Transparency for the masks
        save_path: Optional path to save the visualization
    """
    if not masks_dict:
        print(f"No {'thing' if is_thing else 'stuff'} categories found")
        return
    
    n_categories = len(masks_dict)
    fig, axes = plt.subplots(1, n_categories, figsize=(5*n_categories, 5))
    
    # Handle case of a single category
    if n_categories == 1:
        axes = [axes]
    
    for ax, (category_name, masks) in zip(axes, masks_dict.items()):
        ax.imshow(img)
        ax.set_title(f"{category_name} ({len(masks)} instances)")
        ax.axis('off')
        
        # Generate a random color for each instance of this category
        colors = np.random.random((len(masks), 3))
        
        # Plot each instance mask with its unique color
        for i, mask in enumerate(masks):
            colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3))
            for c in range(3):
                colored_mask[:, :, c] = mask * colors[i, c]
            
            ax.imshow(colored_mask, alpha=alpha)
    
    plt.suptitle(title)
    
    if save_path:
        plt.savefig(save_path)
    
    plt.tight_layout()
    plt.show()



