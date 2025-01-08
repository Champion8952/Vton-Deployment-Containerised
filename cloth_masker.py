import os 
import numpy as np
import cv2
from PIL import Image
import argparse
from typing import Union
from scipy.ndimage import binary_dilation, binary_erosion
from skimage.morphology import convex_hull_image
import time
import random


def parse_args():
    parser = argparse.ArgumentParser(description="Simple Cloth Masker.")
    parser.add_argument(
        "--width",
        type=int,
        default=768,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        default=True,
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


DENSE_INDEX_MAP = {
    "background": [0],
    "torso": [1, 2],
    "right hand": [3],
    "left hand": [4],
    "right foot": [5],
    "left foot": [6],
    "right thigh": [7, 9],
    "left thigh": [8, 10],
    "right leg": [11, 13],
    "left leg": [12, 14],
    "left big arm": [15, 17],
    "right big arm": [16, 18],
    "left forearm": [19, 21],
    "right forearm": [20, 22],
    "face": [23, 24],
    "thighs": [7, 8, 9, 10],
    "legs": [11, 12, 13, 14],
    "hands": [3, 4],
    "feet": [5, 6],
    "big arms": [15, 16, 17, 18],
    "forearms": [19, 20, 21, 22],
    "neck": [25, 26],
    "hair": [27, 28],
}

ATR_MAPPING = {
    'Background': 0, 'Hat': 1, 'Hair': 2, 'Sunglasses': 3, 
    'Upper-clothes': 4, 'Skirt': 5, 'Pants': 6, 'Dress': 7,
    'Belt': 8, 'Left-shoe': 9, 'Right-shoe': 10, 'Face': 11, 
    'Left-leg': 12, 'Right-leg': 13, 'Left-arm': 14, 'Right-arm': 15,
    'Bag': 16, 'Scarf': 17
}

LIP_MAPPING = {
    'Background': 0, 'Hat': 1, 'Hair': 2, 'Glove': 3, 
    'Sunglasses': 4, 'Upper-clothes': 5, 'Dress': 6, 'Coat': 7,
    'Socks': 8, 'Pants': 9, 'Jumpsuits': 10, 'Scarf': 11, 
    'Skirt': 12, 'Face': 13, 'Left-arm': 14, 'Right-arm': 15,
    'Left-leg': 16, 'Right-leg': 17, 'Left-shoe': 18, 'Right-shoe': 19
}

PASCAL_PART_MAPPING = {
    'background': 0, 'head': 1, 'torso': 2, 'upper_arm': 3,
    'lower_arm': 4, 'hand': 5, 'upper_leg': 6, 'lower_leg': 7,
    'foot': 8, 'neck': 9, 'hair': 10
}

def part_mask_of(part: Union[str, list],
                parse: np.ndarray, mapping: dict):
    if isinstance(part, str):
        part = [part]
    mask = np.zeros_like(parse)
    for _ in part:
        if _ not in mapping:
            continue
        if isinstance(mapping[_], list):
            for i in mapping[_]:
                mask += (parse == i)
        else:
            mask += (parse == mapping[_])
    return mask

def add_noise_and_denoise(image, noise_factor=0.1):
    """Add Gaussian noise to image and denoise using bilateral filter"""
    # Convert to float32 for noise addition
    img_float = image.astype(np.float32) / 255.0
    
    # Add Gaussian noise
    noise = np.random.normal(0, noise_factor, img_float.shape)
    noisy_img = img_float + noise
    noisy_img = np.clip(noisy_img, 0, 1)
    
    # Convert back to uint8
    noisy_img = (noisy_img * 255).astype(np.uint8)
    
    # Apply bilateral filter for denoising
    denoised_img = cv2.bilateralFilter(noisy_img, d=9, sigmaColor=75, sigmaSpace=75)
    
    return denoised_img

def visualize_dense_labels(image_path, densepose, atr_model, lip_model):
    """
    Visualize DensePose labels on an image by detecting the main human and leaving the background.
    
    Args:
        image_path: Path to input image
        save_path: Optional path to save the visualization. If None, displays the image.
    """
    
    # Read image and convert to numpy array
    # Handle both PIL Image and string path inputs
    img = np.array(Image.open(image_path))
    
    # Show image
    # Image.fromarray(img).show()

    height, width = img.shape[:2]


    # Get predictions
    dense_output = densepose(image_path)
    dense_mask = np.array(dense_output)

    # Create human mask (everything except background)
    human_mask = (dense_mask != 0).astype(np.uint8)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5,5), np.uint8)
    human_mask = binary_dilation(human_mask, kernel, iterations=2)
    human_mask = binary_erosion(human_mask, kernel, iterations=1)
    
    # Create convex hull to fill holes
    human_mask = convex_hull_image(human_mask)

    # Crop human mask image from original image
    # Find bounding box of human mask
    y_indices, x_indices = np.where(human_mask > 0)
    if len(y_indices) > 0 and len(x_indices) > 0:
        top, bottom = y_indices.min(), y_indices.max()
        left, right = x_indices.min(), x_indices.max()
        
        # Add padding around the bounding box
        padding = 10
        top = max(0, top - padding)
        bottom = min(height, bottom + padding)
        left = max(0, left - padding)
        right = min(width, right + padding)
        
        # Crop the image using the bounding box
        cropped_img = img[top:bottom, left:right]
        
        # Create blank image of original size
        blank_img = np.zeros_like(img)
        
        # Place cropped image back at original position
        blank_img[top:bottom, left:right] = cropped_img
        
        # Save image temporarily
        temp_id = f"{int(time.time())%10000:04d}{''.join(random.choices('0123456789', k=4))}"
        temp_cropped_path = os.path.join(f"temp_cropped_{temp_id}.jpg")
        Image.fromarray(blank_img).save(temp_cropped_path)

    # Expand human_mask to 3 channels to match the image
    human_mask_3ch = np.stack([human_mask] * 3, axis=-1)
    blurred_img = cv2.GaussianBlur(img, (3,3), 0)
    
    # Create output image starting with blurred background
    output_img = blurred_img.copy()
    
    # Apply human mask to keep original image in foreground
    output_img = np.where(human_mask_3ch == 1, img, output_img)

    temp_mask_path = temp_cropped_path 
    
    # SCHP Prediction
    atr_mask = np.array(atr_model(temp_mask_path))
    lip_mask = np.array(lip_model(temp_mask_path))

    # Clean up temporary file
    os.remove(temp_mask_path)

    # Initialize masks
    dense_parts_mask = np.zeros_like(dense_mask)
    atr_clothes_mask = np.zeros_like(atr_mask)
    lip_clothes_mask = np.zeros_like(lip_mask)
    
    # Masks to add in the Final Mask
    # Define label arrays for each segmentation model
    DENSEPOSE_PARTS = ["big arms", "forearms"]
    ATR_PARTS = ['Upper-clothes', 'Belt', 'Scarf', 'Dress', 'Skirt', 'Coat']    
    LIP_PARTS = []
    
    # Create DensePose part masks
    for part_name in DENSEPOSE_PARTS:
        if part_name in DENSE_INDEX_MAP:
            indices = DENSE_INDEX_MAP[part_name]
            for idx in indices:
                dense_parts_mask = dense_parts_mask | (dense_mask == idx)
                
    # Create ATR part masks
    for part_name in ATR_PARTS:
        if part_name in ATR_MAPPING:
            atr_clothes_mask = atr_clothes_mask | part_mask_of(part_name, atr_mask, ATR_MAPPING)

    # Create LIP part masks  
    for part_name in LIP_PARTS:
        if part_name in LIP_MAPPING:
            lip_clothes_mask = lip_clothes_mask | part_mask_of(part_name, lip_mask, LIP_MAPPING)

    # Combine all masks with boolean operations
    combined_mask = np.zeros_like(atr_clothes_mask)
    combined_mask = np.logical_or(combined_mask, dense_parts_mask)
    combined_mask = np.logical_or(combined_mask, atr_clothes_mask)
    combined_mask = np.logical_or(combined_mask, lip_clothes_mask)
    
    # Clean up mask using morphological operations
    combined_mask = binary_dilation(combined_mask, iterations=2)
    combined_mask = binary_erosion(combined_mask, iterations=1)

    # Add convex hull to fill holes
    combined_mask = convex_hull_image(combined_mask)

    # Labels to be removed from the final mask
    dense_labels_to_remove = ['hands']
    atr_labels_to_remove = []
    lip_labels_to_remove = ['Face', 'Hair']
    
    # Remove areas from DensePose mask
    for label in dense_labels_to_remove:
        if label in DENSE_INDEX_MAP:
            indices = DENSE_INDEX_MAP[label]
            for idx in indices:
                combined_mask[dense_mask == idx] = 0

    # Remove areas from ATR mask  
    for label in atr_labels_to_remove:
        if label in ATR_MAPPING:
            idx = ATR_MAPPING[label]
            combined_mask[atr_mask == idx] = 0

    # Remove areas from LIP mask
    for label in lip_labels_to_remove:
        if label in LIP_MAPPING:
            idx = LIP_MAPPING[label]
            combined_mask[lip_mask == idx] = 0

    # Clean up the refined mask
    combined_mask = binary_dilation(combined_mask, iterations=1)
    combined_mask = binary_erosion(combined_mask, iterations=1)

    return combined_mask
