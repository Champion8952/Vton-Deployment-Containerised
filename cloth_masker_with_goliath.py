import os 
import numpy as np
import cv2
from PIL import Image
import argparse
from typing import Union
from scipy.ndimage import binary_dilation, binary_erosion
from skimage.morphology import convex_hull_image, remove_small_holes, diameter_opening, area_closing, remove_small_objects
import time
import random
from classes_and_palettes import GOLIATH_CLASSES
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import io
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    try:
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

        return args
    except Exception as e:
        logger.error(f"Error parsing arguments: {str(e)}")
        raise


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

GOLIATH_MAPPING = {cat: idx for idx, cat in enumerate(GOLIATH_CLASSES)}

class GoliathMasker:
    def __init__(self):
        logger.info("Initializing ImageProcessor")
        self.transform_fn = transforms.Compose([
            transforms.Resize((1024, 768)),
            transforms.ToTensor(),
        ])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    @torch.inference_mode()
    def run_model(model, input_tensor, height, width):
        output = model(input_tensor)
        output = F.interpolate(output, size=(height, width), mode="bilinear", align_corners=False)
        _, preds = torch.max(output, 1)
        return preds
    
    def process_image(self, image: Image.Image, goliath_model):
        start_time = time.time()
        logger.info("Processing image with Goliath model")

        model = goliath_model.to(self.device)
        input_tensor = self.transform_fn(image).unsqueeze(0).to(self.device)

        # TODO: Make this Faster 
        with torch.inference_mode():
            preds = self.run_model(model, input_tensor, image.height, image.width)
            mask = preds.squeeze(0)
            mask = (mask.cpu().numpy()).astype(np.uint8)
            logger.info(f"Image processed in {time.time() - start_time:.2f} seconds")
            return mask


def part_mask_of(part: Union[str, list],
                parse: np.ndarray, mapping: dict):
    try:
        if isinstance(part, str):
            part = [part]
        mask = np.zeros_like(parse)
        for _ in part:
            if _ not in mapping:
                logger.warning(f"Part {_} not found in mapping")
                continue
            if isinstance(mapping[_], list):
                for i in mapping[_]:
                    mask += (parse == i)
            else:
                mask += (parse == mapping[_])
        return mask
    except Exception as e:
        logger.error(f"Error creating part mask: {str(e)}")
        raise

def add_noise_and_denoise(image, noise_factor=0.1):
    try:
        img_float = image.astype(np.float32) / 255.0
        noise = np.random.normal(0, noise_factor, img_float.shape)
        noisy_img = np.clip(img_float + noise, 0, 1)
        noisy_img = (noisy_img * 255).astype(np.uint8)
        denoised_img = cv2.bilateralFilter(noisy_img, d=9, sigmaColor=75, sigmaSpace=75)
        return denoised_img
    except Exception as e:
        logger.error(f"Error in noise processing: {str(e)}")
        raise

def convert_image_to_tensor(image_array, size=(768, 1024)):
    """
    Convert numpy array or PIL Image to tensor
    
    Args:
        image_array: Can be numpy array or PIL Image
        size: Target size (width, height)
    """
    # Convert numpy array to PIL Image if needed
    if isinstance(image_array, np.ndarray):
        # If array is BGR (from OpenCV), convert to RGB
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image_array.astype('uint8'))
    else:
        image = image_array

    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    tensor = transform(image).unsqueeze(0)
    return tensor

processor = GoliathMasker()

def visualize_dense_labels(image_path, densepose, atr_model, lip_model, goliath_model):
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        img = np.array(Image.open(image_path))
        height, width = img.shape[:2]

        # Add resize function for masks
        def resize_mask(mask, target_height, target_width):
            return cv2.resize(mask.astype(np.uint8), (target_width, target_height), 
                            interpolation=cv2.INTER_NEAREST)

        # Get DensePose mask and resize
        try:
            dense_output = densepose(image_path)
            dense_mask = np.array(dense_output)
            dense_mask = resize_mask(dense_mask, height, width)
        except Exception as e:
            logger.error(f"DensePose model failed: {str(e)}")
            raise RuntimeError("DensePose processing failed") from e

        human_mask = (dense_mask != 0).astype(np.uint8)
        
        kernel = np.ones((5,5), np.uint8)
        human_mask = binary_dilation(human_mask, kernel, iterations=2)
        human_mask = binary_erosion(human_mask, kernel, iterations=1)

        # Crop the image to the human mask
        tmp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "tmp"))
        os.makedirs(tmp_dir, exist_ok=True)
        temp_id = f"{int(time.time())%10000:04d}{''.join(random.choices('0123456789', k=4))}"
        temp_cropped_path = os.path.join(tmp_dir, f"temp_cropped_{temp_id}.jpg")
        
        y_indices, x_indices = np.where(human_mask > 0)
        if len(y_indices) > 0 and len(x_indices) > 0:
            top, bottom = y_indices.min(), y_indices.max()
            left, right = x_indices.min(), x_indices.max()
            
            padding = 10
            top = max(0, top - padding)
            bottom = min(height, bottom + padding)
            left = max(0, left - padding)
            right = min(width, right + padding)
            
            cropped_img = img[top:bottom, left:right]
            blank_img = np.zeros_like(img)
            blank_img[top:bottom, left:right] = cropped_img

            
            
            try:
                Image.fromarray(blank_img).save(temp_cropped_path)
                if not os.path.exists(temp_cropped_path):
                    raise IOError(f"Failed to save temporary file at {temp_cropped_path}")
            except IOError as e:
                logger.error(f"Failed to save temporary file: {str(e)}")
                raise

        temp_mask_path = temp_cropped_path 

        # Get Goliath mask and resize
        try:
            #Convert to PIL Image
            goliath_image  = Image.open(temp_cropped_path)
            goliath_mask = processor.process_image(goliath_image, goliath_model)
            goliath_mask = resize_mask(goliath_mask, height, width)
        except Exception as e:
            logger.error(f"Goliath model failed: {str(e)}")
            raise RuntimeError("Goliath processing failed") from e

        # Get ATR mask and resize
        try:
            atr_mask = np.array(atr_model(temp_mask_path))
            atr_mask = resize_mask(atr_mask, height, width)
            
            lip_mask = np.array(lip_model(temp_mask_path))
            lip_mask = resize_mask(lip_mask, height, width)
        except Exception as e:
            logger.error(f"Model prediction failed: {str(e)}")
            raise RuntimeError("Model prediction failed") from e

        try:
            os.remove(temp_cropped_path)
        except OSError as e:
            logger.error(f"Failed to remove temporary file: {str(e)}")
            raise

        atr_clothes_mask = np.zeros_like(atr_mask)
        lip_clothes_mask = np.zeros_like(lip_mask)
        goliath_upper_parts_mask = np.zeros_like(goliath_mask)
        goliath_right_arms_mask = np.zeros_like(goliath_mask)
        goliath_left_arms_mask = np.zeros_like(goliath_mask)

        # Make Masks for Upper Body & Dilute the Mask & Add Convex Hull
        ATR_PARTS = ['Belt', 'Scarf', 'Dress', 'Skirt', 'Coat']  
        LIP_PARTS = ['Upper-clothes']
        GOLIATH_CLASSES_UPPER = ['Upper_Clothing', 'Torso', 'Face_Neck']
        
        for part_name in ATR_PARTS:
            if part_name in ATR_MAPPING:
                atr_clothes_mask = atr_clothes_mask | part_mask_of(part_name, atr_mask, ATR_MAPPING)

        for part_name in LIP_PARTS:
            if part_name in LIP_MAPPING:
                lip_clothes_mask = lip_clothes_mask | part_mask_of(part_name, lip_mask, LIP_MAPPING)
       
        for label in GOLIATH_CLASSES_UPPER:
            if label in GOLIATH_MAPPING:
                goliath_upper_parts_mask = goliath_upper_parts_mask | (goliath_mask == GOLIATH_MAPPING[label])
        
        combined_upper_mask = np.zeros_like(dense_mask, dtype=bool)
        combined_upper_mask = np.logical_or(combined_upper_mask, atr_clothes_mask)
        combined_upper_mask = np.logical_or(combined_upper_mask, goliath_upper_parts_mask)

        # Dilute the Mask
        combined_upper_mask = binary_dilation(combined_upper_mask, iterations=2)
        combined_upper_mask = binary_erosion(combined_upper_mask, iterations=1)

        # Remove small objects and small holes
        combined_upper_mask = remove_small_objects(combined_upper_mask, min_size=1000)
        combined_upper_mask = remove_small_holes(combined_upper_mask, area_threshold=1500)
        
        GOLIATH_CLASSES_RIGHT_ARMS = ['Right_Upper_Arm', 'Right_Lower_Arm']
        for part_name in GOLIATH_CLASSES_RIGHT_ARMS:
            if part_name in GOLIATH_MAPPING:
                goliath_right_arms_mask = goliath_right_arms_mask | (goliath_mask == GOLIATH_MAPPING[part_name])

        goliath_right_arms_mask = binary_dilation(goliath_right_arms_mask, iterations=2)
        goliath_right_arms_mask = binary_erosion(goliath_right_arms_mask, iterations=1)

        # Make Masks for Left Arms
        GOLIATH_CLASSES_LEFT_ARMS = ['Left_Lower_Arm', 'Left_Upper_Arm']
        for part_name in GOLIATH_CLASSES_LEFT_ARMS:
            goliath_left_arms_mask = goliath_left_arms_mask | (goliath_mask == GOLIATH_MAPPING[part_name])

        goliath_left_arms_mask = binary_dilation(goliath_left_arms_mask, iterations=2)
        goliath_left_arms_mask = binary_erosion(goliath_left_arms_mask, iterations=1)

        # Combine all masks
        combined_mask = np.zeros_like(dense_mask, dtype=bool)
        combined_mask = np.logical_or(combined_mask, combined_upper_mask)
        combined_mask = np.logical_or(combined_mask, goliath_right_arms_mask)
        combined_mask = np.logical_or(combined_mask, goliath_left_arms_mask)

        # Add Face_Neck from Goliath mask
        face_neck_mask = (goliath_mask == GOLIATH_MAPPING['Face_Neck'])
        combined_mask = np.logical_or(combined_mask, face_neck_mask)

        GOLIATH_LABELS_TO_REMOVE = ['Right_Hand', 'Left_Hand']
        DENSEPOSE_LABELS_TO_REMOVE = ['face','hands']
        LIP_LABELS_TO_REMOVE = ['face']

        for label in DENSEPOSE_LABELS_TO_REMOVE:
            if label in DENSE_INDEX_MAP:
                indices = DENSE_INDEX_MAP[label]
                for idx in indices:
                    combined_mask[dense_mask == idx] = 0

        for label in LIP_LABELS_TO_REMOVE:
            if label in LIP_MAPPING:
                idx = LIP_MAPPING[label]
                combined_mask[lip_mask == idx] = 0

        for label in GOLIATH_LABELS_TO_REMOVE:
            if label in GOLIATH_MAPPING:
                combined_mask[goliath_mask == GOLIATH_MAPPING[label]] = 0

        GOLIATH_BACKGROUND_LABEL = 'Background'
        if GOLIATH_BACKGROUND_LABEL in GOLIATH_MAPPING:
            combined_mask[goliath_mask == GOLIATH_MAPPING[GOLIATH_BACKGROUND_LABEL]] = 0

        # Remove small objects
        combined_mask = remove_small_objects(combined_mask, min_size=1000)
        combined_mask = area_closing(combined_mask, area_threshold=1500)
        combined_mask = remove_small_holes(combined_mask, area_threshold=1500)

        # Dilute the Mask
        # Increase mask around edges using morphological operations
        combined_mask = binary_dilation(combined_mask, iterations=1)
        combined_mask = binary_erosion(combined_mask, iterations=1)

        # Add Convex Hull
        combined_mask = convex_hull_image(combined_mask)

        # Remove Background
        combined_mask[goliath_mask == GOLIATH_MAPPING['Background']] = 0
        # Image.fromarray((combined_mask*255).astype(np.uint8)).save("28_background_removed.jpg")

        return combined_mask
        
    except Exception as e:
        logger.error(f"Error in visualize_dense_labels: {str(e)}")
        raise

