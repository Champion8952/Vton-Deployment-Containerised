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
    except Exception as e:
        logger.error(f"Error parsing arguments: {str(e)}")
        raise


class Config:
    CHECKPOINTS_DIR = r"pretrained"
    CHECKPOINTS = {
        "1b": "sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2",
        "2b": "sapiens_2b_goliath_best_goliath_mIoU_8131_epoch_200_torchscript.pt2",
    }
    KEEP_CATEGORIES = [
        "Upper_Clothing",
        "Torso", 
        "Right_Upper_Arm",
        "Left_Upper_Arm",
        "Right_Lower_Arm",
        "Left_Lower_Arm"
    ]

class ModelManager:
    _models = {}

    @classmethod
    def initialize_models(cls):
        try:
            if not os.path.exists(Config.CHECKPOINTS_DIR):
                raise FileNotFoundError(f"Checkpoints directory {Config.CHECKPOINTS_DIR} does not exist")
                
            for model_name in Config.CHECKPOINTS:
                checkpoint_path = os.path.join(Config.CHECKPOINTS_DIR, Config.CHECKPOINTS[model_name])
                if not os.path.exists(checkpoint_path):
                    raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found")
                    
                try:
                    model = torch.jit.load(checkpoint_path)
                except RuntimeError as e:
                    logger.error(f"Error loading model {model_name}: {str(e)}")
                    raise RuntimeError(f"Failed to load model {model_name}") from e
                    
                model.eval()
                try:
                    if not torch.cuda.is_available():
                        raise RuntimeError("CUDA is not available")
                    model.to("cuda")
                except RuntimeError as e:
                    logger.error(f"Error moving model to CUDA: {str(e)}")
                    raise
                    
                cls._models[model_name] = model
                
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise

    @classmethod
    def get_model(cls, model_name: str):
        if model_name not in cls._models:
            logger.error(f"Model {model_name} not found")
            return None
        return cls._models[model_name]

    @staticmethod
    @torch.inference_mode()
    def run_model(model, input_tensor, height, width):
        try:
            output = model(input_tensor)
            output = F.interpolate(output, size=(height, width), mode="bilinear", align_corners=False)
            _, preds = torch.max(output, 1)
            return preds
        except RuntimeError as e:
            logger.error(f"Error running model inference: {str(e)}")
            raise RuntimeError("Model inference failed") from e

class ImageProcessor:
    def __init__(self):
        try:
            self.transform_fn = transforms.Compose([
                transforms.Resize((1024, 768)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[123.5/255, 116.5/255, 103.5/255], std=[58.5/255, 57.0/255, 57.5/255]),
            ])
            self.category_indices = {cat: idx for idx, cat in enumerate(GOLIATH_CLASSES)}
        except Exception as e:
            logger.error(f"Error initializing ImageProcessor: {str(e)}")
            raise

    def process_image(self, image: Image.Image, model_name: str):
        try:
            model = ModelManager.get_model(model_name)
            if model is None:
                raise ValueError(f"Invalid model name: {model_name}")
            
            try:
                input_tensor = self.transform_fn(image).unsqueeze(0).to("cuda")
            except RuntimeError as e:
                logger.error(f"Error processing input image: {str(e)}")
                raise RuntimeError("Failed to process input image") from e
            
            preds = ModelManager.run_model(model, input_tensor, image.height, image.width)
            
            mask = preds.squeeze(0)
            mask_upper_body = torch.zeros_like(mask, device="cuda")
            
            keep_categories = ["Upper_Clothing", "Torso", "Face_Neck", "Right_Upper_Arm", 
                             "Right_Lower_Arm", "Left_Lower_Arm", "Left_Upper_Arm"]
            keep_upper_body_indices = torch.tensor(
                [self.category_indices[cat] for cat in keep_categories if cat in self.category_indices], 
                device="cuda"
            )

            for idx in keep_upper_body_indices:
                mask_upper_body[mask == idx] = 255

            return mask_upper_body.cpu()
            
        except Exception as e:
            logger.error(f"Error in process_image: {str(e)}")
            raise

    @staticmethod
    def visualize_pred_with_overlay(img_tensor, sem_seg, alpha=0.5):
        try:
            overlay = torch.zeros((*sem_seg.shape, 3), dtype=torch.uint8, device="cuda")
            overlay[sem_seg > 0] = 255
            blended = (img_tensor * (1 - alpha) + overlay * alpha).to(torch.uint8)
            return blended
        except Exception as e:
            logger.error(f"Error in visualization: {str(e)}")
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

def visualize_dense_labels(image_path, densepose, atr_model, lip_model):
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        img = np.array(Image.open(image_path))
        height, width = img.shape[:2]

        try:
            dense_output = densepose(image_path)
            dense_mask = np.array(dense_output)
        except Exception as e:
            logger.error(f"DensePose model failed: {str(e)}")
            raise RuntimeError("DensePose processing failed") from e

        human_mask = (dense_mask != 0).astype(np.uint8)
        
        kernel = np.ones((5,5), np.uint8)
        human_mask = binary_dilation(human_mask, kernel, iterations=2)
        human_mask = binary_erosion(human_mask, kernel, iterations=1)
        human_mask = convex_hull_image(human_mask)

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
            
            temp_id = f"{int(time.time())%10000:04d}{''.join(random.choices('0123456789', k=4))}"
            temp_cropped_path = os.path.join(f"temp_cropped_{temp_id}.jpg")
            
            try:
                Image.fromarray(blank_img).save(temp_cropped_path)
            except IOError as e:
                logger.error(f"Failed to save temporary file: {str(e)}")
                raise

        human_mask_3ch = np.stack([human_mask] * 3, axis=-1)
        blurred_img = cv2.GaussianBlur(img, (3,3), 0)
        output_img = blurred_img.copy()
        output_img = np.where(human_mask_3ch == 1, img, output_img)

        temp_mask_path = temp_cropped_path 
        
        try:
            atr_mask = np.array(atr_model(temp_mask_path))
            lip_mask = np.array(lip_model(temp_mask_path))
        except Exception as e:
            logger.error(f"Model prediction failed: {str(e)}")
            raise RuntimeError("Model prediction failed") from e

        try:
            os.remove(temp_mask_path)
        except OSError as e:
            logger.error(f"Failed to remove temporary file: {str(e)}")
            raise

        dense_parts_mask = np.zeros_like(dense_mask)
        atr_clothes_mask = np.zeros_like(atr_mask)
        lip_clothes_mask = np.zeros_like(lip_mask)
        
        DENSEPOSE_PARTS = ["big arms", "forearms"]
        ATR_PARTS = ['Upper-clothes', 'Belt', 'Scarf', 'Dress', 'Skirt', 'Coat']    
        LIP_PARTS = []
        
        for part_name in DENSEPOSE_PARTS:
            if part_name in DENSE_INDEX_MAP:
                indices = DENSE_INDEX_MAP[part_name]
                for idx in indices:
                    dense_parts_mask = dense_parts_mask | (dense_mask == idx)
                    
        for part_name in ATR_PARTS:
            if part_name in ATR_MAPPING:
                atr_clothes_mask = atr_clothes_mask | part_mask_of(part_name, atr_mask, ATR_MAPPING)

        for part_name in LIP_PARTS:
            if part_name in LIP_MAPPING:
                lip_clothes_mask = lip_clothes_mask | part_mask_of(part_name, lip_mask, LIP_MAPPING)

        combined_mask = np.zeros_like(atr_clothes_mask)
        combined_mask = np.logical_or(combined_mask, dense_parts_mask)
        combined_mask = np.logical_or(combined_mask, atr_clothes_mask)
        combined_mask = np.logical_or(combined_mask, lip_clothes_mask)
        
        combined_mask = binary_dilation(combined_mask, iterations=2)
        combined_mask = binary_erosion(combined_mask, iterations=1)
        combined_mask = convex_hull_image(combined_mask)

        dense_labels_to_remove = ['hands']
        atr_labels_to_remove = []
        lip_labels_to_remove = ['Face', 'Hair']
        
        for label in dense_labels_to_remove:
            if label in DENSE_INDEX_MAP:
                indices = DENSE_INDEX_MAP[label]
                for idx in indices:
                    combined_mask[dense_mask == idx] = 0

        for label in atr_labels_to_remove:
            if label in ATR_MAPPING:
                idx = ATR_MAPPING[label]
                combined_mask[atr_mask == idx] = 0

        for label in lip_labels_to_remove:
            if label in LIP_MAPPING:
                idx = LIP_MAPPING[label]
                combined_mask[lip_mask == idx] = 0

        combined_mask = binary_dilation(combined_mask, iterations=1)
        combined_mask = binary_erosion(combined_mask, iterations=1)

        return combined_mask
        
    except Exception as e:
        logger.error(f"Error in visualize_dense_labels: {str(e)}")
        raise

def segment_image(image_path, densepose_model, atr_model, lip_model):
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        temp_image_path = image_path
        image = Image.open(temp_image_path)

        try:
            lip_mask = np.array(lip_model(temp_image_path))
            densepose_mask = np.array(densepose_model(temp_image_path))
        except Exception as e:
            logger.error(f"Model prediction failed: {str(e)}")
            raise RuntimeError("Model prediction failed") from e

        lip_face_mask = np.zeros_like(lip_mask)
        densepose_face_mask = np.zeros_like(densepose_mask)

        LIP_PARTS = ['Face']
        DENSEPOSE_PARTS = ['face']

        for part_name in LIP_PARTS:
            if part_name in LIP_MAPPING:
                lip_face_mask = lip_face_mask | part_mask_of(part_name, lip_mask, LIP_MAPPING)

        for part_name in DENSEPOSE_PARTS:
            if part_name in DENSE_INDEX_MAP:
                densepose_face_mask = densepose_face_mask | part_mask_of(part_name, densepose_mask, DENSE_INDEX_MAP)

        processor = ImageProcessor()
        try:
            mask_upper_body = processor.process_image(image, "1b")
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise RuntimeError("Image processing failed") from e
        
        mask_upper_body_uint8 = mask_upper_body.numpy().astype(np.uint8)
        mask_upper_body_uint8[lip_face_mask > 0] = 0
        mask_upper_body_uint8 = remove_small_holes(mask_upper_body_uint8, area_threshold=100)
        mask_upper_body_uint8 = diameter_opening(mask_upper_body_uint8, 20)
        mask_upper_body_uint8 = area_closing(mask_upper_body_uint8, 100)
        mask_upper_body_uint8 = remove_small_objects(mask_upper_body_uint8, min_size=500)

        kernel_size = 2
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        mask_upper_body_uint8 = cv2.dilate(mask_upper_body_uint8, kernel, iterations=2)
        mask_upper_body_uint8 = cv2.morphologyEx(mask_upper_body_uint8, cv2.MORPH_CLOSE, kernel)
        mask_upper_body_uint8 = cv2.GaussianBlur(mask_upper_body_uint8, (5, 5), 0)
        _, mask_upper_body_uint8 = cv2.threshold(mask_upper_body_uint8, 127, 255, cv2.THRESH_BINARY)
        
        mask_upper_body_image = Image.fromarray(mask_upper_body_uint8)
        return mask_upper_body_image
        
    except Exception as e:
        logger.error(f"Error in segment_image: {str(e)}")
        raise
