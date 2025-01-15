import numpy as np
import cv2
from PIL import Image
from typing import Union
from skimage.morphology import remove_small_holes, diameter_opening, area_closing, remove_small_objects
from classes_and_palettes import GOLIATH_CLASSES
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

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

@staticmethod
@torch.inference_mode()
def run_model(model, input_tensor, height, width):
    output = model(input_tensor)
    output = F.interpolate(output, size=(height, width), mode="bilinear", align_corners=False)
    _, preds = torch.max(output, 1)
    return preds

class ImageProcessor:
    def __init__(self):
        logger.info("Initializing ImageProcessor")
        self.transform_fn = transforms.Compose([
            transforms.Resize((1024, 768)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[123.5/255, 116.5/255, 103.5/255], 
                              std=[58.5/255, 57.0/255, 57.5/255]),
        ])
        self.category_indices = {cat: idx for idx, cat in enumerate(GOLIATH_CLASSES)}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.keep_categories = ["Upper_Clothing", "Torso", "Face_Neck", "Right_Upper_Arm", 
                            "Right_Lower_Arm", "Left_Lower_Arm", "Left_Upper_Arm"]
        self.keep_upper_body_indices = None
        logger.info(f"ImageProcessor initialized with device: {self.device}")

    def process_image(self, image: Image.Image, goliath_model):
        start_time = time.time()
        logger.info("Processing image with Goliath model")
        
        if self.keep_upper_body_indices is None:
            self.keep_upper_body_indices = torch.tensor(
                [self.category_indices[cat] for cat in self.keep_categories if cat in self.category_indices], 
                device=self.device
            )

        model = goliath_model.to(self.device)
        input_tensor = self.transform_fn(image).unsqueeze(0).to(self.device)

        # TODO: Make this Faster 
        with torch.inference_mode():
            preds = run_model(model, input_tensor, image.height, image.width)
            mask = preds.squeeze(0)
            mask_upper_body = torch.zeros_like(mask, device=self.device)
            
            for idx in self.keep_upper_body_indices:
                mask_upper_body[mask == idx] = 255

            logger.info(f"Image processed in {time.time() - start_time:.2f} seconds")
            return mask_upper_body.cpu()

DENSE_INDEX_MAP = {
    "face": [23, 24],
}

LIP_MAPPING = {
    'Face': 13
}

def part_mask_of(part: Union[str, list], parse: np.ndarray, mapping: dict):
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

def segment_image(image_input, densepose_model, atr_model, lip_model, goliath_model):
    start_time = time.time()
    logger.info("Starting image segmentation")
    
    temp_image_path = image_input
    image = Image.open(temp_image_path)

    # Run face detection models in parallel
    logger.info("Running face detection models")
    lip_mask = np.array(lip_model(temp_image_path))
    densepose_mask = np.array(densepose_model(temp_image_path))

    # Create face masks
    logger.info("Creating face masks")
    lip_face_mask = part_mask_of('Face', lip_mask, LIP_MAPPING)
    densepose_face_mask = part_mask_of('face', densepose_mask, DENSE_INDEX_MAP)

    # Process upper body
    logger.info("Processing upper body")
    processor = ImageProcessor()
    mask_upper_body = processor.process_image(image, goliath_model)
    mask_upper_body_uint8 = (mask_upper_body.numpy() > 0).astype(np.uint8) * 255

    # Combine face masks and remove from upper body
    logger.info("Combining and processing masks")
    combined_face_mask = np.logical_or(lip_face_mask, densepose_face_mask).astype(np.uint8) * 255
    face_kernel = np.ones((5,5), np.uint8)
    combined_face_mask = cv2.dilate(combined_face_mask, face_kernel, iterations=2)
    mask_upper_body_uint8[combined_face_mask > 0] = 0

    # Post-process mask
    logger.info("Post-processing mask")
    # Clean up mask with morphological operations
    kernel = np.ones((15, 15), np.uint8)
    logger.info("Removing small objects from mask")
    mask_upper_body_uint8 = remove_small_objects(mask_upper_body_uint8 > 0, min_size=200)
    
    logger.info("Dilating mask")
    mask_upper_body_uint8 = cv2.dilate(mask_upper_body_uint8.astype(np.uint8) * 255, kernel, iterations=3)
    
    logger.info("Applying morphological closing operation") 
    mask_upper_body_uint8 = cv2.morphologyEx(mask_upper_body_uint8, cv2.MORPH_CLOSE, kernel)
    
    logger.info("Applying Gaussian blur and thresholding")
    blurred = cv2.GaussianBlur(mask_upper_body_uint8, (7, 7), 0)
    _, mask_upper_body_uint8 = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    logger.info(f"Image segmentation completed in {time.time() - start_time:.2f} seconds")
    return mask_upper_body_uint8
