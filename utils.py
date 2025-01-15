import logging
import inspect
from datetime import datetime
from typing import Optional
import numpy as np
from PIL import Image, ImageOps

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [Line %(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

def log_with_line_number(level: str, message: str) -> None:
    caller_frame = inspect.currentframe().f_back
    line_number = caller_frame.f_lineno
    if level == 'error':
        logger.error(f"[Line {line_number}] {message}")
    elif level == 'warning':
        logger.warning(f"[Line {line_number}] {message}")
    else:
        logger.info(f"[Line {line_number}] {message}")

def format_time_taken(start_time: datetime, end_time: Optional[datetime] = None) -> float:
    if end_time is None:
        end_time = datetime.now()
    return (end_time - start_time).total_seconds()

_M_RGB2YUV = [[0.299, 0.587, 0.114], [-0.14713, -0.28886, 0.436], [0.615, -0.51499, -0.10001]]
_M_YUV2RGB = [[1.0, 0.0, 1.13983], [1.0, -0.39465, -0.58060], [1.0, 2.03211, 0.0]]

def convert_PIL_to_numpy(image, format):
    """
    Convert PIL image to numpy array of target format.

    Args:
        image (PIL.Image): a PIL image
        format (str): the format of output image

    Returns:
        (np.ndarray): also see `read_image`
    """
    if format is not None:
        # PIL only supports RGB, so convert to RGB and flip channels over below
        conversion_format = format
        if format in ["BGR", "YUV-BT.601"]:
            conversion_format = "RGB"
        image = image.convert(conversion_format)
    image = np.asarray(image)
    # PIL squeezes out the channel dimension for "L", so make it HWC
    if format == "L":
        image = np.expand_dims(image, -1)

    # handle formats not supported by PIL
    elif format == "BGR":
        # flip channels if needed
        image = image[:, :, ::-1]
    elif format == "YUV-BT.601":
        image = image / 255.0
        image = np.dot(image, np.array(_M_RGB2YUV).T)

    return image



def pil_to_binary_mask(pil_image, threshold=0):
    # Preserve EXIF orientation when converting to binary mask
    pil_image = ImageOps.exif_transpose(pil_image)
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    for i in range(binary_mask.shape[0]):
        for j in range(binary_mask.shape[1]):
            if binary_mask[i,j] == True :
                mask[i,j] = 1
    mask = (mask*255).astype(np.uint8)
    output_mask = Image.fromarray(mask)
    return output_mask

