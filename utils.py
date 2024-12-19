import logging
import inspect
from datetime import datetime
from typing import Optional

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

