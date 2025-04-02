from flask import Flask, request, jsonify, send_file
from flask.wrappers import Response
from flask_cors import CORS
import torch
import os
import math
from PIL import Image
import numpy as np
import io
import logging
import time
import random
from typing import List
from huggingface_hub import hf_hub_download
from preprocess.humanparsing.run_parsing import Parsing
from check_front_image import analyze_image
from preprocess.dwpose import DWposeDetector
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from src.pose_guider import PoseGuider
from cloth_masker_with_goliath import visualize_dense_labels
from src.pipeline_stable_diffusion_3_tryon import StableDiffusion3TryOnPipeline
from src.transformer_sd3_garm import SD3Transformer2DModel as SD3Transformer2DModel_Garm
from src.transformer_sd3_vton import SD3Transformer2DModel as SD3Transformer2DModel_Vton
from model.SCHP import SCHP
from model.DensePose import DensePose
from src.utils_mask import get_mask_location

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class TryOnInferenceEngine:
    def __init__(self):
        logger.info("Initializing TryOnInferenceEngine")
        start_time = time.time()
        
        self.device = "cuda"
        self.weight_dtype = torch.bfloat16
        self.cache_dir = os.path.join(os.getcwd(), "model_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.repo_path = os.path.join(os.path.dirname(__file__), 'models')
        self.pipeline = None
        self.dwprocessor = None
        self.densepose = None
        self.atr_model = None
        self.lip_model = None
        self.parsing_model = None
        self.goliath_model = None
        
        logger.info(f"TryOnInferenceEngine initialized in {time.time() - start_time:.2f} seconds")

    def initialize_model(self):
        if self.pipeline is not None:
            return
            
        try:
            logger.info("Loading models...")
            start_time = time.time()
            
            # Try to load from cache first
            cache_path = os.path.join(self.cache_dir, "model_cache.pt")
            if os.path.exists(cache_path):
                logger.info("Loading model from cache...")
                try:
                    cached_state = torch.load(cache_path, map_location=self.device)
                    self.pipeline, self.dwprocessor, self.densepose, self.atr_model, self.lip_model, self.parsing_model, self.goliath_model = cached_state
                    logger.info("Successfully loaded models from cache")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load from cache: {str(e)}. Loading fresh models...")
            
            # Load models normally if cache doesn't exist or fails
            self._load_models()
            
            # Save to cache after successful load
            logger.info("Saving models to cache...")
            try:
                torch.save(
                    (self.pipeline, self.dwprocessor, self.densepose, self.atr_model, self.lip_model, self.parsing_model, self.goliath_model),
                    cache_path
                )
                logger.info("Successfully saved models to cache")
            except Exception as e:
                logger.warning(f"Failed to save models to cache: {str(e)}")
                
            logger.info(f"Models loaded and initialized in {time.time() - start_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise Exception(f"Failed to initialize model: {str(e)}")

    def _load_models(self):
        transformer_garm = SD3Transformer2DModel_Garm.from_pretrained(
            os.path.join(self.repo_path, "transformer_garm"), 
            torch_dtype=self.weight_dtype
        )
        transformer_vton = SD3Transformer2DModel_Vton.from_pretrained(
            os.path.join(self.repo_path, "transformer_vton"), 
            torch_dtype=self.weight_dtype
        )
        
        pose_guider = PoseGuider(
            conditioning_embedding_channels=1536,
            conditioning_channels=3,
            block_out_channels=(32, 64, 256, 512)
        )
        pose_guider.load_state_dict(
            torch.load(os.path.join(self.repo_path, "pose_guider", "diffusion_pytorch_model.bin"))
        )
        
        image_encoder_large = CLIPVisionModelWithProjection.from_pretrained(
            "openai/clip-vit-large-patch14", 
            torch_dtype=self.weight_dtype
        )
        image_encoder_bigG = CLIPVisionModelWithProjection.from_pretrained(
            "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", 
            torch_dtype=self.weight_dtype
        )
        
        pose_guider.to(device=self.device, dtype=self.weight_dtype)
        image_encoder_large.to(device=self.device)
        image_encoder_bigG.to(device=self.device)
        
        # Initialize Goliath model
        goliath_model = torch.jit.load(
            hf_hub_download(
                repo_id="Roopansh/Ailusion-Goliath-Segmentation",
                filename="sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2",
                cache_dir="pretrained"
            )
        ).to(self.device)
        
        self.pipeline = StableDiffusion3TryOnPipeline.from_pretrained(
            self.repo_path,
            torch_dtype=self.weight_dtype,
            transformer_garm=transformer_garm,
            transformer_vton=transformer_vton,
            pose_guider=pose_guider,
            image_encoder_large=image_encoder_large,
            image_encoder_bigG=image_encoder_bigG
        )
        self.pipeline.to(self.device)
        
        self.dwprocessor = DWposeDetector(model_root=self.repo_path, device=self.device)
        self.densepose = DensePose(os.path.join(os.getcwd(), 'pretrained'), device="cuda")
        self.atr_model = SCHP(ckpt_path=os.path.join(os.getcwd(), 'pretrained/exp-schp-201908301523-atr.pth'), device="cuda")
        self.lip_model = SCHP(ckpt_path=os.path.join(os.getcwd(), 'pretrained/exp-schp-201908261155-lip.pth'), device="cuda")
        self.parsing_model = Parsing(model_root=self.repo_path, device=self.device)
        self.goliath_model = goliath_model
        
        return self.pipeline, self.dwprocessor, self.densepose, self.atr_model, self.lip_model, self.parsing_model, self.goliath_model

    @staticmethod
    def pad_and_resize(im, new_width=768, new_height=1024, pad_color=(255, 255, 255), mode=Image.LANCZOS):
        old_width, old_height = im.size
        
        ratio_w = new_width / old_width
        ratio_h = new_height / old_height
        if ratio_w < ratio_h:
            new_size = (new_width, round(old_height * ratio_w))
        else:
            new_size = (round(old_width * ratio_h), new_height)
        
        im_resized = im.resize(new_size, mode)
        pad_w = math.ceil((new_width - im_resized.width) / 2)
        pad_h = math.ceil((new_height - im_resized.height) / 2)
        new_im = Image.new('RGB', (new_width, new_height), pad_color)
        new_im.paste(im_resized, (pad_w, pad_h))
        
        return new_im, pad_w, pad_h

    @staticmethod
    def unpad_and_resize(padded_im, pad_w, pad_h, original_width, original_height):
        width, height = padded_im.size
        cropped_im = padded_im.crop((pad_w, pad_h, width - pad_w, height - pad_h))
        return cropped_im.resize((original_width, original_height), Image.LANCZOS)

    @staticmethod
    def resize_image(img, target_size=768):
        width, height = img.size
        scale = target_size / width if width < height else target_size / height
        new_width = int(round(width * scale))
        new_height = int(round(height * scale))
        return img.resize((new_width, new_height), Image.LANCZOS)

    @torch.inference_mode()
    def process_images(self, vton_img: Image.Image, garm_img: Image.Image, 
                      category: str = "Upper-body",
                      n_steps: int = 30,
                      image_scale: float = 2.0,
                      seed: int = -1,
                      num_images_per_prompt: int = 1,
                      resolution: str = "768x1024") -> List[Image.Image]:
        
        start_time = time.time()
        logger.info("Starting image processing...")
        
        try:
            # Parse resolution
            new_width, new_height = map(int, resolution.split("x"))
            
            # Save temporary image for mask generation
            temp_id = f"{int(time.time())%10000:04d}{''.join(random.choices('0123456789', k=4))}"
            temp_path = os.path.join("model", f"temp_input_{temp_id}.jpg")
            
            try:
                # Generate Mask using get_mask_location like app_copy.py
                vton_img_det = self.resize_image(vton_img)
                pose_image, keypoints, _, candidate = self.dwprocessor(np.array(vton_img_det)[:,:,::-1])
                
                # Process candidate data
                candidate[candidate < 0] = 0
                candidate = candidate[0]
                candidate[:, 0] *= vton_img_det.width
                candidate[:, 1] *= vton_img_det.height

                # Get parsing model output
                model_parse, _ = self.parsing_model(vton_img_det)

                # Generate mask using get_mask_location
                mask, mask_gray = get_mask_location(
                    category, 
                    model_parse,
                    candidate, 
                    model_parse.width, 
                    model_parse.height,
                    offset_top=0, 
                    offset_bottom=0, 
                    offset_left=0, 
                    offset_right=0
                )

                # Convert mask to numpy array, modify it, then convert back to PIL Image
                mask_array = np.array(mask)
                mask_array[mask_array == 10] = 0  # Remove right hand
                mask_array[mask_array == 11] = 0  # Remove left hand
                mask = Image.fromarray(mask_array)

                logger.info(f"Mask generated in {time.time() - start_time:.2f} seconds")
            
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            # Process images
            model_image_size = vton_img.size
            garm_img, _, _ = self.pad_and_resize(garm_img, new_width, new_height)
            vton_img, pad_w, pad_h = self.pad_and_resize(vton_img, new_width, new_height)
            mask = mask.resize(vton_img.size).convert("L")
            
            # Process pose image
            pose_image = Image.fromarray(pose_image[:,:,::-1])  # Convert BGR to RGB
            pose_image, _, _ = self.pad_and_resize(pose_image, new_width, new_height, pad_color=(0,0,0))
            
            if seed == -1:
                seed = random.randint(0, 2147483647)
                
            # View the mask for debugging
            logger.info(f"Mask shape: {np.array(mask).shape}")
            logger.info(f"Mask unique values: {np.unique(np.array(mask))}")
            
            # Generate images
            results = self.pipeline(
                height=new_height,
                width=new_width,
                guidance_scale=image_scale,
                num_inference_steps=n_steps,
                generator=torch.Generator("cpu").manual_seed(seed),
                cloth_image=garm_img,
                model_image=vton_img,
                mask=mask,
                pose_image=pose_image,
                num_images_per_prompt=num_images_per_prompt
            ).images
            
            # Post-process results
            for idx in range(len(results)):
                results[idx] = self.unpad_and_resize(
                    results[idx], pad_w, pad_h,
                    model_image_size[0], model_image_size[1]
                )
                
            logger.info(f"Image processing completed in {time.time() - start_time:.2f} seconds")
            return results
            
        except Exception as e:
            logger.error(f"Failed during processing: {str(e)}")
            raise RuntimeError(f"Failed during processing: {str(e)}")

app = Flask(__name__)
CORS(app)

logger.info("Creating TryOnInferenceEngine instance...")
engine = TryOnInferenceEngine()
logger.info("Initializing model...")
engine.initialize_model()

@app.route("/process_images", methods=['POST'])
def process_images():
    start_time = time.time()
    logger.info("Received process_images request")
    
    if 'vton_image' not in request.files or 'garm_image' not in request.files:
        logger.error("Missing required files in request")
        return jsonify({"error": "Missing required files"}), 400

    try:
        # Get parameters from request
        category = request.form.get('category', "Upper-body")
        n_steps = int(request.form.get('n_steps', 20))
        image_scale = float(request.form.get('image_scale', 2.0))
        seed = int(request.form.get('seed', -1))
        num_images = int(request.form.get('num_images', 1))
        resolution = request.form.get('resolution', "768x1024")
        
        # Process images
        vton_img = Image.open(request.files['vton_image']).convert('RGB')
        garm_img = Image.open(request.files['garm_image']).convert('RGB')
        
        results = engine.process_images(
            vton_img, garm_img,
            category=category,
            n_steps=n_steps,
            image_scale=image_scale,
            seed=seed,
            num_images_per_prompt=num_images,
            resolution=resolution
        )
        
        # Return first image (can be modified to return multiple images if needed)
        img_io = io.BytesIO()
        results[0].save(img_io, format='JPEG')
        img_io.seek(0)
        
        logger.info(f"Request processed successfully in {time.time() - start_time:.2f} seconds")
        return send_file(img_io, mimetype='image/jpeg')

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500

def convert_numpy_types(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    else:
        return obj

@app.route('/check-front-image', methods=['POST'])
def check_front_image():
    start_time = time.time()
    logger.info("Received check-front-image request")
    
    if 'human_image' not in request.files:
        logger.error("Missing required file in request")
        return jsonify({'error': 'Missing required file'}), 400

    try:
        cloth_image = request.files['human_image']
        cloth_image_path = 'FrontCheckImage/cloth.jpg'
        cloth_image.save(cloth_image_path)
        result = analyze_image(cloth_image_path)
        logger.info(f"Front image check completed in {time.time() - start_time:.2f} seconds")
        return jsonify(convert_numpy_types(result))
    except Exception as e:
        logger.error(f"Error checking front image: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route("/health")
def health_check():
    start_time = time.time()
    logger.info("Received health check request")
    
    gpu_metrics = {}
    if torch.cuda.is_available():
        try:
            current_gpu = torch.cuda.current_device()
            gpu_props = torch.cuda.get_device_properties(current_gpu)
            gpu_metrics = {
                "total_memory": f"{gpu_props.total_memory / (1024**3):.2f} GB",
                "memory_allocated": f"{torch.cuda.memory_allocated(current_gpu) / (1024**3):.2f} GB",
                "memory_utilization": f"{(torch.cuda.memory_allocated(current_gpu) / gpu_props.total_memory) * 100:.1f}%"
            }
        except Exception as e:
            logger.error(f"Error getting GPU metrics: {str(e)}")
            gpu_metrics = {"error": str(e)}

    response = {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "gpu_metrics": gpu_metrics,
        "model_loaded": engine.pipeline is not None
    }
    
    logger.info(f"Health check completed in {time.time() - start_time:.2f} seconds")
    return jsonify(response)

if __name__ == "__main__":
    logger.info("Starting Flask application...")
    app.run(debug=True) 