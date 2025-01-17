from flask import Flask, request, jsonify, send_file
from flask.wrappers import Response
from flask_cors import CORS
import torch
from PIL import Image, ImageOps
import os
import time
import random
from typing import List
from torchvision import transforms
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection, 
    CLIPTextModel,
    CLIPTextModelWithProjection,
    AutoTokenizer
)
import numpy as np
from check_front_image import analyze_image
from config import get_settings
# from cloth_masker_without_goliath import visualize_dense_labels
from cloth_masker_with_goliath import visualize_dense_labels
from utils import convert_PIL_to_numpy
import apply_net
from model.SCHP import SCHP
from model.DensePose import DensePose
import asyncio
import platform
import io
import logging
from huggingface_hub import hf_hub_download

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

if platform.system() == "Linux":
    import triton
    import triton.language as tl
    from torch.utils.cpp_extension import CUDA_HOME

class TryOnInferenceEngine:
    def __init__(self):
        logger.info("Initializing TryOnInferenceEngine")
        start_time = time.time()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_triton = (
            platform.system() == "Linux" and
            CUDA_HOME is not None and
            torch.cuda.is_available()
        )
        self.model = None
        self.densepose = None
        self.atr_model = None
        self.lip_model = None
        self.goliath_model = None
        self.densepose_args = None
        self.transform = transforms.Compose([
            transforms.Resize((1024, 768)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        logger.info(f"TryOnInferenceEngine initialized in {time.time() - start_time:.2f} seconds")

    def initialize_model(self):
        if self.model is not None:
            return
        try:
            logger.info("Loading models...")
            start_time = time.time()
            
            self.model, self.densepose, self.atr_model, self.lip_model, self.goliath_model = self._load_models()
            if self.use_triton:
                try:
                    logger.info("Compiling models with Triton...")
                    compile_options = {
                        "max_autotune": True,
                        "layout_optimization": True,
                        "triton.autotune_pointwise": True,
                        "triton.autotune_cublasLt": True,
                        "triton.max_tiles": 1024,
                        "triton.persistent_reductions": True
                    }
                    self.model.unet = torch.compile(self.model.unet, backend='inductor', options=compile_options)
                    self.model.vae = torch.compile(self.model.vae, backend='inductor', options=compile_options)
                except Exception as e:
                    logger.warning(f"Failed to compile with Triton: {str(e)}")
                    self.use_triton = False
            
            self.densepose_args = apply_net.create_argument_parser().parse_args((
                'show', './pretrained/densepose_rcnn_R_50_FPN_s1x.yaml', 
                './pretrained/model_final_162be9.pkl', 'dp_segm', 
                '-v', '--opts', 'MODEL.DEVICE', 'cuda'
            ))
            self.goliath_model.eval()
            
            logger.info(f"Models loaded and initialized in {time.time() - start_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise Exception(f"Failed to initialize model: {str(e)}")

    def _load_models(self):
        start_time = time.time()
        logger.info("Loading model components...")
        
        from diffusers import AutoencoderKL, DDPMScheduler
        from src.unet_hacked_tryon import UNet2DConditionModel
        from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
        from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline

        MODEL_PATH = get_settings().MODEL_PATH
        components = {
            "scheduler": DDPMScheduler.from_pretrained(MODEL_PATH, subfolder="scheduler", cache_dir="downloaded_models", torch_dtype=torch.float16),
            "vae": AutoencoderKL.from_pretrained(MODEL_PATH, subfolder="vae", cache_dir="downloaded_models", torch_dtype=torch.float16),
            "unet": UNet2DConditionModel.from_pretrained(MODEL_PATH, subfolder="unet", cache_dir="downloaded_models", torch_dtype=torch.float16),
            "image_encoder": CLIPVisionModelWithProjection.from_pretrained(MODEL_PATH, subfolder="image_encoder", cache_dir="downloaded_models", torch_dtype=torch.float16),
            "unet_encoder": UNet2DConditionModel_ref.from_pretrained(MODEL_PATH, subfolder="unet_encoder", cache_dir="downloaded_models", torch_dtype=torch.float16),
            "text_encoder": CLIPTextModel.from_pretrained(MODEL_PATH, subfolder="text_encoder", cache_dir="downloaded_models", torch_dtype=torch.float16),
            "text_encoder_2": CLIPTextModelWithProjection.from_pretrained(MODEL_PATH, subfolder="text_encoder_2", cache_dir="downloaded_models", torch_dtype=torch.float16),
            "tokenizer": AutoTokenizer.from_pretrained(MODEL_PATH, subfolder="tokenizer", use_fast=False, cache_dir="downloaded_models"),
            "tokenizer_2": AutoTokenizer.from_pretrained(MODEL_PATH, subfolder="tokenizer_2", use_fast=False, cache_dir="downloaded_models")
        }

        logger.info("Creating pipeline...")
        pipe = TryonPipeline.from_pretrained(
            MODEL_PATH,
            **components,
            feature_extractor=CLIPImageProcessor(),
            torch_dtype=torch.float16,
        ).to(self.device)
          
        logger.info("Loading additional models...")
        densepose = DensePose(os.path.join(os.getcwd(), 'pretrained'), device="cuda")
        atr_model = SCHP(ckpt_path=os.path.join(os.getcwd(), 'pretrained/exp-schp-201908301523-atr.pth'), device="cuda")
        lip_model = SCHP(ckpt_path=os.path.join(os.getcwd(), 'pretrained/exp-schp-201908261155-lip.pth'), device="cuda")
        goliath_model = torch.jit.load(
            hf_hub_download(
                repo_id="Roopansh/Ailusion-Goliath-Segmentation",
                filename="sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2",
                cache_dir="pretrained"
            )
        )

        logger.info(f"All models loaded in {time.time() - start_time:.2f} seconds")
        return pipe, densepose, atr_model, lip_model, goliath_model

    async def _process_human_image(self, human_img_orig: Image.Image) -> Image.Image:
        start_time = time.time()
        logger.info("Processing human image...")
        
        width, height = human_img_orig.size
        target_ratio = 3 / 4
        current_ratio = width / height

        if current_ratio > target_ratio:
            new_height = int(width / target_ratio)
            padding = (new_height - height) // 2
            padded_img = Image.new("RGB", (width, new_height), (255, 255, 255))
            padded_img.paste(human_img_orig, (0, padding))
        else:
            new_width = int(height * target_ratio)
            padding = (new_width - width) // 2
            padded_img = Image.new("RGB", (new_width, height), (255, 255, 255))
            padded_img.paste(human_img_orig, (padding, 0))

        result = padded_img.resize((768, 1024))
        logger.info(f"Human image processed in {time.time() - start_time:.2f} seconds")
        return result

    async def _generate_mask(self, human_img: Image.Image) -> Image.Image:
        start_time = time.time()
        logger.info("Generating mask...")
        
        temp_id = f"{int(time.time())%10000:04d}{''.join(random.choices('0123456789', k=4))}"
        temp_path = os.path.join("model", f"temp_input_{temp_id}.jpg")
        
        try:
            human_img.save(temp_path)
            # mask = segment_image(temp_path, self.densepose, self.atr_model, self.lip_model, self.goliath_model)
            mask = visualize_dense_labels(temp_path, self.densepose, self.atr_model, self.lip_model, self.goliath_model)
            # mask = visualize_dense_labels(temp_path, self.densepose, self.atr_model, self.lip_model)
            result = Image.fromarray(mask).resize((768, 1024))
            logger.info(f"Mask generated in {time.time() - start_time:.2f} seconds")
            return result
        except Exception as e:
            logger.error(f"Failed to generate mask: {str(e)}")
            raise RuntimeError(f"Failed to generate mask: {str(e)}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


    @torch.inference_mode()       
    async def process_images(self, person_image: Image.Image, cloth_image: Image.Image, 
                            garment_des: str = "", 
                            denoise_steps: int = 20) -> Image.Image:
        start_time = time.time()
        logger.info("Starting image processing...")
        
        if not self.model:
            logger.error("Model not initialized")
            raise ValueError("Model not initialized")
        
        try:
            cloth_image = ImageOps.exif_transpose(cloth_image.convert("RGB"))
            human_img_orig = ImageOps.exif_transpose(person_image.convert("RGB"))
            human_img = await self._process_human_image(human_img_orig)
            mask = await self._generate_mask(human_img)
            cloth_image = await self._process_human_image(cloth_image)
            human_img_arg = convert_PIL_to_numpy(ImageOps.exif_transpose(human_img.resize((768, 1024))), format="BGR")
            
            logger.info("Generating pose image...")
            pose_img = self.densepose_args.func(self.densepose_args, human_img_arg)
            pose_img = Image.fromarray(pose_img[:,:,::-1]).resize((768, 1024))

            prompt = f"This garment is a T shirt {garment_des}"
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

            logger.info("Encoding prompts...")
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = \
                self.model.encode_prompt(prompt, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=negative_prompt)
            prompt_embeds_c, _, _, _ = self.model.encode_prompt([prompt], num_images_per_prompt=1, do_classifier_free_guidance=False, negative_prompt=[negative_prompt])

            pose_img = self.transform(pose_img).unsqueeze(0).to(self.device, torch.float16)
            garm_tensor = self.transform(cloth_image).unsqueeze(0).to(self.device, torch.float16)

            logger.info("Running inference...")
            with torch.cuda.amp.autocast():
                images = self.model(
                    prompt_embeds=prompt_embeds.to(self.device, torch.float16),
                    negative_prompt_embeds=negative_prompt_embeds.to(self.device, torch.float16),
                    pooled_prompt_embeds=pooled_prompt_embeds.to(self.device, torch.float16),
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(self.device, torch.float16),
                    num_inference_steps=denoise_steps,
                    generator=torch.Generator(self.device).manual_seed(42),
                    strength=1.0,
                    pose_img=pose_img,
                    text_embeds_cloth=prompt_embeds_c.to(self.device, torch.float16),
                    cloth=garm_tensor,
                    mask_image=mask,
                    image=human_img,
                    height=1024,
                    width=768,
                    ip_adapter_image=cloth_image,
                    guidance_scale=2.0,
                    use_compile=True
                )[0]

            logger.info("Cleaning up memory...")
            del pose_img, garm_tensor
            # torch.cuda.empty_cache()
            # gc.collect()
            
            logger.info(f"Image processing completed in {time.time() - start_time:.2f} seconds")
            return images[0]

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
    
    if engine.model is None:
        try:
            logger.info("Model not initialized, initializing now...")
            engine.initialize_model()
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            return jsonify({"error": f"Failed to initialize model: {str(e)}"}), 500

    if 'cloth_image' not in request.files or 'human_image' not in request.files:
        logger.error("Missing required files in request")
        return jsonify({"error": "Missing required files"}), 400

    try:
        logger.info("Processing images...")
        cloth_img = Image.open(request.files['cloth_image']).convert('RGB')
        human_img = Image.open(request.files['human_image']).convert('RGB')
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(engine.process_images(human_img, cloth_img))
        loop.close()
        
        img_io = io.BytesIO()
        result.save(img_io, format='JPEG')
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
        "model_loaded": engine.model is not None
    }
    
    logger.info(f"Health check completed in {time.time() - start_time:.2f} seconds")
    return jsonify(response)

if __name__ == "__main__":
    logger.info("Starting Flask application...")
    app.run(debug=True)
