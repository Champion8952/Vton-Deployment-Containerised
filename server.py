from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
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
from config import get_settings
from utils import pil_to_binary_mask 
from cloth_masker_2 import visualize_dense_labels
from detectron2.data.detection_utils import convert_PIL_to_numpy,_apply_exif_orientation
import apply_net

class TryOnInferenceEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = transforms.Compose([
            transforms.Resize((1024, 768)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    async def initialize_model(self):
        """Initialize the model if not already loaded"""
        if self.model is not None:
            return
        try:
            self.model = self._load_models()
        except Exception as e:
            raise Exception(f"Model initialization failed: {str(e)}")

    def _load_models(self):
        """Load all required model components"""
        from diffusers import AutoencoderKL, DDPMScheduler
        from src.unet_hacked_tryon import UNet2DConditionModel
        from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
        from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline

        MODEL_PATH = get_settings().MODEL_PATH

        # Load model components
        components = {
            "scheduler": DDPMScheduler.from_pretrained(MODEL_PATH, subfolder="scheduler", cache_dir="downloaded_models", torch_dtype=torch.float32),
            "vae": AutoencoderKL.from_pretrained(MODEL_PATH, subfolder="vae", cache_dir="downloaded_models", torch_dtype=torch.float32),
            "unet": UNet2DConditionModel.from_pretrained(MODEL_PATH, subfolder="unet", cache_dir="downloaded_models", torch_dtype=torch.float32),
            "image_encoder": CLIPVisionModelWithProjection.from_pretrained(MODEL_PATH, subfolder="image_encoder", cache_dir="downloaded_models", torch_dtype=torch.float32),
            "unet_encoder": UNet2DConditionModel_ref.from_pretrained(MODEL_PATH, subfolder="unet_encoder", cache_dir="downloaded_models", torch_dtype=torch.float32),
            "text_encoder": CLIPTextModel.from_pretrained(MODEL_PATH, subfolder="text_encoder", cache_dir="downloaded_models", torch_dtype=torch.float32),
            "text_encoder_2": CLIPTextModelWithProjection.from_pretrained(MODEL_PATH, subfolder="text_encoder_2", cache_dir="downloaded_models", torch_dtype=torch.float32),
            "tokenizer": AutoTokenizer.from_pretrained(MODEL_PATH, subfolder="tokenizer", use_fast=False, cache_dir="downloaded_models"),
            "tokenizer_2": AutoTokenizer.from_pretrained(MODEL_PATH, subfolder="tokenizer_2", use_fast=False, cache_dir="downloaded_models")
        }

        # Create pipeline
        pipe = TryonPipeline.from_pretrained(
            MODEL_PATH,
            unet=components["unet"],
            vae=components["vae"],
            feature_extractor=CLIPImageProcessor(),
            text_encoder=components["text_encoder"],
            text_encoder_2=components["text_encoder_2"],
            tokenizer=components["tokenizer"],
            tokenizer_2=components["tokenizer_2"],
            scheduler=components["scheduler"],
            image_encoder=components["image_encoder"],
            unet_encoder=components["unet_encoder"],
            torch_dtype=torch.float32,
        ).to(self.device)

        # Enable optimizations
        if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_vae_slicing()

        return pipe

    def _process_human_image(self, human_img_orig: Image.Image) -> Image.Image:
        """Process human image with padding and resizing"""
        
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

        return padded_img.resize((768, 1024))

    def _generate_mask(self, human_img: Image.Image) -> Image.Image:
        """Generate mask for the human image"""
        temp_id = f"{int(time.time())%10000:04d}{''.join(random.choices('0123456789', k=4))}"
        temp_path = os.path.join("model", f"temp_input_{temp_id}.jpg")
        
        try:
            human_img.save(temp_path)
            mask = visualize_dense_labels(temp_path)
            mask_pil = Image.fromarray(mask)
            return mask_pil.resize((768, 1024))
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
       
    async def process_images(self, person_image: Image.Image, cloth_image: Image.Image, 
                            garment_des: str = "", 
                           denoise_steps: int = 30) -> Image.Image:

        """Process person and cloth images to generate try-on result"""
        if not self.model:
            raise ValueError("Model not initialized")
        
        # Process images with EXIF orientation preserved
        cloth_image = ImageOps.exif_transpose(cloth_image.convert("RGB")).resize((768, 1024))
        human_img_orig = ImageOps.exif_transpose(person_image.convert("RGB"))
        
        human_img = self._process_human_image(human_img_orig)
        mask = self._generate_mask(human_img)

        # Generate pose image
        human_img_arg = convert_PIL_to_numpy(
            ImageOps.exif_transpose(human_img.resize((384, 512))), 
            format="BGR"
        )
        
        args = apply_net.create_argument_parser().parse_args((
            'show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml', 
            './ckpt/densepose/model_final_162be9.pkl', 'dp_segm', 
            '-v', '--opts', 'MODEL.DEVICE', 'cuda'
        ))
        
        pose_img = args.func(args, human_img_arg)
        pose_img = pose_img[:,:,::-1]
        pose_img = Image.fromarray(pose_img).resize((768, 1024))

        # Prepare prompts
        prompt = f"This garment is a T shirt {garment_des}"
        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        # Generate embeddings
        with torch.inference_mode():
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = \
                self.model.encode_prompt(
                    prompt,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                    negative_prompt=negative_prompt,
                )

            prompt_embeds_c, _, _, _ = self.model.encode_prompt(
                [prompt] if not isinstance(prompt, List) else prompt,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
                negative_prompt=[negative_prompt] if not isinstance(negative_prompt, List) else negative_prompt,
            )

        # Prepare tensors
        pose_img = self.transform(pose_img).unsqueeze(0).to(self.device, torch.float32)
        garm_tensor = self.transform(cloth_image).unsqueeze(0).to(self.device, torch.float32)

        # Generate image
        with torch.cuda.amp.autocast(), torch.no_grad():
            images = self.model(
                prompt_embeds=prompt_embeds.to(self.device, torch.float32),
                negative_prompt_embeds=negative_prompt_embeds.to(self.device, torch.float32),
                pooled_prompt_embeds=pooled_prompt_embeds.to(self.device, torch.float32),
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(self.device, torch.float32),
                num_inference_steps=denoise_steps,
                generator=torch.Generator(self.device).manual_seed(42),
                strength=1.0,
                pose_img=pose_img,
                text_embeds_cloth=prompt_embeds_c.to(self.device, torch.float32),
                cloth=garm_tensor,
                mask_image=mask,
                image=human_img,
                height=1024,
                width=768,
                ip_adapter_image=cloth_image.resize((768, 1024)),
                guidance_scale=2.0,
            )[0]

            return images[0]


# FastAPI app setup
app = FastAPI(title="Virtual Try-On API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engine
engine = TryOnInferenceEngine()

@app.on_event("startup")
async def startup_event():
    await engine.initialize_model()

@app.post("/try_on")
async def try_on(request: Request):
    """Handle try-on requests"""
    form = await request.form()
    
    if 'cloth_image' not in form or 'human_image' not in form:
        raise HTTPException(status_code=400, detail="Missing images in request")

    # Create model directory
    os.makedirs("model", exist_ok=True)

    # Generate unique temp file names
    temp_id = f"{int(time.time())%10000:04d}{''.join(random.choices('0123456789', k=4))}"
    temp_files = {
        'cloth': os.path.join("model", f"cloth_image{temp_id}.jpg"),
        'human': os.path.join("model", f"human_image{temp_id}.jpg"),
        'output': os.path.join("model", f"output_image{temp_id}.jpg")
    }

    try:
        # Save uploaded files
        cloth_contents = await form['cloth_image'].read()
        human_contents = await form['human_image'].read()
        
        for path, contents in [
            (temp_files['cloth'], cloth_contents),
            (temp_files['human'], human_contents)
        ]:
            with open(path, 'wb') as f:
                f.write(contents)

        # Process images
        cloth_img = Image.open(temp_files['cloth']).convert('RGB')
        human_img = Image.open(temp_files['human']).convert('RGB')
        
        # Generate result
        result = await engine.process_images(human_img, cloth_img)
        
        # Save and return result
        result.save(temp_files['output'], format='JPEG')
        with open(temp_files['output'], 'rb') as f:
            return Response(content=f.read(), media_type='image/jpeg')

    finally:
        # Cleanup
        for path in temp_files.values():
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass

@app.get("/health")
async def health_check():
    """Check API health status"""
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "gpu_device": str(engine.device)
    }