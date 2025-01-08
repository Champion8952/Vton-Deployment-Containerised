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
from config import get_settings
from cloth_masker import visualize_dense_labels
from detectron2.data.detection_utils import convert_PIL_to_numpy
import apply_net
from model.SCHP import SCHP
from model.DensePose import DensePose
import asyncio
import platform

# Conditionally import Triton only on Linux
if platform.system() == "Linux":
    import triton
    import triton.language as tl
    from torch.utils.cpp_extension import CUDA_HOME

class TryOnInferenceEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Check OS and Triton availability
        self.use_triton = (
            platform.system() == "Linux" and  # Check if OS is Linux
            CUDA_HOME is not None and 
            torch.cuda.is_available()
        )
        
        if self.use_triton:
            try:
                import triton
                print(f"Running on Linux with Triton {triton.__version__} enabled")
            except ImportError:
                self.use_triton = False
                print("Triton not available, falling back to default processing")
        else:
            print(f"Running on {platform.system()}, Triton optimizations disabled")

        self.model = None
        self.densepose = None
        self.atr_model = None
        self.lip_model = None
        self.densepose_predictor = None
        self.densepose_args = None
        self.transform = transforms.Compose([
            transforms.Resize((1024, 768)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def initialize_model(self):
        """Initialize the model if not already loaded"""
        if self.model is not None:
            return
        try:
            self.model, self.densepose, self.atr_model, self.lip_model = self._load_models()
            
            # Apply Triton optimizations if available
            if self.use_triton:
                try:
                    # Use inductor backend with Triton acceleration
                    self.model.unet = torch.compile(
                        self.model.unet, 
                        backend='inductor',
                        mode="max-autotune",
                        options={"triton.autotune": True}
                    )
                    self.model.vae = torch.compile(
                        self.model.vae, 
                        backend='inductor',
                        mode="max-autotune",
                        options={"triton.autotune": True}
                    )
                    print("Models compiled with Triton-enabled inductor backend")
                except Exception as compile_error:
                    print(f"Model compilation failed, falling back to default: {str(compile_error)}")
                    self.use_triton = False
            
            self.densepose_args = apply_net.create_argument_parser().parse_args((
                'show', './pretrained/densepose_rcnn_R_50_FPN_s1x.yaml', 
                './pretrained/model_final_162be9.pkl', 'dp_segm', 
                '-v', '--opts', 'MODEL.DEVICE', 'cuda'
            ))
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
            torch_dtype=torch.float16,
        ).to(self.device)
        
        densepose_ckpt = "./pretrained"
        schp_ckpt = "./pretrained"
        
        # Create DensePose predictor
        densepose = DensePose(densepose_ckpt, device="cuda")
        # SCHP Predictor
        atr_model = SCHP(ckpt_path=os.path.join(schp_ckpt, 'exp-schp-201908301523-atr.pth'), device="cuda")
        lip_model = SCHP(ckpt_path=os.path.join(schp_ckpt, 'exp-schp-201908261155-lip.pth'), device="cuda")

        return pipe, densepose, atr_model, lip_model

    async def _process_human_image(self, human_img_orig: Image.Image) -> Image.Image:
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

    async def _generate_mask(self, human_img: Image.Image) -> Image.Image:
        """Generate mask for the human image"""
        temp_id = f"{int(time.time())%10000:04d}{''.join(random.choices('0123456789', k=4))}"
        temp_path = os.path.join("model", f"temp_input_{temp_id}.jpg")
        
        try:
            human_img.save(temp_path)
            mask = visualize_dense_labels(temp_path, self.densepose, self.atr_model, self.lip_model)
            mask_pil = Image.fromarray(mask)
            return mask_pil.resize((768, 1024))
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
       
    async def process_images(self, person_image: Image.Image, cloth_image: Image.Image, garment_des: str = "",  denoise_steps: int = 20) -> Image.Image:
        """Process person and cloth images to generate try-on result"""
        if not self.model:
            raise ValueError("Model not initialized")
        
        # Move image processing to GPU where possible
        cloth_image = ImageOps.exif_transpose(cloth_image.convert("RGB"))
        human_img_orig = ImageOps.exif_transpose(person_image.convert("RGB"))
        
        human_img = await self._process_human_image(human_img_orig)
        mask = await self._generate_mask(human_img)
        cloth_image = await self._process_human_image(cloth_image)

        # Use pre-initialized predictor for DensePose
        human_img_arg = convert_PIL_to_numpy(
            ImageOps.exif_transpose(human_img.resize((768, 1024))), 
            format="BGR"
        )
        
        pose_img = self.densepose_args.func(self.densepose_args, human_img_arg)
        pose_img = pose_img[:,:,::-1]
        pose_img = Image.fromarray(pose_img).resize((768, 1024))

        # Prepare prompts
        prompt = f"This garment is a T shirt {garment_des}"
        cloth_prompt = f"A T shirt {garment_des}"  # Added cloth-specific prompt
        negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        # Generate embeddings with Triton optimization if available
        with torch.inference_mode():
            if self.use_triton:
                prompt_embeds_fn = torch.compile(self.model.encode_prompt, backend="triton")
                # Main prompt embeddings
                prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = \
                    prompt_embeds_fn(
                        prompt,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=True,
                        negative_prompt=negative_prompt,
                    )
                # Cloth prompt embeddings
                prompt_embeds_c, _, _, _ = prompt_embeds_fn(
                    [prompt] if not isinstance(prompt, List) else prompt,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                    negative_prompt=[negative_prompt] if not isinstance(negative_prompt, List) else negative_prompt,
                )
            else:
                # Main prompt embeddings
                prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = \
                    self.model.encode_prompt(
                        prompt,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=True,
                        negative_prompt=negative_prompt,
                    )
                # Cloth prompt embeddings
                prompt_embeds_c, _, _, _ = self.model.encode_prompt(
                    [prompt] if not isinstance(prompt, List) else prompt,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                    negative_prompt=[negative_prompt] if not isinstance(negative_prompt, List) else negative_prompt,
                )

        # Prepare tensors
        pose_img = self.transform(pose_img).unsqueeze(0).to(self.device, torch.float16)
        garm_tensor = self.transform(cloth_image).unsqueeze(0).to(self.device, torch.float16)

        # Generate image with Triton optimizations if available
        with torch.cuda.amp.autocast(), torch.no_grad():
            if self.use_triton:
                # Compile the forward pass with Triton
                model_fn = torch.compile(
                    self.model,
                    backend="triton",
                    mode="max-autotune"
                )
                images = model_fn(
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
            else:
                # Original non-Triton processing
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

            return images[0]

app = Flask(__name__)
CORS(app)

# Initialize engine
engine = TryOnInferenceEngine()
engine.initialize_model()

@app.route("/try_on", methods=['POST'])
def try_on():
    """Handle try-on requests"""
    if engine.model is None:
        engine.initialize_model()

    if 'cloth_image' not in request.files or 'human_image' not in request.files:
        return jsonify({"error": "Missing images in request"}), 400

    try:
        cloth_img = Image.open(request.files['cloth_image']).convert('RGB')
        human_img = Image.open(request.files['human_image']).convert('RGB')
        
        # Run async function in event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(engine.process_images(human_img, cloth_img))
        loop.close()
        
        # Convert PIL Image to bytes
        import io
        img_io = io.BytesIO()
        result.save(img_io, format='JPEG')
        img_io.seek(0)
        
        return send_file(img_io, mimetype='image/jpeg')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health_check():
    """Check API health status"""
    return jsonify({
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "gpu_device": str(engine.device)
    })

if __name__ == "__main__":
    app.run()
