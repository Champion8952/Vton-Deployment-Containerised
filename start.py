import torch
from waitress import serve
from server import TryOnInferenceEngine, app
from config import get_settings
from flask import Flask, request
import os
import threading
from queue import Queue
import logging

logger = logging.getLogger(__name__)

def save_triton_cache(engine):
    """Save Triton optimization cache and settings"""
    cache_dir = os.path.join(os.getcwd(), "triton_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        # Save Triton kernel configurations
        kernel_config_path = os.path.join(cache_dir, "kernel_configs.pt")
        if hasattr(engine.model.unet, '_inductor_kernels'):
            torch.save({
                'unet_kernels': engine.model.unet._inductor_kernels,
                'vae_kernels': engine.model.vae._inductor_kernels,
                'compile_options': engine.model.unet._compile_options,
                'kernel_metadata': getattr(engine.model.unet, '_kernel_metadata', {}),
                'optimization_params': {
                    'max_autotune': True,
                    'layout_optimization': True,
                    'triton.autotune_pointwise': True,
                    'triton.autotune_cublasLt': True,
                    'triton.max_tiles': 2048,
                    'triton.persistent_reductions': True,
                    'triton.cudagraphs': True
                }
            }, kernel_config_path)
            
        # Save compiled cache
        if os.path.exists(os.environ.get('TORCH_COMPILE_CACHE_DIR', '')):
            import shutil
            compile_cache_dest = os.path.join(cache_dir, 'compile_cache')
            shutil.copytree(
                os.environ['TORCH_COMPILE_CACHE_DIR'],
                compile_cache_dest,
                dirs_exist_ok=True
            )
            
        logger.info(f"Triton cache saved to {cache_dir}")
        return True
    except Exception as e:
        logger.error(f"Failed to save Triton cache: {e}")
        return False

def load_triton_cache(engine):
    """Load Triton optimization cache and settings"""
    cache_dir = os.path.join(os.getcwd(), "triton_cache")
    
    try:
        # Load kernel configurations
        kernel_config_path = os.path.join(cache_dir, "kernel_configs.pt")
        if os.path.exists(kernel_config_path):
            cached_data = torch.load(kernel_config_path)
            
            if hasattr(engine.model.unet, '_inductor_kernels'):
                engine.model.unet._inductor_kernels = cached_data['unet_kernels']
                engine.model.vae._inductor_kernels = cached_data['vae_kernels']
                engine.model.unet._compile_options = cached_data['compile_options']
                setattr(engine.model.unet, '_kernel_metadata', 
                       cached_data.get('kernel_metadata', {}))
            
        # Load compiled cache
        compile_cache_src = os.path.join(cache_dir, 'compile_cache')
        if os.path.exists(compile_cache_src):
            os.environ['TORCH_COMPILE_CACHE_DIR'] = compile_cache_src
            
        logger.info(f"Triton cache loaded from {cache_dir}")
        return True
    except Exception as e:
        logger.error(f"Failed to load Triton cache: {e}")
        return False

def create_engine():
    """Create single engine with optimized settings"""
    # Optimize GPU settings if GPU is available
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
    # Create single engine
    engine = TryOnInferenceEngine()
    engine.initialize_model()
    
    # Try to load existing Triton cache
    if not load_triton_cache(engine):
        logger.info("No existing Triton cache found, will create new optimizations")
    
    return engine

class EngineManager:
    def __init__(self, engine):
        self.engine = engine
        self.request_queue = Queue()
        self.worker_thread = None
        self.initialize_worker()

    def initialize_worker(self):
        self.worker_thread = threading.Thread(target=self.engine_worker)
        self.worker_thread.daemon = True
        self.worker_thread.start()

    def engine_worker(self):
        while True:
            request_data, result_queue = self.request_queue.get()
            try:
                result = self.engine.process_request(request_data)
                result_queue.put((True, result))
            except Exception as e:
                result_queue.put((False, str(e)))

    def process_request(self, request_data):
        result_queue = Queue()
        self.request_queue.put((request_data, result_queue))
        success, result = result_queue.get()
        
        if not success:
            raise RuntimeError(f"Engine processing failed: {result}")
        return result

    def __del__(self):
        """Save Triton cache when engine manager is destroyed"""
        save_triton_cache(self.engine)

if __name__ == "__main__":
    settings = get_settings()
    
    # Create single engine
    engine = create_engine()
    
    # Initialize engine manager
    engine_manager = EngineManager(engine)
    app.config['engine_manager'] = engine_manager
    
    # Register shutdown handler to save Triton cache
    import atexit
    atexit.register(lambda: save_triton_cache(engine))
    
    serve(
        app,
        host="0.0.0.0", 
        port=8002,
        threads=4,  # Fixed number of threads for single engine
        connection_limit=1000,
        channel_timeout=300,
        ident="TryOn Server"
    )
