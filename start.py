import torch
from waitress import serve
from server import TryOnInferenceEngine, app
from config import get_settings
from flask import Flask, request
import os

def get_available_gpu_memory():
    """Get available memory for each GPU in GB"""
    available_gpus = {}
    for i in range(torch.cuda.device_count()):
        total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # Convert to GB
        memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
        available_memory = total_memory - memory_allocated
        available_gpus[i] = available_memory
    return available_gpus

def create_engines():
    """Create multiple engines based on available GPU memory"""
    available_gpus = get_available_gpu_memory()
    engines = []
    
    for gpu_id, available_memory in available_gpus.items():
        num_engines = int(available_memory // 20)  # Each engine takes 2GB
        for _ in range(num_engines):
            engine = TryOnInferenceEngine()
            with torch.cuda.device(gpu_id):
                engine.initialize_model()
            engines.append(engine)
    
    return engines

if __name__ == "__main__":
    settings = get_settings()
    
    # Create multiple engines
    engines = create_engines()
    if not engines:
        raise RuntimeError("No GPU with sufficient memory found")
    
    # Add engines to app context
    app.config['engines'] = engines
    app.config['current_engine'] = 0
    
    serve(
        app,
        host="0.0.0.0", 
        port=8002,
        threads=len(engines) * len(engines),  # Scale threads with number of engines
        connection_limit=1000,
        channel_timeout=300,
        ident="TryOn Server"
    )
