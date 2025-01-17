import torch
from waitress import serve
from server import TryOnInferenceEngine, app
from config import get_settings
from flask import Flask, request
import os
import threading
from queue import Queue

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
        num_engines = int(available_memory // 65)  # Each engine takes 2GB
        for _ in range(num_engines):
            engine = TryOnInferenceEngine()
            with torch.cuda.device(gpu_id):
                engine.initialize_model()
            engines.append(engine)
    
    return engines

class EngineManager:
    def __init__(self, engines):
        self.engines = engines
        self.engine_queues = [Queue() for _ in engines]
        self.engine_threads = []
        self.initialize_workers()

    def initialize_workers(self):
        for i, engine in enumerate(self.engines):
            thread = threading.Thread(target=self.engine_worker, args=(i, engine))
            thread.daemon = True
            thread.start()
            self.engine_threads.append(thread)

    def engine_worker(self, engine_id, engine):
        while True:
            request_data, result_queue = self.engine_queues[engine_id].get()
            try:
                result = engine.process_request(request_data)
                result_queue.put((True, result))
            except Exception as e:
                result_queue.put((False, str(e)))

    def process_request(self, request_data):
        # Find queue with minimum length
        min_queue_id = min(range(len(self.engine_queues)), 
                          key=lambda i: self.engine_queues[i].qsize())
        
        result_queue = Queue()
        self.engine_queues[min_queue_id].put((request_data, result_queue))
        success, result = result_queue.get()
        
        if not success:
            raise RuntimeError(f"Engine processing failed: {result}")
        return result

if __name__ == "__main__":
    settings = get_settings()
    
    # Create multiple engines
    engines = create_engines()
    if not engines:
        raise RuntimeError("No GPU with sufficient memory found")
    
    # Initialize engine manager
    engine_manager = EngineManager(engines)
    app.config['engine_manager'] = engine_manager
    
    serve(
        app,
        host="0.0.0.0", 
        port=8002,
        threads=len(engines) * 2,  # 2 threads per engine
        connection_limit=1000,
        channel_timeout=300,
        ident="TryOn Server"
    )
