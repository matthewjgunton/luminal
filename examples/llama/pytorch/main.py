import torch
import time
import threading
from transformers import pipeline
import psutil
import gc
import os

# Try to import GPUtil, but don't fail if it's not available (for MPS systems)
try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

class GPUMonitor:
    def __init__(self):
        self.max_utilization = 0
        self.monitoring = False
        self.thread = None
        self.device_type = self._detect_device()
    
    def _detect_device(self):
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def start(self):
        self.monitoring = True
        self.max_utilization = 0
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=1)
        return self.max_utilization
    
    def _monitor(self):
        while self.monitoring:
            try:
                if self.device_type == "cuda" and GPUTIL_AVAILABLE:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        current_util = gpus[0].load * 100
                        self.max_utilization = max(self.max_utilization, current_util)
                elif self.device_type == "mps":
                    # For MPS, we can't get real-time utilization easily
                    # We'll use memory allocation as a proxy
                    if torch.backends.mps.is_available():
                        allocated = torch.mps.current_allocated_memory()
                        if allocated > 0:
                            self.max_utilization = min(100, self.max_utilization + 1)
                # For CPU, we don't track GPU utilization
            except:
                pass
            time.sleep(0.1)

def benchmark_llama():
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    
    print("Starting Llama 3.2-3B Benchmark...")
    print("=" * 50)
    
    # Detect device
    if torch.cuda.is_available():
        device_info = f"CUDA GPU: {torch.cuda.get_device_name()}"
        torch_dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        device_info = "Apple MPS"
        torch_dtype = torch.float16  # MPS works better with float16
    else:
        device_info = "CPU"
        torch_dtype = torch.float32
    
    print(f"Device: {device_info}")
    print(f"Using dtype: {torch_dtype}")
    
    # Set environment variables for better stability
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # Initialize GPU monitor
    gpu_monitor = GPUMonitor()
    
    # Start timing and monitoring
    start_time = time.time()
    gpu_monitor.start()
    
    pipe = None
    try:
        # Cold start - model loading
        print("Loading model...")
        load_start = time.time()
        
        pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        
        load_time = time.time() - load_start
        print(f"Model loaded in {load_time:.2f} seconds")
        
        # Prepare messages
        messages = [
            {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
            {"role": "user", "content": "Who are you?"},
        ]
        
        # Time to first token + generation
        print("Generating response...")
        generation_start = time.time()
        
        outputs = pipe(
            messages,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            pad_token_id=pipe.tokenizer.eos_token_id
        )
        
        generation_time = time.time() - generation_start
        total_time = time.time() - start_time
        
        # Stop GPU monitoring
        max_gpu_util = gpu_monitor.stop()
        
        # Calculate metrics
        generated_text = outputs[0]["generated_text"][-1]["content"]
        
        # Count tokens (approximate using whitespace split)
        token_count = len(generated_text.split())
        
        # Calculate tokens per second
        toks_per_second = token_count / generation_time if generation_time > 0 else 0
        
        # Get memory usage info
        memory_info = get_memory_info(gpu_monitor.device_type)
        
        # Results
        print("\n" + "=" * 50)
        print("BENCHMARK RESULTS")
        print("=" * 50)
        print(f"Device: {device_info}")
        print(f"Time to First Token (Cold Start): {load_time:.2f}s")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Generation Time: {generation_time:.2f}s")
        print(f"Tokens Generated: {token_count}")
        print(f"Tokens/Second: {toks_per_second:.2f}")
        
        if gpu_monitor.device_type == "cuda":
            print(f"Max GPU Utilization: {max_gpu_util:.1f}%")
        elif gpu_monitor.device_type == "mps":
            print(f"MPS Memory Activity: {'Active' if max_gpu_util > 0 else 'Inactive'}")
        
        print(f"Memory Info: {memory_info}")
        print("=" * 50)
        
        print(f"\nGenerated Response:")
        print(f"'{generated_text}'")
        
    except Exception as e:
        print(f"Error during benchmarking: {e}")
        gpu_monitor.stop()
    
    finally:
        # Cleanup
        if pipe:
            del pipe
        
        # Clear cache based on device type
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        gc.collect()

def get_memory_info(device_type):
    """Get memory usage information based on device type"""
    try:
        if device_type == "cuda":
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            return f"CUDA: {allocated:.1f}GB allocated, {cached:.1f}GB cached"
        elif device_type == "mps":
            allocated = torch.mps.current_allocated_memory() / 1024**3
            return f"MPS: {allocated:.1f}GB allocated"
        else:
            return "CPU memory monitoring not implemented"
    except:
        return "Memory info unavailable"

if __name__ == "__main__":
    benchmark_llama()