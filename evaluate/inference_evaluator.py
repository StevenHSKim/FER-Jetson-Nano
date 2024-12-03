import time
import numpy as np
import torch
from typing import Optional, Dict

class InferenceEvaluator:
    def __init__(self, warm_up: int = 5):
        self.warm_up = warm_up
        self.timing_results = {}
        
    def measure_inference_time(self, 
                             model: torch.nn.Module,
                             input_tensor: torch.Tensor,
                             model_name: str,
                             device: Optional[torch.device] = None,
                             num_runs: int = 100) -> Dict[str, float]:

        # Get model's device if not specified
        if device is None:
            device = next(model.parameters()).device
            
        # Ensure model and input are on the same device
        model = model.to(device)
        input_tensor = input_tensor.to(device)
        model.eval()

        # Clear CUDA cache before starting
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Warm up
        print("\nWarming up {}...".format(model_name))
        with torch.no_grad():
            for _ in range(self.warm_up):
                _ = model(input_tensor)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
        
        # Measure inference times
        print("Measuring inference time for {}...".format(model_name))
        inference_times = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                # Ensure cache is cleared
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                start_time = time.perf_counter()
                _ = model(input_tensor)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                inference_times.append((end_time - start_time) * 1000)  # Convert to milliseconds
        
        # Calculate statistics
        mean_time = np.mean(inference_times)
        std_time = np.std(inference_times)
        
        results = {
            'inference_time_mean_std': "{:.3f}±{:.3f}".format(mean_time, std_time)
        }
        
        # Store result
        self.timing_results[model_name] = results
        
        return results
    
    def print_results(self):
        """Print timing results for all evaluated models"""
        print("\nInference Time Results:")
        print("-" * 50)
        print("{:<20} {:<15}".format("Model", "Time (ms)"))
        print("-" * 50)
        
        for model_name, stats in self.timing_results.items():
            print("{:<20} {}".format(model_name, stats['inference_time_mean_std']))