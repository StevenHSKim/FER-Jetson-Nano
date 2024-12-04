import torch
import torch.nn as nn
from typing import Dict, Any
import pandas as pd

class ModelEvaluator:
    def __init__(self):
        self.evaluation_results = {}
        
    def count_parameters(self, model: nn.Module) -> Dict[str, int]:
        """
        Count total and trainable parameters in the model
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        results = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params
        }
        
        return results

    def calculate_flops(self, model: nn.Module, input_shape: tuple) -> Dict[str, Any]:
        """
        Calculate FLOPs for the model
        Args:
            model: PyTorch model
            input_shape: Input shape (B, C, H, W)
        """
        def hook_fn(module, input, output):
            flops = 0
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                input = input[0]
                batch_size, in_channels, in_h, in_w = input.shape
                out_channels, out_h, out_w = output.shape[1:]
                kernel_h, kernel_w = module.kernel_size
                flops = (
                    batch_size * out_channels * in_channels * 
                    kernel_h * kernel_w * out_h * out_w / 
                    (module.groups or 1)
                )
            elif isinstance(module, nn.Linear):
                input = input[0]
                batch_size = input.shape[0]
                flops = batch_size * input.shape[1] * output.shape[1]
                
            module.__flops__ = flops
            
        hooks = []
        model.eval()
        
        # Register hooks for all modules
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                hooks.append(module.register_forward_hook(hook_fn))
        
        # Get the device of the model
        device = next(model.parameters()).device
                
        # Run model to trigger hooks
        dummy_input = torch.randn(input_shape).to(device)  # Move dummy input to same device as model
        with torch.no_grad():
            model(dummy_input)
            
        # Calculate total FLOPs
        total_flops = 0
        for module in model.modules():
            if hasattr(module, '__flops__'):
                total_flops += module.__flops__
                
        # Remove hooks
        for hook in hooks:
            hook.remove()
            
        # Clear temporary attributes
        for module in model.modules():
            if hasattr(module, '__flops__'):
                delattr(module, '__flops__')
                
        results = {
            'total_flops': total_flops,
            'total_gflops': total_flops / (1024**3),  # Convert to GFLOPs
        }
        
        return results
        
    def calculate_model_size(self, model: nn.Module) -> Dict[str, float]:
        """
        Calculate model size in MB
        Args:
            model: PyTorch model
        Returns:
            Dictionary containing model size information
        """
        param_size = 0
        buffer_size = 0

        # Calculate parameters size
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        # Calculate buffers size (e.g., running mean and variance in batch norm)
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_mb = (param_size + buffer_size) / (1024 * 1024)  # Convert to MB

        results = {
            'model_size_mb': round(size_mb, 3)
        }

        return results

    def evaluate_model(self, model: nn.Module, model_name: str, input_shape: tuple):
        """
        Evaluate model and store results
        """
        param_stats = self.count_parameters(model)
        flop_stats = self.calculate_flops(model, input_shape)
        size_stats = self.calculate_model_size(model)
        
        # Create results dictionary without dictionary unpacking
        self.evaluation_results[model_name] = {}
        
        # Update with all stats
        for d in [param_stats, flop_stats, size_stats]:
            self.evaluation_results[model_name].update(d)
        
        return self.evaluation_results[model_name]
    
    def print_results(self):
        """Print evaluation results"""
        print("\nModel Evaluation Results:")
        print("-" * 80)
        for model_name, stats in self.evaluation_results.items():
            print(f"\nModel: {model_name}")
            print(f"Total Parameters: {stats['total_parameters']:,}")
            print(f"Trainable Parameters: {stats['trainable_parameters']:,}")
            print(f"Non-trainable Parameters: {stats['non_trainable_parameters']:,}")
            print(f"Total FLOPs: {stats['total_flops']:,}")
            print(f"Total GFLOPs: {stats['total_gflops']:.2f}")
            
    def save_results(self, filepath: str):
        """Save results to CSV file"""
        df = pd.DataFrame.from_dict(self.evaluation_results, orient='index')
        df.to_csv(filepath)
        print(f"\nModel evaluation results saved to {filepath}")