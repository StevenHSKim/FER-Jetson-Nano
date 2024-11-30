import torch
from ptflops import get_model_complexity_info
import time

def calculate_parameters(model):
    """Calculate the number of parameters in the model."""
    return sum(p.numel() for p in model.parameters())

def calculate_flops(model, input_size=(3, 224, 224)):
    """Calculate FLOPs of the model."""
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model, input_size, as_strings=False, print_per_layer_stat=False)
    return macs

def measure_inference_time(model, dummy_input):
    """Measure inference time."""
    model.eval()
    with torch.no_grad():
        # Warm-up
        for _ in range(10):
            _ = model(dummy_input)
        # Timing
        start_time = time.time()
        for _ in range(100):
            _ = model(dummy_input)
        end_time = time.time()
    avg_inference_time = (end_time - start_time) / 100
    return avg_inference_time

if __name__ == "__main__":
    models = {
        "KD": "path_to_kd_model.pth",
        "KD + PR": "path_to_kd_pr_model.pth",
        "KD + Q": "path_to_kd_q_model.pth",
        "KD + PR + Q": "path_to_kd_pr_q_model.pth"
    }

    dummy_input = torch.randn(1, 3, 224, 224).cuda()

    for name, model_path in models.items():
        model = torch.load(model_path).cuda()
        params = calculate_parameters(model)
        flops = calculate_flops(model)
        inference_time = measure_inference_time(model, dummy_input)
        
        print(f"Model: {name}")
        print(f"  Parameters: {params}")
        print(f"  FLOPs: {flops}")
        print(f"  Inference Time: {inference_time:.4f} seconds")
