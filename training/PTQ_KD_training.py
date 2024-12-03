import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.KD_training import StudentModel
from data.ckplus_dataset import CKPlusDataset, split_dataset

class WeightQuantizer:
    def __init__(self, num_bits):
        self.num_bits = num_bits
        # 32비트는 float32 그대로 사용
        if num_bits == 32:
            self.qmin = float('-inf')
            self.qmax = float('inf')
        else:
            self.qmin = -(2**(num_bits-1))
            self.qmax = 2**(num_bits-1) - 1
        self.scale = None
        self.zero_point = None
    
    def calibrate(self, tensor):
        """양자화 파라미터 계산"""
        # 32비트는 양자화하지 않음
        if self.num_bits == 32:
            self.scale = 1.0
            self.zero_point = 0
            return
            
        tensor_max = tensor.max().item()
        tensor_min = tensor.min().item()
        
        # 스케일 계산 (symmetric quantization)
        max_abs = max(abs(tensor_min), abs(tensor_max))
        self.scale = max_abs / ((2**(self.num_bits-1)) - 1)
        
        if self.scale == 0:
            self.scale = 1e-8
        
        # zero point는 symmetric quantization에서는 0
        self.zero_point = 0
    
    def quantize(self, tensor):
        """텐서 양자화"""
        if self.scale is None:
            self.calibrate(tensor)
        
        # 32비트는 원본 텐서 반환
        if self.num_bits == 32:
            return tensor
            
        # 스케일링 및 반올림
        quantized = torch.round(tensor / self.scale).clamp(self.qmin, self.qmax)
        
        # 적절한 데이터 타입으로 변환
        if self.num_bits <= 8:
            quantized = quantized.char()  # int8
        else:
            quantized = quantized.short()  # int16
            
        return quantized
    
    def dequantize(self, quantized_tensor):
        """양자화된 텐서 복원"""
        # 32비트는 그대로 반환
        if self.num_bits == 32:
            return quantized_tensor
            
        return quantized_tensor.float() * self.scale

class QuantizedLayer:
    def __init__(self, weight_tensor, num_bits):
        self.num_bits = num_bits
        self.original_shape = weight_tensor.shape
        self.original_type = weight_tensor.dtype
        self.quantizer = WeightQuantizer(num_bits)
        
        # 가중치 양자화
        self.quantized_weight = self.quantizer.quantize(weight_tensor)
    
    def get_quantized_weight(self):
        """양자화된 가중치 반환"""
        return self.quantizer.dequantize(self.quantized_weight)

class QuantizedModel(nn.Module):
    def __init__(self, original_model, num_bits):
        super(QuantizedModel, self).__init__()
        self.original_model = original_model
        self.num_bits = num_bits
        self.quantized_layers = {}
        self.quantize_model()
    
    def quantize_model(self):
        """모델의 모든 가중치 양자화"""
        for name, module in self.original_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self.quantized_layers[name] = QuantizedLayer(
                    module.weight.data,
                    self.num_bits
                )
                # 양자화된 가중치로 업데이트
                module.weight.data.copy_(
                    self.quantized_layers[name].get_quantized_weight()
                )
    
    def forward(self, x):
        return self.original_model(x)
    
    def get_size_estimate(self):
        """양자화된 모델의 예상 크기 계산 (bytes)"""
        total_params = 0
        param_size = 1 if self.num_bits <= 8 else 2  # 8bit 이하는 1바이트, 이상은 2바이트
        
        for name, module in self.original_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                total_params += module.weight.numel()
        
        return (total_params * param_size) + 1024  # 추가 1KB는 메타데이터용

    def save_quantized_model(self, path):
        """양자화된 모델 저장"""
        save_dict = {
            'num_bits': self.num_bits,
            'state_dict': {},
            'quantization_params': {}
        }
        
        for name, qlayer in self.quantized_layers.items():
            save_dict['state_dict'][name] = qlayer.quantized_weight
            save_dict['quantization_params'][name] = {
                'scale': qlayer.quantizer.scale,
                'zero_point': qlayer.quantizer.zero_point
            }
        
        torch.save(save_dict, path)

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100. * correct / total

def main():
    # 데이터셋 준비
    root_dir = './ck+/'
    full_dataset = CKPlusDataset(root_dir)
    train_dataset, val_dataset, test_dataset = split_dataset(full_dataset)
    
    # 데이터셋 정보 출력
    class_counts = {}
    for _, label in train_dataset:
        class_name = train_dataset.classes[label]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print("데이터셋 클래스 분포:")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count} samples")
    
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    device = torch.device("cpu")
    
    # 원본 모델 준비
    original_model = StudentModel().to(device)
    original_model.load_state_dict(torch.load('weights/KD_student_model.pth'))
    original_model.eval()
    
    # 원본 모델 평가
    print("\nEvaluating original model...")
    original_accuracy = evaluate_model(original_model, test_loader, device)
    original_size = os.path.getsize('weights/KD_student_model.pth') / (1024 * 1024)
    
    print(f"Original Model:")
    print(f"Accuracy: {original_accuracy:.2f}%")
    print(f"Model Size: {original_size:.2f} MB")
    
    # 다양한 비트 수로 실험
    bit_widths = [32, 16, 8, 4]
    results = []
    
    for num_bits in bit_widths:
        print(f"\nQuantizing with {num_bits} bits...")
        
        # 모델 복사 및 양자화
        model_copy = StudentModel().to(device)
        model_copy.load_state_dict(original_model.state_dict())
        quantized_model = QuantizedModel(model_copy, num_bits)
        
        # 모델 저장 및 크기 측정
        save_path = f'weights/PTQ_student_model_{num_bits}bits.pth'
        quantized_model.save_quantized_model(save_path)
        
        # 성능 평가
        accuracy = evaluate_model(quantized_model, test_loader, device)
        size = os.path.getsize(save_path) / (1024 * 1024)
        size_reduction = (original_size - size) / original_size * 100
        
        results.append({
            'bits': num_bits,
            'accuracy': accuracy,
            'size': size,
            'size_reduction': size_reduction,
            'theoretical_size': quantized_model.get_size_estimate() / (1024 * 1024)
        })
        
        print(f"{num_bits}-bit Quantized Model:")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Model Size: {size:.2f} MB")
        print(f"Size Reduction: {size_reduction:.2f}%")
        print(f"Theoretical Size: {results[-1]['theoretical_size']:.2f} MB")
    
    # 최적의 결과 출력
    best_result = max(results, key=lambda x: x['accuracy'])
    print("\nBest Configuration:")
    print(f"Bits: {best_result['bits']}")
    print(f"Accuracy: {best_result['accuracy']:.2f}%")
    print(f"Size: {best_result['size']:.2f} MB")
    print(f"Size Reduction: {best_result['size_reduction']:.2f}%")

if __name__ == '__main__':
    main()