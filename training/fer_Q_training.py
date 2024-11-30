import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import cv2
import os
import copy
from data.ckplus_dataset import ImageTransform, CKPlusDataset, split_dataset

class QuantizableStudentModel(nn.Module):
    def __init__(self, num_classes=7):
        super(QuantizableStudentModel, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64 * 12 * 12, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.dequant(x)
        return x
    
    def fuse_model(self):
        """Conv2d-BatchNorm2d-ReLU 패턴을 하나의 모듈로 퓨전"""
        for m in self.modules():
            if type(m) == nn.Sequential:
                torch.quantization.fuse_modules(m, ['0', '1', '2'], inplace=True)

def create_calibration_dataloader(dataset, num_samples=100):
    """Quantization calibration을 위한 작은 데이터셋 생성"""
    indices = torch.randperm(len(dataset))[:num_samples]
    calibration_dataset = torch.utils.data.Subset(dataset, indices)
    return DataLoader(calibration_dataset, batch_size=16)

def train_model(model, train_loader, val_loader, device, num_epochs=50):
    """일반적인 학습 함수"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}] '
                      f'Loss: {loss.item():.4f}')
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Acc: {val_acc:.2f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(model)
            torch.save(model.state_dict(), 'best_float_model.pth')
    
    return best_model

def quantize_model(model, calibration_loader):
    """모델 Quantization 수행"""
    # Quantization 설정
    model.eval()
    model.fuse_model()
    
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    
    # Calibration
    print("Calibrating...")
    with torch.no_grad():
        for inputs, _ in calibration_loader:
            model(inputs)
    
    # Quantization 적용
    torch.quantization.convert(model, inplace=True)
    
    return model

def evaluate_model(model, test_loader, device):
    """모델 평가"""
    model.eval()
    correct = 0
    total = 0
    
    class_correct = [0] * 7
    class_total = [0] * 7
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            if device == 'cpu':
                inputs = inputs
            else:
                inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 클래스별 정확도
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    accuracy = 100. * correct / total
    print(f'Overall Accuracy: {accuracy:.2f}%')
    
    classes = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
    for i in range(7):
        class_acc = 100 * class_correct[i] / class_total[i]
        print(f'Accuracy of {classes[i]}: {class_acc:.2f}%')
    
    return accuracy

def main():
    # 데이터셋 로드 및 분할
    root_dir = './ck+/'
    full_dataset = CKPlusDataset(root_dir)
    
    train_dataset, val_dataset, test_dataset = split_dataset(
        full_dataset,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    # 데이터 로더 생성
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Calibration 데이터 로더 생성
    calibration_loader = create_calibration_dataloader(train_dataset)
    
    # 모델 초기화 (floating point model)
    print("\nInitializing model...")
    float_model = QuantizableStudentModel()
    
    # GPU 학습 (가능한 경우)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    float_model = float_model.to(device)
    
    # Floating point 모델 학습
    print("\nTraining floating point model...")
    float_model = train_model(float_model, train_loader, val_loader, device)
    
    # CPU로 이동하여 quantization 준비
    float_model = float_model.cpu()
    float_model.eval()
    
    # 모델 양자화
    print("\nQuantizing model...")
    quantized_model = quantize_model(float_model, calibration_loader)
    
    # 모델 평가
    print("\nEvaluating floating point model...")
    float_accuracy = evaluate_model(float_model, test_loader, device)
    
    print("\nEvaluating quantized model...")
    quantized_accuracy = evaluate_model(quantized_model, test_loader, 'cpu')
    
    # 모델 저장
    torch.save(quantized_model.state_dict(), 'quantized_model.pth')
    
    # 결과 출력
    print("\nFinal Results:")
    print(f"Floating Point Model Accuracy: {float_accuracy:.2f}%")
    print(f"Quantized Model Accuracy: {quantized_accuracy:.2f}%")
    
    # 모델 크기 비교
    float_size = os.path.getsize('best_float_model.pth') / (1024 * 1024)
    quantized_size = os.path.getsize('quantized_model.pth') / (1024 * 1024)
    
    print(f"\nModel Size Comparison:")
    print(f"Floating Point Model: {float_size:.2f} MB")
    print(f"Quantized Model: {quantized_size:.2f} MB")
    print(f"Size Reduction: {(1 - quantized_size/float_size)*100:.1f}%")

if __name__ == '__main__':
    main()