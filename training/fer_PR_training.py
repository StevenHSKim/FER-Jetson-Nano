import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.prune as prune
from PIL import Image
import cv2
import numpy as np
import os
from data.ckplus_dataset import ImageTransform, CKPlusDataset, split_dataset

class PrunableStudentModel(nn.Module):
    def __init__(self, num_classes=7):
        super(PrunableStudentModel, self).__init__()
        # 프루닝을 위해 각 레이어를 개별적으로 정의
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.fc1 = nn.Linear(64 * 12 * 12, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.maxpool(self.bn1(self.relu(self.conv1(x))))
        x = self.maxpool(self.bn2(self.relu(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def apply_pruning(model, pruning_amount=0.5):
    """모델에 프루닝 적용"""
    # Convolutional 레이어 프루닝
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=pruning_amount)
            
        # Linear 레이어 프루닝
        elif isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=pruning_amount)
    
    return model

def remove_pruning(model):
    """임시 프루닝 마스크를 영구적으로 적용"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.remove(module, 'weight')
    
    return model

def calculate_sparsity(model):
    """모델의 스파시티(0인 파라미터의 비율) 계산"""
    total_params = 0
    zero_params = 0
    
    for param in model.parameters():
        if param is not None:
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
    
    return zero_params / total_params * 100

def train_pruned_model(model, train_loader, val_loader, device, num_epochs=50, 
                      pruning_amount=0.5, pruning_iterations=3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    best_val_acc = 0.0
    
    # 점진적 프루닝 수행
    for iteration in range(pruning_iterations):
        print(f"\nPruning Iteration {iteration + 1}/{pruning_iterations}")
        print(f"Applying {pruning_amount*100}% pruning...")
        
        # 프루닝 적용
        model = apply_pruning(model, pruning_amount)
        current_sparsity = calculate_sparsity(model)
        print(f"Current model sparsity: {current_sparsity:.2f}%")
        
        # 프루닝된 모델 재학습
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
                # 임시 프루닝 마스크를 영구적으로 적용
                model = remove_pruning(model)
                torch.save(model.state_dict(), f'pruned_model_iter_{iteration+1}.pth')
                print(f'Model saved with validation accuracy: {val_acc:.2f}%')
    
    return model

def main():
    # 데이터셋 로드 및 분할 (기존 코드와 동일)
    root_dir = './ck+/'
    full_dataset = CKPlusDataset(root_dir)
    
    train_dataset, val_dataset, test_dataset = split_dataset(
        full_dataset,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    # 데이터 로더 생성
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prunable 모델 초기화
    model = PrunableStudentModel().to(device)
    
    # 프루닝 파라미터 설정
    pruning_amount = 0.3  # 각 반복마다 30%의 파라미터 프루닝
    pruning_iterations = 3  # 총 3번의 프루닝 수행
    
    # 모델 학습 및 프루닝
    pruned_model = train_pruned_model(
        model, 
        train_loader, 
        val_loader, 
        device,
        num_epochs=50,
        pruning_amount=pruning_amount,
        pruning_iterations=pruning_iterations
    )
    
    # 최종 모델 평가
    print("\nEvaluating final pruned model...")
    evaluate_model(pruned_model, test_loader, device)
    
    # 최종 스파시티 계산
    final_sparsity = calculate_sparsity(pruned_model)
    print(f"\nFinal model sparsity: {final_sparsity:.2f}%")

if __name__ == '__main__':
    main()