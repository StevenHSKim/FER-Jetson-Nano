import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split


class ImageTransform:
    def __init__(self):
        pass
        
    def __call__(self, image):
        # PIL Image를 numpy array로 변환
        image = np.array(image)
        
        # Convert to float and normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Normalize with ImageNet statistics
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # (H, W, C) -> (C, H, W)
        image = image.transpose(2, 0, 1)
        
        # Convert to tensor
        image = torch.FloatTensor(image)
        return image

class CKPlusDataset(Dataset):
    def __init__(self, root_dir, image_paths=None, labels=None, transform=None):
        self.transform = transform if transform else ImageTransform()
        self.classes = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
        
        if image_paths is None or labels is None:
            self.image_paths = []
            self.labels = []
            
            # CK+ 데이터셋 로드
            for label, emotion in enumerate(self.classes):
                emotion_dir = os.path.join(root_dir, emotion)
                if os.path.exists(emotion_dir):
                    for img_name in os.listdir(emotion_dir):
                        self.image_paths.append(os.path.join(emotion_dir, img_name))
                        self.labels.append(label)
        else:
            self.image_paths = image_paths
            self.labels = labels
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    데이터셋을 train, validation, test set으로 분할
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "Ratios must sum to 1"
    
    # 전체 데이터 개수
    n_total = len(dataset)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val
    
    # 데이터셋 분할
    train_data, temp_data = train_test_split(
        list(zip(dataset.image_paths, dataset.labels)),
        test_size=(val_ratio + test_ratio),
        random_state=random_seed,
        stratify=dataset.labels
    )
    
    # validation과 test 데이터 분할
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_data, test_data = train_test_split(
        temp_data,
        test_size=(1 - val_ratio_adjusted),
        random_state=random_seed,
        stratify=[label for _, label in temp_data]
    )
    
    # 데이터셋 생성
    train_paths, train_labels = zip(*train_data)
    val_paths, val_labels = zip(*val_data)
    test_paths, test_labels = zip(*test_data)
    
    transform = dataset.transform
    train_dataset = CKPlusDataset(None, train_paths, train_labels, transform)
    val_dataset = CKPlusDataset(None, val_paths, val_labels, transform)
    test_dataset = CKPlusDataset(None, test_paths, test_labels, transform)
    
    return train_dataset, val_dataset, test_dataset

class TeacherModel(nn.Module):
    def __init__(self, num_classes=7):
        super(TeacherModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),  # 24x24
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),  # 12x12
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),  # 6x6
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.classifier(x)
        return x

class StudentModel(nn.Module):
    def __init__(self, num_classes=7):
        super(StudentModel, self).__init__()
        self.features = nn.Sequential(
            # Increase filters slightly from previous version
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),  # Reduced pooling size for better feature preservation
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 48, kernel_size=3, padding=1),  # Added one more conv layer
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(48),
            nn.MaxPool2d(2, 2)
        )
        
        # Increased classifier capacity slightly
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),  # Added lighter dropout
            nn.Linear(48 * 6 * 6, 128),  # Increased hidden layer size
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class DistillationLoss(nn.Module):
    def __init__(self, temp=4.0, alpha=0.5):
        super(DistillationLoss, self).__init__()
        self.temp = temp
        self.alpha = alpha
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, student_outputs, teacher_outputs, labels):
        hard_loss = self.criterion(student_outputs, labels)
        
        soft_student = nn.functional.log_softmax(student_outputs / self.temp, dim=1)
        soft_teacher = nn.functional.softmax(teacher_outputs / self.temp, dim=1)
        soft_loss = nn.KLDivLoss(reduction='batchmean')(soft_student, soft_teacher) * (self.temp ** 2)
        
        loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        return loss

def train_teacher_model(model, train_loader, val_loader, device, num_epochs=50):
    criterion = nn.CrossEntropyLoss()
    # Reduce learning rate for CPU training
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
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
            
            # Add progress printing for CPU training
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
            torch.save(model.state_dict(), 'weights/KD_teacher_model.pth')
            print(f'Model saved with validation accuracy: {val_acc:.2f}%')

def train_student_model_with_distillation(teacher_model, student_model, train_loader, val_loader, 
                                          device, num_epochs=50):
    teacher_model.eval()
    student_model.train()
    
    criterion = DistillationLoss(temp=4.0, alpha=0.5)
    # Reduce learning rate for CPU training
    optimizer = optim.Adam(student_model.parameters(), lr=0.0005)
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            with torch.no_grad():
                teacher_outputs = teacher_model(inputs)
            
            optimizer.zero_grad()
            student_outputs = student_model(inputs)
            
            loss = criterion(student_outputs, teacher_outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            _, predicted = student_outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Add progress printing for CPU training
            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}] '
                      f'Loss: {loss.item():.4f}')
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Validation
        student_model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = student_model(inputs)
                
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Acc: {val_acc:.2f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(student_model.state_dict(), 'weights/KD_student_model.pth')
        
        student_model.train()
    
    return student_model

def evaluate_model(model, test_loader, device):
    """
    모델 평가를 위한 함수
    """
    model.eval()
    correct = 0
    total = 0
    
    class_correct = [0] * 7
    class_total = [0] * 7
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 클래스별 정확도 계산
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    # 전체 정확도
    accuracy = 100. * correct / total
    print(f'Overall Accuracy: {accuracy:.2f}%')
    
    # 클래스별 정확도
    classes = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
    for i in range(7):
        class_acc = 100 * class_correct[i] / class_total[i]
        print(f'Accuracy of {classes[i]}: {class_acc:.2f}%')
    
    return accuracy

def count_parameters(model):
    """모델의 학습 가능한 파라미터 수를 계산하는 함수"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    # 데이터셋 로드 및 분할
    root_dir = './ck+/'  # Update path to local directory
    full_dataset = CKPlusDataset(root_dir)
    
    print("Total dataset size:", len(full_dataset))
    
    # 데이터셋 분할
    train_dataset, val_dataset, test_dataset = split_dataset(
        full_dataset,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    
    # 디바이스 설정
    device = torch.device("cpu")
    
    # Teacher 모델 초기화 및 파라미터 수 출력
    print("\nInitializing teacher model...")
    teacher_model = TeacherModel().to(device)
    teacher_params = count_parameters(teacher_model)
    print(f"Number of parameters in teacher model: {teacher_params:,}")
    
    # Student 모델 초기화 및 파라미터 수 출력
    print("\nInitializing student model...")
    student_model = StudentModel().to(device)
    student_params = count_parameters(student_model)
    print(f"Number of parameters in student model: {student_params:,}")
    print(f"Parameter reduction: {(1 - student_params/teacher_params)*100:.1f}%")
    
    # 데이터 로더 생성
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # 디바이스 설정
    device = torch.device("cpu")  # CPU 사용
    
    # Teacher 모델 초기화 및 학습
    print("\nInitializing and training teacher model...")
    teacher_model = TeacherModel().to(device)
    
    # Teacher 모델 학습
    train_teacher_model(teacher_model, train_loader, val_loader, device, num_epochs=50)
    
    # 최고 성능의 Teacher 모델 로드
    teacher_model.load_state_dict(torch.load('weights/KD_teacher_model.pth'))
    
    # Teacher 모델 평가
    print("\nEvaluating teacher model...")
    teacher_acc = evaluate_model(teacher_model, test_loader, device)
    
    # Student 모델 초기화
    print("\nInitializing and training student model...")
    student_model = StudentModel().to(device)
    
    # Knowledge Distillation 학습
    trained_student = train_student_model_with_distillation(
        teacher_model, student_model, train_loader, val_loader, device, num_epochs=50)
    
    # 최고 성능의 Student 모델 로드
    student_model.load_state_dict(torch.load('weights/KD_student_model.pth'))
    
    # Student 모델 평가
    print("\nEvaluating student model...")
    student_acc = evaluate_model(student_model, test_loader, device)
    
    print("\nFinal Results:")
    print(f"Teacher Model Accuracy: {teacher_acc:.2f}%")
    print(f"Student Model Accuracy: {student_acc:.2f}%")

if __name__ == '__main__':
    main()