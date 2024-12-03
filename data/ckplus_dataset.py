import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torchvision import transforms

class ImageTransform:
    def __init__(self, is_train=True):
        # 학습용 transform - 데이터 증강 포함
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 검증/테스트용 transform - 기본 전처리만
        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.is_train = is_train
        
    def __call__(self, image):
        if self.is_train:
            return self.train_transform(image)
        return self.val_transform(image)

class CKPlusDataset(Dataset):
    def __init__(self, root_dir, image_paths=None, labels=None, transform=None, is_train=True):
        self.transform = transform if transform else ImageTransform(is_train=is_train)
        self.classes = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
        
        if image_paths is None or labels is None:
            self.image_paths = []
            self.labels = []
            
            # 클래스별 이미지 수 카운트
            class_counts = {emotion: 0 for emotion in self.classes}
            
            # CK+ 데이터셋 로드
            for label, emotion in enumerate(self.classes):
                emotion_dir = os.path.join(root_dir, emotion)
                if os.path.exists(emotion_dir):
                    for img_name in os.listdir(emotion_dir):
                        if img_name.endswith(('.jpg', '.jpeg', '.png')):  # 이미지 파일만 추가
                            self.image_paths.append(os.path.join(emotion_dir, img_name))
                            self.labels.append(label)
                            class_counts[emotion] += 1
            
            # 클래스별 데이터 분포 출력
            print("\n데이터셋 클래스 분포:")
            for emotion, count in class_counts.items():
                print(f"{emotion}: {count} samples")
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
    """데이터셋을 train, validation, test set으로 분할"""
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
    
    # train, val, test에 각각 다른 transform 적용
    train_dataset = CKPlusDataset(None, train_paths, train_labels, transform=ImageTransform(is_train=True))
    val_dataset = CKPlusDataset(None, val_paths, val_labels, transform=ImageTransform(is_train=False))
    test_dataset = CKPlusDataset(None, test_paths, test_labels, transform=ImageTransform(is_train=False))
    
    return train_dataset, val_dataset, test_dataset