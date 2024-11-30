import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
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