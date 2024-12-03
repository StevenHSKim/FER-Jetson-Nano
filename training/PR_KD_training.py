import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from training.KD_training import StudentModel
from data.ckplus_dataset import CKPlusDataset, split_dataset

class PrunedStudentModel(nn.Module):
    # 각 pruning level에 따른 필터 구성
    PRUNED_CONFIGS = {
        'low': {'filters': [12, 22, 28]},
        'medium': {'filters': [11, 19, 24]},
        'high': {'filters': [9, 16, 19]}
    }
    
    def __init__(self, pruning_level='medium'):
        super().__init__()
        if pruning_level not in self.PRUNED_CONFIGS:
            raise ValueError(f"Pruning level must be one of {list(self.PRUNED_CONFIGS.keys())}")
            
        filters = self.PRUNED_CONFIGS[pruning_level]['filters']
        
        self.features = nn.Sequential(
            nn.Conv2d(3, filters[0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(filters[0]),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(filters[0], filters[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(filters[1]),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(filters[1], filters[2], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(filters[2]),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(filters[2] * 6 * 6, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, 7)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class StructuredPruner(nn.Module):
    # Predefined pruning ratios for each level
    PRUNING_CONFIGS = {
        'low': {'conv1': 0.2, 'conv2': 0.3, 'conv3': 0.4},
        'medium': {'conv1': 0.3, 'conv2': 0.4, 'conv3': 0.5},
        'high': {'conv1': 0.4, 'conv2': 0.5, 'conv3': 0.6}
    }
    
    def __init__(self, model=None, pruning_level='medium'):
        """
        Args:
            model: The pre-trained student model
            pruning_level: One of 'low', 'medium', 'high'
        """
        super().__init__()
        if model is None:
            model = StudentModel()
        
        # model을 named_modules로 등록하기 위해 Module로 감싸기
        self.model = model
        
        if pruning_level not in self.PRUNING_CONFIGS:
            raise ValueError(f"Pruning level must be one of {list(self.PRUNING_CONFIGS.keys())}")
        
        self.prune_ratios = self.PRUNING_CONFIGS[pruning_level]
        self.pruning_level = pruning_level
        
    def calculate_filter_importance(self, layer):
        """Calculate L1-norm of each filter in a convolutional layer"""
        weights = layer.weight.data
        l1_norm = torch.sum(torch.abs(weights), dim=[1, 2, 3])
        return l1_norm
        
    def get_pruning_indexes(self, importance_scores, prune_ratio):
        """Get indexes of filters to keep based on importance scores"""
        num_filters = len(importance_scores)
        num_keep = int(num_filters * (1 - prune_ratio))
        
        # Get indexes of top K filters by importance score
        keep_indexes = torch.argsort(importance_scores, descending=True)[:num_keep]
        return keep_indexes.sort()[0]  # Sort indexes for consistent pruning
        
    def prune_conv_layer(self, layer, keep_indexes):
        """Prune filters from convolutional layer"""
        # Prune output channels (filters)
        layer.weight.data = layer.weight.data[keep_indexes]
        if layer.bias is not None:
            layer.bias.data = layer.bias.data[keep_indexes]
        layer.out_channels = len(keep_indexes)
        
        return layer
        
    def prune_batchnorm_layer(self, bn_layer, keep_indexes):
        """Prune corresponding channels in batch normalization layer"""
        bn_layer.weight.data = bn_layer.weight.data[keep_indexes]
        bn_layer.bias.data = bn_layer.bias.data[keep_indexes]
        bn_layer.running_mean = bn_layer.running_mean[keep_indexes]
        bn_layer.running_var = bn_layer.running_var[keep_indexes]
        bn_layer.num_features = len(keep_indexes)
        
        return bn_layer
        
    def adjust_following_conv(self, conv_layer, prev_keep_indexes):
        """Adjust input channels of following convolutional layer"""
        conv_layer.weight.data = conv_layer.weight.data[:, prev_keep_indexes]
        conv_layer.in_channels = len(prev_keep_indexes)
        
        return conv_layer
        
    def adjust_fc_layer(self, fc_layer, last_conv_keep_indexes, original_shape):
        """첫 번째 FC 레이어의 입력 특성 조정"""
        # 이전 가중치 저장
        original_fc_weight = fc_layer.weight.data
        original_fc_bias = fc_layer.bias.data if fc_layer.bias is not None else None
        
        # 새로운 입력 특성 수 계산
        new_in_features = len(last_conv_keep_indexes) * 6 * 6  # 6x6는 마지막 conv layer의 출력 크기
        
        # 새로운 FC 레이어 생성
        new_fc = nn.Linear(new_in_features, fc_layer.out_features)
        
        # 가중치 재구성
        old_weight = original_fc_weight.view(fc_layer.out_features, -1, 6, 6)
        new_weight = old_weight[:, last_conv_keep_indexes, :, :]
        new_fc.weight.data = new_weight.reshape(fc_layer.out_features, new_in_features)
        
        # 바이어스 복사
        if original_fc_bias is not None:
            new_fc.bias.data = original_fc_bias
        
        return new_fc

    def prune_model(self):
        """모델의 구조적 프루닝 수행"""
        features = self.model.features
        prev_keep_indexes = None
        last_keep_indexes = None
        
        # Prune convolutional layers
        for i in range(len(features)):
            layer = features[i]
            
            if isinstance(layer, nn.Conv2d):
                # Get layer name for pruning ratio
                layer_name = f'conv{i//3 + 1}'
                if layer_name not in self.prune_ratios:
                    continue
                
                # Adjust input channels based on previous pruning
                if prev_keep_indexes is not None:
                    layer = self.adjust_following_conv(layer, prev_keep_indexes)
                
                # Calculate importance and get pruning indexes
                importance = self.calculate_filter_importance(layer)
                keep_indexes = self.get_pruning_indexes(importance, 
                                                    self.prune_ratios[layer_name])
                
                # Prune current layer
                layer = self.prune_conv_layer(layer, keep_indexes)
                features[i] = layer
                
                prev_keep_indexes = keep_indexes
                last_keep_indexes = keep_indexes
                
            elif isinstance(layer, nn.BatchNorm2d) and prev_keep_indexes is not None:
                layer = self.prune_batchnorm_layer(layer, prev_keep_indexes)
                features[i] = layer
        
        # Adjust first FC layer
        if last_keep_indexes is not None:
            # StudentModel의 classifier는 Sequential이므로 첫 번째 Linear 레이어만 조정
            self.model.classifier[1] = self.adjust_fc_layer(
                self.model.classifier[1],
                last_keep_indexes,
                None  # 마지막 conv layer의 출력 크기는 항상 6x6
            )
        
        return self.model
    
    def forward(self, x):
        return self.model(x)

def fine_tune_pruned_model(model, train_loader, val_loader, device, level, num_epochs=10):
    """Fine-tune the pruned model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Lower learning rate for fine-tuning
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
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
            save_path = f'weights/PR_student_model_{level}.pth'
            torch.save(model.state_dict(), save_path)
    
    return model

def main():
    # 데이터셋 준비
    root_dir = './ck+/'
    full_dataset = CKPlusDataset(root_dir)
    train_dataset, val_dataset, test_dataset = split_dataset(full_dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # CPU 사용
    device = torch.device("cpu")
    
    print("\n=== Starting Pruning Process ===")
    
    # 각 pruning level에 대해 처리
    for level in ['low', 'medium', 'high']:
        print(f"\n=== Processing {level.upper()} pruning level ===")
        
        # Knowledge Distilled 모델 로드
        print("Loading Knowledge Distilled model...")
        student_model = StudentModel().to(device)
        student_model.load_state_dict(torch.load('weights/KD_student_model.pth'))
        student_model.eval()
        
        # Pruner 생성 및 pruning 수행
        print(f"\nPerforming {level} level structured pruning...")
        pruner = StructuredPruner(student_model, pruning_level=level)
        pruned_model = pruner.prune_model()
        
        # Fine-tuning
        print(f"\nFine-tuning {level} pruned model...")
        fine_tuned_model = fine_tune_pruned_model(
            pruned_model, train_loader, val_loader, device, level, num_epochs=10)
        
        # 모델 저장
        save_path = f'weights/PR_student_model_{level}.pth'
        torch.save(fine_tuned_model.state_dict(), save_path)
        print(f"\nSaved {level} pruned model to: {save_path}")
        
        # 모델 평가
        print(f"\nEvaluating {level} pruned model...")
        fine_tuned_model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = fine_tuned_model(inputs)
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = 100. * correct / total
        
        # 결과 출력
        print(f"\nResults for {level.upper()} pruning level:")
        print(f"Test Accuracy: {accuracy:.2f}%")
        
        # 모델 크기 계산
        param_count = sum(p.numel() for p in fine_tuned_model.parameters())
        model_size = os.path.getsize(save_path) / (1024 * 1024)  # MB 단위
        
        print(f"Parameter Count: {param_count:,}")
        print(f"Model Size: {model_size:.2f} MB")
        
        # 구분선 출력
        print("\n" + "="*50)
    
    print("\nPruning process completed for all levels!")
    print("\nSummary of saved models:")
    print("-" * 40)
    for level in ['low', 'medium', 'high']:
        model_path = f'weights/PR_student_model_{level}.pth'
        if os.path.exists(model_path):
            size = os.path.getsize(model_path) / (1024 * 1024)
            print(f"{level.upper():6} pruning model: {size:.2f} MB")
    print("-" * 40)

if __name__ == '__main__':
    main()