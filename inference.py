import torch
import torch.nn as nn
import numpy as np
import cv2
import time
from pathlib import Path
import argparse

class ImageTransform:
    def __init__(self, size=(48, 48)):
        self.size = size
        
    def __call__(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.size)
        image = image.astype(np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        image = image.transpose(2, 0, 1)
        image = torch.FloatTensor(image)
        return image

class BaseModel(nn.Module):
    """기본 Student 모델 구조"""
    def __init__(self, num_classes=7):
        super(BaseModel, self).__init__()
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
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def load_model(model_path, device):
    """모델 로드 함수"""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # 모델 초기화
    model = BaseModel()
    
    # 가중치 로드
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")
    
    model = model.to(device)
    model.eval()
    
    return model

def calculate_model_stats(model, device):
    """모델 통계 계산 (파라미터 수, 모델 크기, 추론 시간)"""
    # 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    
    # 모델 크기 계산 (MB)
    model_size = total_params * 4 / (1024 * 1024)  # 32-bit float 기준
    
    # 추론 시간 측정
    dummy_input = torch.randn(1, 3, 48, 48).to(device)
    
    # 워밍업
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # 실제 측정
    times = []
    with torch.no_grad():
        for _ in range(100):
            start_time = time.time()
            _ = model(dummy_input)
            times.append(time.time() - start_time)
    
    avg_inference_time = np.mean(times) * 1000  # ms 단위로 변환
    fps = 1.0 / np.mean(times)
    
    return {
        'total_params': total_params,
        'model_size_mb': model_size,
        'avg_inference_time_ms': avg_inference_time,
        'theoretical_fps': fps
    }

def realtime_emotion_recognition(model_path, device='cuda', confidence_threshold=0.5):
    """통합된 실시간 감정 인식 함수"""
    emotions = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # 모델 로드
    model = load_model(model_path, device)
    
    # 모델 통계 계산
    stats = calculate_model_stats(model, device)
    print("\nModel Statistics:")
    print(f"Total Parameters: {stats['total_params']:,}")
    print(f"Model Size: {stats['model_size_mb']:.2f} MB")
    print(f"Average Inference Time: {stats['avg_inference_time_ms']:.2f} ms")
    print(f"Theoretical FPS: {stats['theoretical_fps']:.2f}")
    
    transform = ImageTransform()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    cap = cv2.VideoCapture(0)
    frame_times = []
    fps_update_interval = 30
    
    print("\nPress 'q' to quit")
    try:
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                img_tensor = transform(face_roi)
                img_tensor = img_tensor.unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(img_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    predicted_idx = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0][predicted_idx].item()
                
                if confidence > confidence_threshold:
                    emotion = emotions[predicted_idx]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    text = f"{emotion} ({confidence:.2%})"
                    cv2.putText(frame, text, (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            frame_time = time.time() - start_time
            frame_times.append(frame_time)
            
            if len(frame_times) >= fps_update_interval:
                current_fps = 1.0 / (sum(frame_times) / len(frame_times))
                frame_times = []
                
                cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Inference: {stats['avg_inference_time_ms']:.1f}ms", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Emotion Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Facial Expression Recognition Inference')
    parser.add_argument('--model', type=str, required=True, help='Path to the model weights')
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'], 
                      help='Device to run inference on')
    parser.add_argument('--threshold', type=float, default=0.5, 
                      help='Confidence threshold for emotion detection')
    
    args = parser.parse_args()
    
    try:
        realtime_emotion_recognition(
            model_path=args.model,
            device=args.device,
            confidence_threshold=args.threshold
        )
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == '__main__':
    main()