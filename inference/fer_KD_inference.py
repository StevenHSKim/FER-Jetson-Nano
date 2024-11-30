import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import time

class ImageTransform:
    def __init__(self, size=(48, 48)):  # 128x128에서 48x48로 변경
        self.size = size
        
    def __call__(self, image):
        # OpenCV BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = cv2.resize(image, self.size)
        
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

class StudentModel(nn.Module):
    def __init__(self, num_classes=7):
        super(StudentModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),  # 24x24
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),  # 12x12
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

def realtime_emotion_recognition(model_path, device='cuda', confidence_threshold=0.5):
    """
    실시간 카메라 스트림에서 감정을 인식하는 함수
    Args:
        model_path: 학습된 모델 가중치 경로
        device: 추론에 사용할 디바이스
        confidence_threshold: 감정을 표시할 최소 신뢰도 임계값
    """
    # 감정 클래스 정의
    emotions = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
    
    # 디바이스 설정
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # 모델 초기화 및 가중치 로드
    model = StudentModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # 이미지 전처리를 위한 transform 초기화
    transform = ImageTransform()
    
    # OpenCV face detector 초기화
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # 카메라 초기화
    cap = cv2.VideoCapture(0)
    
    # FPS 계산을 위한 변수
    prev_time = 0
    
    print("Press 'q' to quit")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # FPS 계산
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time
            
            # 프레임을 그레이스케일로 변환
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 얼굴 검출
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            # 각 얼굴에 대해 감정 인식 수행
            for (x, y, w, h) in faces:
                # 얼굴 영역 추출 및 전처리
                face_roi = frame[y:y+h, x:x+w]
                
                # 이미지 전처리
                img_tensor = transform(face_roi)
                img_tensor = img_tensor.unsqueeze(0).to(device)
                
                # 추론
                with torch.no_grad():
                    outputs = model(img_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    predicted_idx = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0][predicted_idx].item()
                
                # 결과 시각화
                if confidence > confidence_threshold:
                    emotion = emotions[predicted_idx]
                    
                    # 얼굴 주변에 박스 그리기
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # 감정과 신뢰도 표시
                    text = f"{emotion} ({confidence:.2%})"
                    cv2.putText(frame, text, (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # FPS 표시
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 결과 화면 표시
            cv2.imshow('Emotion Recognition', frame)
            
            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        # 리소스 해제
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    model_path = 'KD_best_student_model.pth'  # 학습된 모델 가중치 경로
    realtime_emotion_recognition(model_path, confidence_threshold=0.5)