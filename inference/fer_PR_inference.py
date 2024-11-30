import torch
import torch.nn as nn
import numpy as np
import cv2
import time
from PIL import Image

class ImageTransform:
    def __init__(self, size=(48, 48)):
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

class PrunedStudentModel(nn.Module):
    def __init__(self, num_classes=7):
        super(PrunedStudentModel, self).__init__()
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

def calculate_inference_stats(model, device, input_size=(48, 48), num_runs=100):
    """추론 시간과 FPS 측정"""
    dummy_input = torch.randn(1, 3, *input_size).to(device)
    
    # 워밍업
    for _ in range(10):
        _ = model(dummy_input)
    
    # 시간 측정
    start_time = time.time()
    for _ in range(num_runs):
        _ = model(dummy_input)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    fps = 1.0 / avg_time
    
    return avg_time * 1000, fps  # ms 단위로 반환

def realtime_emotion_recognition(model_path, device='cuda', confidence_threshold=0.5):
    """
    프루닝된 모델을 사용한 실시간 감정 인식
    """
    emotions = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
    
    # 디바이스 설정
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # 모델 초기화 및 가중치 로드
    model = PrunedStudentModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # 추론 성능 측정
    avg_inference_time, avg_fps = calculate_inference_stats(model, device)
    print(f"Average inference time: {avg_inference_time:.2f}ms")
    print(f"Average FPS: {avg_fps:.2f}")
    
    # 이미지 전처리를 위한 transform 초기화
    transform = ImageTransform()
    
    # OpenCV face detector 초기화
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # 카메라 초기화
    cap = cv2.VideoCapture(0)
    
    # FPS 측정을 위한 변수
    frame_times = []
    fps_update_interval = 30  # 30프레임마다 FPS 업데이트
    
    print("Press 'q' to quit")
    
    try:
        while True:
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            # 프레임을 그레이스케일로 변환
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 얼굴 검출
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            # 각 얼굴에 대해 감정 인식 수행
            for (x, y, w, h) in faces:
                # 얼굴 영역 추출
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
            
            # 프레임 처리 시간 계산
            frame_time = time.time() - start_time
            frame_times.append(frame_time)
            
            # FPS 계산 및 표시 (주기적으로 업데이트)
            if len(frame_times) >= fps_update_interval:
                current_fps = 1.0 / (sum(frame_times) / len(frame_times))
                frame_times = []  # Reset for next interval
                
                # FPS 표시
                cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 추론 시간 표시
                cv2.putText(frame, f"Inference: {avg_inference_time:.1f}ms", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 결과 화면 표시
            cv2.imshow('Pruned Emotion Recognition', frame)
            
            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        # 리소스 해제
        cap.release()
        cv2.destroyAllWindows()

def main():
    model_path = 'pruned_model_iter_3.pth'  # 최종 프루닝된 모델 경로
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Running inference on {device}")
    print(f"Loading pruned model from {model_path}")
    
    try:
        realtime_emotion_recognition(
            model_path=model_path,
            device=device,
            confidence_threshold=0.5
        )
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == '__main__':
    main()