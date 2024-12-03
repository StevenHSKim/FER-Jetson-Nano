# -*- coding: utf-8 -*-

import os
import time
import argparse
import torch
import cv2
import pandas as pd
import platform
import urllib.request

from evaluate.model_evaluator import ModelEvaluator
from evaluate.inference_evaluator import InferenceEvaluator
from training.KD_training import TeacherModel, StudentModel
from training.PR_KD_training import PrunedStudentModel
from training.PTQ_KD_training import QuantizedModel

# Get current script directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
fer_dir = os.path.join(project_dir, "FER")

# Modify import paths to use absolute paths from project root
import sys
sys.path.append(project_dir)

class LightEmotionRecognizer:
    def __init__(self, model, device, input_size=(48, 48)):
        self.model = model
        self.device = device
        self.input_size = input_size
        self.emotions = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise']
        
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        # Improved Haar Cascade file path handling
        self.system = platform.system()
        if self.system == "Darwin":  # macOS
            cascade_path = os.path.join(current_dir, 'haarcascade_frontalface_default.xml')
            if not os.path.exists(cascade_path):
                print("Downloading Haar Cascade classifier...")
                url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
                try:
                    urllib.request.urlretrieve(url, cascade_path)
                except Exception as e:
                    print("Failed to download cascade file: {}".format(e))
                    # Try OpenCV default installation path
                    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        else:  # Linux
            cascade_path = '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
            if not os.path.exists(cascade_path):
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        # Check cascade file existence and load
        if not os.path.exists(cascade_path):
            raise FileNotFoundError("Cascade file not found at: {}".format(cascade_path))
            
        self.face_detector = cv2.CascadeClassifier(cascade_path)
        if self.face_detector.empty():
            raise ValueError("Failed to load cascade classifier")
            
        print("Loaded face detector from: {}".format(cascade_path))
    
    def preprocess_frame(self, frame):
        """Optimized frame preprocessing"""
        if frame is None:
            print("Error: Empty frame received")
            return [], []
            
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except cv2.error:
            print("Error: Failed to convert frame to grayscale")
            return [], []
            
        try:
            faces = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                maxSize=(300, 300)
            )
        except cv2.error as e:
            print("Face detection error: {}".format(e))
            return [], []
        
        if len(faces) == 0:
            return [], []
            
        face_imgs = []
        face_positions = []
        
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face, self.input_size)
            
            face_tensor = torch.from_numpy(face_resized).permute(2, 0, 1).float()
            face_tensor = face_tensor / 255.0
            face_tensor = (face_tensor - self.mean) / self.std
            face_tensor = face_tensor.unsqueeze(0)
            
            face_imgs.append(face_tensor)
            face_positions.append((x, y, w, h))
            
        return face_imgs, face_positions
    
    @torch.no_grad()
    def predict_emotion(self, face_img):
        """Optimized emotion prediction"""
        output = self.model(face_img)
        probabilities = torch.softmax(output, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()
        
        return self.emotions[prediction], confidence

    def draw_results(self, frame, face_positions, emotions, confidences):
        """Optimized result drawing"""
        for (x, y, w, h), emotion, conf in zip(face_positions, emotions, confidences):
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            label = "{}: {:.2f}".format(emotion, conf)
            cv2.putText(frame, label, 
                       (x, max(y - 10, 0)),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (0, 255, 0), 1)
        
        return frame

def get_camera_pipeline():
    """Return appropriate camera pipeline based on platform"""
    system = platform.system()
    
    if system == "Darwin":  # macOS
        return cv2.VideoCapture(0)
    elif system == "Linux":  # Linux (Jetson Nano)
        if os.path.exists('/etc/nv_tegra_release'):
            pipeline = (
                "nvarguscamerasrc ! "
                "video/x-raw(memory:NVMM), "
                "width=(int){}, height=(int){}, "
                "format=(string)NV12, framerate=(fraction){}/1 ! "
                "nvvidconv flip-method={} ! "
                "video/x-raw, width=(int){}, height=(int){}, "
                "format=(string)BGRx ! "
                "videoconvert ! "
                "video/x-raw, format=(string)BGR ! "
                "appsink".format(
                    640, 480,  # capture width, height
                    15,        # framerate
                    0,         # flip method
                    640, 480   # display width, height
                )
            )
            return cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        else:  # Regular Linux
            return cv2.VideoCapture(0)
    else:  # Windows or other
        return cv2.VideoCapture(0)


def evaluate_model(model, device, model_type, subtype=None):
    """Evaluate model performance and characteristics
    Args:
        model: The model to evaluate
        device: Device to run evaluation on
        model_type: Type of model ('teacher', 'kd', 'pr_kd', 'ptq_kd')
        subtype: Additional info - pruning level for PR-KD or bits for PTQ-KD
    """
    print("\nEvaluating {} model performance and characteristics...".format(model_type))
    
    model_evaluator = ModelEvaluator()
    model_stats = model_evaluator.evaluate_model(
        model=model,
        model_name="EmotionRecognition",
        input_shape=(1, 3, 48, 48)
    )
    
    dummy_input = torch.randn(1, 3, 48, 48).to(device)
    inference_evaluator = InferenceEvaluator(warm_up=10)
    inference_stats = inference_evaluator.measure_inference_time(
        model=model,
        input_tensor=dummy_input,
        model_name="EmotionRecognition",
        device=device,
        num_runs=100
    )
    
    model_evaluator.print_results()
    inference_evaluator.print_results()
    
    # Create combined results dictionary
    combined_results = {
        'model_type': model_type,
        'subtype': subtype if subtype else 'default',
    }
    # Update dictionary with model_stats items
    for key, value in model_stats.items():
        combined_results[key] = value
    # Add inference time stats
    combined_results['100_inference_time_mean_std'] = inference_stats['inference_time_mean_std']
    
    # Add additional info to print results
    print("\nModel Size: {:.3f} MB".format(combined_results['model_size_mb']))
    
    results_dir = os.path.join(current_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if subtype:
        results_file = os.path.join(results_dir, "{0}_{1}_model_evaluation_{2}.csv".format(
            model_type, subtype, timestamp))
    else:
        results_file = os.path.join(results_dir, "{0}_model_evaluation_{1}.csv".format(
            model_type, timestamp))
    
    df = pd.DataFrame([combined_results])
    df.to_csv(results_file, index=False)
    print("\nEvaluation results saved to {}".format(results_file))
    
    return combined_results


def main():
    parser = argparse.ArgumentParser(description='Lightweight Emotion Recognition')
    parser.add_argument('--model', type=str, required=True, 
                       choices=['teacher', 'kd', 'pr_kd', 'ptq_kd'],
                       help='Model type to use for inference')
    parser.add_argument('--pr-level', type=str, default='medium',
                       choices=['low', 'medium', 'high'],
                       help='Pruning level for PR-KD model (default: medium)')
    parser.add_argument('--quantize-bits', type=int, default=8,
                       choices=[32, 16, 8, 4],
                       help='Number of bits for PTQ quantization (default: 8)')
    parser.add_argument('--evaluate', action='store_true',
                       help='Run inference evaluation during camera inference')
    args = parser.parse_args()
    
    device = torch.device("cpu")
    print("Using device: {}".format(device))
    
    weights_dir = os.path.join(current_dir, "weights")
    
    if args.model == 'pr_kd':
        # Create PrunedStudentModel for the specified level
        base_model = PrunedStudentModel(pruning_level=args.pr_level)
        weights_path = os.path.join(weights_dir, 'PR_student_model_{}.pth'.format(args.pr_level))
    elif args.model == 'ptq_kd':
        # Load PTQ-KD model
        student_model = StudentModel()
        weights_path = os.path.join(weights_dir, 'PTQ_student_model_{}bits.pth'.format(args.quantize_bits))
        saved_dict = torch.load(weights_path, map_location=device)
        
        base_model = QuantizedModel(student_model, saved_dict['num_bits'])
        for name, qlayer in base_model.quantized_layers.items():
            if name in saved_dict['quantization_params']:
                qlayer.quantizer.scale = saved_dict['quantization_params'][name]['scale']
                qlayer.quantizer.zero_point = saved_dict['quantization_params'][name]['zero_point']
            if name in saved_dict['state_dict']:
                qlayer.quantized_weight = saved_dict['state_dict'][name]
                module = dict(base_model.original_model.named_modules())[name]
                module.weight.data.copy_(qlayer.get_quantized_weight())
    else:
        # Load Teacher or KD model
        model_class = TeacherModel if args.model == 'teacher' else StudentModel
        base_model = model_class()
        weights_file = 'KD_teacher_model.pth' if args.model == 'teacher' else 'KD_student_model.pth'
        weights_path = os.path.join(weights_dir, weights_file)
    
    if args.model != 'ptq_kd':  # PTQ already loaded above
        state_dict = torch.load(weights_path, map_location=device)
        base_model.load_state_dict(state_dict)
    
    base_model.eval()
    recognizer = LightEmotionRecognizer(base_model, device)
    
    cap = get_camera_pipeline()
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera!")
    
    # Set camera properties
    desired_fps = 15  # Suitable for Jetson Nano
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, desired_fps)

    INFERENCE_INTERVAL = 0.2  # 5 FPS inference (0.2 second interval)
    print("Starting emotion recognition... (Inference interval: {} seconds) Press 'q' to quit.".format(INFERENCE_INTERVAL))

    last_inference_time = time.time()
    last_results = {
        'face_positions': [],
        'emotions': [],
        'confidences': []
    }

    # Move model to GPU if CUDA is available
    if torch.cuda.is_available():
        base_model = base_model.cuda()
        device = torch.device("cuda")
        print("Using CUDA acceleration")

    # Add frame skip counter
    frame_counter = 0
    PROCESS_EVERY_N_FRAMES = 2  # Process every 2nd frame

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_counter += 1
            current_time = time.time()
            
            # Apply frame skip
            if frame_counter % PROCESS_EVERY_N_FRAMES != 0:
                # Only draw boxes using last results
                frame = recognizer.draw_results(
                    frame,
                    last_results['face_positions'],
                    last_results['emotions'],
                    last_results['confidences']
                )
            else:
                should_run_inference = current_time - last_inference_time >= INFERENCE_INTERVAL
                
                if should_run_inference:
                    face_imgs, face_positions = recognizer.preprocess_frame(frame)
                    
                    if face_imgs:  # Only update if faces are detected
                        emotions = []
                        confidences = []
                        for face_img in face_imgs:
                            if torch.cuda.is_available():
                                face_img = face_img.cuda()
                            emotion, confidence = recognizer.predict_emotion(face_img)
                            emotions.append(emotion)
                            confidences.append(confidence)
                        
                        # Update results
                        last_results = {
                            'face_positions': face_positions,
                            'emotions': emotions,
                            'confidences': confidences
                        }
                    
                    last_inference_time = current_time
                
                frame = recognizer.draw_results(
                    frame,
                    last_results['face_positions'],
                    last_results['emotions'],
                    last_results['confidences']
                )
            
            cv2.imshow('Emotion Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('\nTerminating program...')
                if args.evaluate:
                    print("\nStarting model evaluation...")
                    if args.model == 'pr_kd':
                        evaluate_model(base_model, device, args.model, subtype=args.pr_level)
                    elif args.model == 'ptq_kd':
                        evaluate_model(base_model, device, args.model, subtype=str(args.quantize_bits))
                    else:
                        evaluate_model(base_model, device, args.model)
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
if __name__ == '__main__':
    main()