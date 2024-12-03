## FER-Jetson-Nano
Facial Expression Recognition(FER)은 다양한 얼굴 표정을 자동으로 인식하고 분류하는 기술로, 인간-컴퓨터 상호작용(Human-Computer Interaction) 및 감정 분석 등의 응용 분야에서 중요한 역할을 합니다. 

본 프로젝트에서는 NVIDIA Jetson Nano 임베디드 시스템 환경에서 FER 모델을 구현하고 다양한 경량화 기법을 적용하여 Jetson Nano에서의 성능을 비교 및 분석하였습니다. 

해당 테스크는 Anger, Disgust, Fear, Happiness, Sadness, Surprise, Contempt로 구성된 총 7개의 감정 클래스로 분류하는 것을 목표합니다.

<br>

## Folder Structure
```bash
FER-Jetson-Nano                        

├── ck+                                # CK+ 데이터셋 폴더
│
├── data/                              # 데이터 로드 및 전처리를 위한 스크립트를 포함
│   └── ckplus_dataset.py              # CK+ 데이터셋 로드, 전처리 및 형식화를 처리
│
├── evaluate/                          # 모델 성능 평가를 위한 스크립트를 포함
│   ├── inference_evaluator.py         # 추론 단계에서 모델의 성능(Inference Time 등)을 평가
│   └── model_evaluator.py             # 모델의 일반적인 성능(모델 사이즈, FLOPs 등)을 평가
│
├── results/                           # 평가 결과 csv 파일을 저장
│
├── training/                          # 경량 FER 모델을 학습시키는 스크립트를 포함
│   ├── KD_training.py                 # Knowledge Distillation(KD)를 이용한 학습
│   ├── PR_KD_training.py              # Student 모델에 Pruning(PR)을 적용
│   └── PTQ_KD_training.py             # Student 모델에 Post-Training Quantization(PTQ)을 적용
│
├── visualize/                         # 시각화를 생성하는 스크립트를 포함
│   └── visualization.py               # results의 시각화 이미지 생성
│
├── weights/                           # 추론에 사용될 학습된 모델 가중치를 저장
│
├── inference.py                       # 추론 수행을 위한 메인 스크립트
└── environment.yml                    # 가상 환경 설정 파일
```

<br>

## Requirements
```bash
conda env create --file environment.yml
```

Jetson-Nano (Linux) Python3 Setting:
- python=3.6.9
- torch=0.8.1
- ...

<br>

## Dataset
학습에 사용한 48x48 사이즈로 전처리된 CK+ 데이터셋은 [여기](https://www.kaggle.com/datasets/shuvoalok/ck-dataset)에서 다운로드 받을 수 있습니다.

<br>

## Inference Scripts
Knowledge Distillation(baseline) model Inference commands
```bash
python3 "./inference.py" --model kd       # Student
python3 "./inference.py" --model teacher  # Teacher
```

<br>

Prunned Student model Inference commands
```bash
python3 "./inference.py" --model pr_kd --pr-level medium

# choice = ['low', 'medium', 'high']
```

<br>

Quantized Student model Inference commands
```bash
python3 "./inference.py" --model ptq_kd --quantize-bits 8

# choice = [32, 16, 6, 4]
```

If you want to save results csv file, add `--evaluate` at the end
```bash
python3 "./inference.py" --model ptq_kd --quantize-bits 8 --evaluate
```

<br>

## Results
![total_comparison](https://github.com/user-attachments/assets/e3c96b45-5de3-40f5-8417-d50ea8cf4cc9)

<br>

## License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
