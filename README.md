## FER-Jetson-Nano
젯슨 나노 환경에서 동작하는 Facial Expression Recognition(FER) 프로젝트

<br>

## 폴더 구조
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
└── requirements.yml                   # 가상 환경 설정 파일
```

<br>

## Requirements
```bash
conda env create --file environment.yaml
```

Jetson-Nano (Linux) Python3 Setting:
- python=3.6.9
- torch=0.8.1
- ...

<br>

## 데이터셋
학습에 사용한 48x48 사이즈로 전처리된 CK+ 데이터셋은 [여기](https://www.kaggle.com/datasets/shuvoalok/ck-dataset)에서 다운로드 받을 수 있습니다.

<br>

## Inference 스크립트
Knowledge Distillation(baseline) model Inference commands
```bash
python "./inference.py" --model kd       # Student
python "./inference.py" --model teacher  # Teacher
```

<br>

Prunned Student model Inference commands
```bash
python "./inference.py" --model pr_kd --pr-level medium

# choice = ['low', 'medium', 'high']
```

<br>

Quantized Student model Inference commands
```bash
python "./inference.py" --model ptq_kd --quantize-bits 8

# choice = [32, 16, 6, 4]
```

If you want to save results csv file, add `--evaluate` at the end
```bash
python "./inference.py" --model ptq_kd --quantize-bits 8 --evaluate
```

<br>

## 결과
![total_comparison](https://github.com/user-attachments/assets/e3c96b45-5de3-40f5-8417-d50ea8cf4cc9)
