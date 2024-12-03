# FER-Jetson-Nano


## 폴더 구조
```bash
FER-Jetson-Nano                        

├── ck+                                # CK+ 데이터셋 폴더
│
├── data                               # 데이터 로드 및 전처리를 위한 스크립트를 포함
│   └── ckplus_dataset.py              # CK+ 데이터셋 로드, 전처리 및 형식화를 처리
│
├── evaluate                           # 모델 성능 평가를 위한 스크립트를 포함
│   ├── inference_evaluator.py         # 추론 단계에서 모델의 성능(Inference Time 등)을 평가
│   └── model_evaluator.py             # 모델의 일반적인 성능(모델 사이즈, FLOPs 등)을 평가
│
├── results                            # 평가 결과 csv 파일을 저장
│
├── training                           # 경량 FER 모델을 학습시키는 스크립트를 포함
│   ├── KD_training.py                 # Knowledge Distillation(KD)를 이용한 학습
│   ├── PR_KD_training.py              # Student 모델에 Pruning(PR)을 결합한 학습
│   └── PTQ_KD_training.py             # Student 모델에 Post-Training Quantization(PTQ)을 결합한 학습
│
├── visualize                          # 시각화를 생성하는 스크립트를 포함
│   └── visualization.py               # results의 시각화 이미지 생성
│
└── weights                            # Jetson Nano에서 추론에 사용될 학습된 모델 가중치를 저장

```


