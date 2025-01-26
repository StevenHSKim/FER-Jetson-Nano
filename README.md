## FER-Jetson-Nano
<div align="center">
  <img src="https://github.com/user-attachments/assets/1a24c2d4-1969-4818-8fc1-49a10c95769c" alt="fer-jetson-nano-demo">
</div>


Facial Expression Recognition (FER) is a technology that automatically detects and classifies various facial expressions. It plays a crucial role in applications such as Human-Computer Interaction and emotion analysis.

In this project, we implemented an FER model on the NVIDIA Jetson Nano embedded system and applied various optimization techniques to compare and analyze performance on the Jetson Nano.

This task aims to classify a total of seven emotion classes: Anger, Disgust, Fear, Happiness, Sadness, Surprise, and Contempt.

<br>

## Folder Structure
```bash
FER-Jetson-Nano                        

├── ck+                                # CK+ dataset folder  
│  
├── data/                              # Scripts for data loading and preprocessing  
│   └── ckplus_dataset.py              # Handles loading, preprocessing, and formatting of the CK+ dataset  
│  
├── evaluate/                          # Scripts for evaluating model performance  
│   ├── inference_evaluator.py         # Evaluates performance (e.g., inference time) during inference  
│   └── model_evaluator.py             # Evaluates general performance (e.g., model size, FLOPs)  
│  
├── results/                           # Stores evaluation result CSV files  
│  
├── training/                          # Scripts for training lightweight FER models  
│   ├── KD_training.py                 # Training using Knowledge Distillation (KD)  
│   ├── PR_KD_training.py              # Applies Pruning (PR) to the Student model  
│   └── PTQ_KD_training.py             # Applies Post-Training Quantization (PTQ) to the Student model  
│  
├── visualize/                         # Scripts for generating visualizations  
│   └── visualization.py               # Generates visualization images from results  
│  
├── weights/                           # Stores trained model weights for inference  
│  
├── inference.py                       # Main script for performing inference  
└── environment.yml                    # Virtual environment configuration file  

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
The CK+ dataset preprocessed to a size of 48x48 for training can be downloaded [here](https://www.kaggle.com/datasets/shuvoalok/ck-dataset).

<br>

## Inference Scripts
Knowledge Distillation model Inference commands
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

# choice = [32, 16, 8, 4]
```

If you want to save results csv file, add `--evaluate` at the end
```bash
python3 "./inference.py" --model ptq_kd --quantize-bits 8 --evaluate
```

<br>

## Results
![image](https://github.com/user-attachments/assets/44951f33-26ec-4604-99c3-e043c7612d8f)


<br>

## License
This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
