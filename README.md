# MRI-Alzheimer-Classification
# Brain MRI Alzheimer's Classification

A deep learning framework for classifying Alzheimer's disease stages from brain MRI scans using 2D and 3D convolutional neural networks.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Overview

This project implements an end-to-end pipeline for Alzheimer's disease classification from brain MRI images. It supports multiple datasets, preprocessing modes, and model architectures for both 2D slice-based and 3D volumetric analysis.

### Key Features

- **Multi-dataset support**:  OASIS-1, Kaggle Alzheimer's Dataset, Kaggle NIfTI
- **Flexible preprocessing**: Convert 3D volumes to 2D slices or use full 3D volumes
- **Multiple architectures**: ResNet18/50, EfficientNet-B0, DenseNet121 (2D) and custom 3D CNNs
- **Model interpretability**: Grad-CAM visualizations for clinical insight
- **Class imbalance handling**: Weighted loss functions for imbalanced medical data

### Classification Task

The model classifies brain MRI scans into 4 stages based on Clinical Dementia Rating (CDR):

| Class | CDR Score | Description |
|-------|-----------|-------------|
| 0 | CDR = 0 | NonDemented |
| 1 | CDR = 0.5 | VeryMildDemented |
| 2 | CDR = 1 | MildDemented |
| 3 | CDR ≥ 2 | ModerateDemented |

## Results

### Model Performance Summary

| Dataset | Model | Test Accuracy | Best AUC | Early-Stage Recall | Overfitting |
|---------|-------|---------------|----------|-------------------|-------------|
| OASIS 2D | ResNet50 | 63.7% | 0.854 (NonDem) | 33.6% | Mild ⚠️ |
| OASIS 3D | Simple3DCNN | **68.1%** | 0.882 (Mild) | - | None ✅ |
| Kaggle 2D | ResNet50 | 69.5% | 0.857 (NonDem) | - | Severe 🔴 |

### 2D Model Comparison (OASIS Dataset)

| Model | Test Acc | NonDem Recall | VeryMild Recall | Mild Recall |
|-------|----------|---------------|-----------------|-------------|
| ResNet50 | 63.7% | 79.1% | 33.6% | 65.0% |
| EfficientNet-B0 | 63.4% | 64.3% | **68.2%** | 48.3% |
| DenseNet121 | 60.0% | 65.0% | 55.4% | 48.3% |
| ResNet18 | 59.2% | 76.9% | 21.1% | 68.3% |

**Key Findings:**
- 3D CNN shows best generalization (no overfitting) with 68% accuracy
- EfficientNet-B0 achieves best early-stage (VeryMildDemented) recall at 68%
- Early-stage detection is clinically most valuable for intervention

### Grad-CAM Visualization

The model focuses on clinically relevant brain regions, particularly the hippocampus and medial temporal lobe — areas known to show early atrophy in Alzheimer's disease. 

<img width="327" height="115" alt="image" src="https://github.com/user-attachments/assets/a9ba889f-0752-4a6e-b083-ae47141a5f20" />


## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/Cecelia1122/MRI-Alzheimer-Classification.git
cd MRI-Alzheimer-Classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
Pillow>=9.5.0
tqdm>=4.65.0
opencv-python>=4.8.0
nibabel>=5.0.0
nilearn>=0.10.0
scipy>=1.11.0
monai>=1.3.0
```

## Dataset Preparation

### Supported Datasets

| Dataset | Format | Source |
|---------|--------|--------|
| OASIS-1 | Analyze 7. 5 (. img/. hdr) | [oasis-brains.org](https://www.oasis-brains.org/) |
| Kaggle Alzheimer's | Preprocessed JPG | [Kaggle](https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images) |
| Kaggle NIfTI | NIfTI (. nii/. nii.gz) | Kaggle |

### Directory Structure

```
data/
├── oasis/                    # Raw OASIS data
│   ├── disc1/
│   │   ├── OAS1_0001_MR1/
│   │   └── ...
│   └── disc12/
├── oasis_2d/                 # Processed 2D slices
│   ├── train/
│   │   ├── NonDemented/
│   │   ├── VeryMildDemented/
│   │   ├── MildDemented/
│   │   └── ModerateDemented/
│   └── test/
├── oasis_3d/                 # Processed 3D volumes
│   ├── train/
│   └── test/
├── Kaggle/                   # Original Kaggle images
└── kaggle_2d/                # Split Kaggle images
```

### Preprocessing Commands

```bash
# OASIS → 2D slices (20 slices per subject)
python main.py --mode preprocess --dataset oasis --dim 2d

# OASIS → 3D volumes (128x128x128)
python main.py --mode preprocess --dataset oasis --dim 3d

# Kaggle NIfTI → 2D slices
python main.py --mode preprocess --dataset kaggle_nifti --dim 2d

# Kaggle NIfTI → 3D volumes
python main.py --mode preprocess --dataset kaggle_nifti --dim 3d

# Split original Kaggle JPGs into train/test
python main.py --mode preprocess --dataset kaggle_orig
```

## Usage

### Training

```bash
# Train 2D model on OASIS
python main.py --mode train --dataset oasis_2d --backbone resnet50 --epochs 30

# Train 3D model on OASIS
python main.py --mode train --dataset oasis_3d --backbone simple --batch_size 4

# Train with custom parameters
python main.py --mode train \
    --dataset oasis_2d \
    --backbone efficientnet_b0 \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-4
```

### Evaluation

```bash
# Evaluate trained model
python main.py --mode evaluate --dataset oasis_2d

# Evaluate specific checkpoint
python main.py --mode evaluate --dataset oasis_2d --checkpoint checkpoints/best_model.pth
```

### Grad-CAM Visualization

```bash
# Generate Grad-CAM visualizations (2D models only)
python main.py --mode gradcam --dataset oasis_2d
```

### Full Pipeline

```bash
# Run preprocessing, training, evaluation, and Grad-CAM
python main. py --mode full --dataset oasis_2d
```

## Project Structure

```
brain-mri-alzheimers/
├── main.py                 # Unified entry point
├── debug. py                # Component testing script
├── requirements.txt        # Dependencies
├── README.md
├── src/
│   ├── __init__.py         # Package initialization
│   ├── preprocessing.py    # Data preprocessing utilities
│   ├── dataset.py          # Kaggle dataset loader
│   ├── dataset_oasis.py    # OASIS 2D dataset loader
│   ├── dataset_3d.py       # 3D volume dataset loader
│   ├── model.py            # 2D CNN architectures
│   ├── model_3d.py         # 3D CNN architectures
│   ├── train.py            # Training loop
│   ├── evaluate. py         # Evaluation metrics
│   └── gradcam.py          # Grad-CAM visualization
├── checkpoints/            # Saved models
├── results/
│   ├── figures/            # Training curves, confusion matrices
│   └── gradcam/            # Grad-CAM visualizations
└── data/                   # Datasets (not included)
```

## Model Architectures

### 2D Models (Transfer Learning)

| Backbone | Parameters | Pretrained | Input Size |
|----------|------------|------------|------------|
| ResNet18 | ~11M | ImageNet | 224×224×3 |
| ResNet50 | ~25M | ImageNet | 224×224×3 |
| EfficientNet-B0 | ~5M | ImageNet | 224×224×3 |
| DenseNet121 | ~8M | ImageNet | 224×224×3 |

### 3D Models (From Scratch)

| Model | Parameters | Input Size |
|-------|------------|------------|
| Simple3DCNN | ~2M | 128×128×128×1 |
| ResNet3D | ~10M | 128×128×128×1 |
| ResNet3D-Small | ~3M | 128×128×128×1 |

## API Usage

```python
from src import (
    create_model,
    create_data_loaders,
    Trainer,
    Evaluator,
    GradCAM,
    get_target_layer
)

# Create model
model = create_model(model_dim='2d', backbone='resnet50', pretrained=True)

# Create data loaders
train_loader, test_loader = create_data_loaders(
    data_dir='./data/oasis_2d',
    dataset_type='oasis',
    model_dim='2d',
    batch_size=32
)

# Train
trainer = Trainer(model, train_loader, test_loader, device='cuda')
history = trainer.train(num_epochs=30)

# Evaluate
evaluator = Evaluator(model, test_loader, device='cuda')
results = evaluator.full_evaluation(save_dir='./results')

# Grad-CAM
target_layer = get_target_layer(model, 'resnet50')
gradcam = GradCAM(model, target_layer, device='cuda')
cam = gradcam.generate_cam(input_tensor)
```

## Debugging

Run component tests individually:

```bash
# Test all components
python debug.py --step all

# Test specific component
python debug.py --step 1  # Preprocessing
python debug.py --step 2  # Datasets
python debug.py --step 3  # Models
python debug.py --step 4  # Training
python debug. py --step 5  # Evaluation
python debug.py --step 6  # Grad-CAM
```

## Technical Notes

### Why 64-68% Accuracy?

This is a challenging 4-class medical imaging problem: 
- **Random baseline**: 25% (4 classes)
- **Our best**:  68% (2. 7× better than random)
- **VeryMildDemented (CDR 0.5)** is extremely difficult to distinguish from healthy aging
- Even expert radiologists struggle with early-stage differentiation

### Class Imbalance

The OASIS dataset has significant class imbalance: 
- NonDemented: ~70%
- VeryMildDemented: ~20%
- MildDemented: ~8%
- ModerateDemented: ~2%

We address this with weighted cross-entropy loss computed dynamically from training data.

### 3D vs 2D

| Approach | Pros | Cons |
|----------|------|------|
| 2D Slices | Transfer learning, faster training | Loses 3D spatial context |
| 3D Volumes | Captures bilateral atrophy patterns | No pretrained weights, memory intensive |

## Future Improvements

- [ ] Vision Transformers (ViT) for medical imaging
- [ ] Multi-task learning with clinical variables (age, MMSE)
- [ ] Cross-dataset validation (OASIS ↔ ADNI)
- [ ] Attention mechanisms for better feature localization
- [ ] Uncertainty quantification for clinical deployment

## References

- [OASIS:  Cross-Sectional MRI Data](https://www.oasis-brains.org/)
- [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391)
- [Deep Learning for Alzheimer's Disease Classification](https://doi.org/10.1016/j.neurobiolaging.2019.01.010)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Xue Li**  
MSc Communications and Signal Processing, Imperial College London  
Email: xueli.xl1122@gmail.com

## Acknowledgments

- OASIS project for providing open-access brain MRI data
- PyTorch team for the deep learning framework
- nibabel developers for neuroimaging file I/O
