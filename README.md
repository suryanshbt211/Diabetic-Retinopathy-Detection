Diabetic Retinopathy Detection using Deep Learning
==================================================

## Project Overview

This project investigates the use of **deep learning–based computer vision techniques** for the **automatic detection and severity classification of Diabetic Retinopathy (DR)** from retinal fundus images. Diabetic Retinopathy is a diabetes-related ocular disease that progressively damages the retina and can lead to irreversible blindness if not detected at an early stage.

The work is motivated by the limitations of manual screening procedures, which are **time-consuming, subjective, and difficult to scale** in regions with limited access to ophthalmologists. The objective of this project is to design and evaluate a **robust, reproducible, and scalable machine learning pipeline** that can assist in large-scale screening by classifying retinal images into clinically defined DR severity levels.

* * *

## Problem Statement

Diabetic Retinopathy screening traditionally relies on expert interpretation of fundus images. This process faces several challenges:

- Lack of early symptoms in initial stages of the disease  
- High dependence on trained specialists  
- Inconsistency due to subjective interpretation  
- Limited scalability for population-level screening  

From a machine learning perspective, the problem is formulated as a **multi-class image classification task** with five severity levels. The task is further complicated by **class imbalance**, **subtle visual differences between stages**, and the **ordinal nature of disease progression**.

* * *

## Dataset Description

- **Dataset**: APTOS 2019 Blindness Detection (Kaggle)  
- **Training samples**: 3,662 retinal fundus images  
- **Test samples**: 1,928 retinal fundus images  
- **Classes**:
  - 0 – No Diabetic Retinopathy  
  - 1 – Mild DR  
  - 2 – Moderate DR  
  - 3 – Severe DR  
  - 4 – Proliferative DR  
- **Image type**: High-resolution color fundus photographs  

The dataset reflects real clinical distributions and is **highly imbalanced**, with a majority of samples belonging to the “No DR” class. This characteristic was preserved to ensure realistic model evaluation.

* * *

## Methodology

### Data Preprocessing

- Image normalization (pixel values scaled to [0,1])  
- Resizing to a fixed resolution (728 × 728 × 3)  
- Removal of invalid or corrupted images  
- Label formatting for categorical learning  

### Data Augmentation

- Horizontal flipping  
- Train–validation split (80% / 20%)  
- Augmentation used as a regularization strategy to improve generalization  

* * *

## Model Architectures

The project evaluates and compares multiple **transfer learning–based CNN architectures**, all pretrained on ImageNet and fine-tuned on the DR dataset.

### 1. ResNet50

- Residual connections to mitigate vanishing gradients  
- Global Average Pooling followed by fully connected layers  
- Used as a baseline deep CNN architecture  

### 2. ResNet50 + VGG16 (Ensemble)

- Combines residual learning with deep convolutional feature extraction  
- Demonstrates improved convergence and validation accuracy over the baseline  

### 3. ResNet50 + Xception (Final Model)

- Xception architecture employs depthwise separable convolutions  
- More effective at capturing fine-grained retinal features  
- Achieved the best overall performance among evaluated models  

* * *

## Training Strategy

- **Transfer Learning**: ImageNet-pretrained weights  
- **Two-Phase Training**:
  - Warm-up phase with frozen backbone layers and higher learning rate  
  - Fine-tuning phase with unfrozen layers and reduced learning rate  
- **Optimization Techniques**:
  - Adam optimizer  
  - Early stopping to prevent overfitting  
  - ReduceLROnPlateau for adaptive learning rate scheduling  
- **Reproducibility**:
  - Fixed random seeds for Python, NumPy, and TensorFlow  

* * *

## Evaluation Metrics

Model performance was evaluated using:

- Accuracy  
- Precision, Recall, and F1-Score (class-wise)  
- Confusion Matrix  
- **Quadratic Cohen’s Kappa Score**, which is particularly relevant for ordinal classification problems such as disease severity grading  

* * *

## Results

| Model | Accuracy |
|------|----------|
| ResNet50 | 0.49 |
| ResNet50 + VGG16 | 0.66 |
| ResNet50 + Xception | **0.72** |

The ensemble of ResNet and Xception demonstrated superior generalization, reduced validation loss, and improved classification consistency compared to single-model approaches.

* * *

## Tools and Technologies Used

### Programming Language

- Python  

### Deep Learning & ML Frameworks

- TensorFlow  
- Keras  

### Data Processing & Visualization

- NumPy  
- Pandas  
- OpenCV  
- Matplotlib  
- Seaborn  

### Evaluation & Utilities

- scikit-learn  
- Confusion matrix and Cohen’s Kappa metrics  

### Development Environment

- Google Colab (GPU-accelerated)  

* * *

## Limitations

- Class imbalance affects recall for severe DR stages  
- Categorical cross-entropy does not explicitly model ordinal relationships  
- High-resolution images increase computational cost  
- Model is not optimized for deployment on edge or mobile devices  

* * *

## Future Scope

- Incorporation of ordinal regression or custom loss functions  
- Class-weighted or focal loss to address imbalance  
- Lightweight architectures (e.g., MobileNet, EfficientNet)  
- Attention mechanisms for lesion localization  
- Training on larger, multi-institutional datasets  

* * *

## Conclusion

This project demonstrates a **systematic, research-oriented application of deep learning** to a clinically relevant medical imaging problem. Through architectural comparison, transfer learning, and careful evaluation, the work highlights both the potential and limitations of CNN-based DR detection systems. The project emphasizes **methodology, reproducibility, and evaluation rigor**, aligning with best practices in applied machine learning research.

