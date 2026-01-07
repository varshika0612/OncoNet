# Skin Lesion Classification using CNNs and Transfer Learning

**Mid-Term Project | OncoNET | WIDS 5.0**

## Overview
This repository contains the implementation and evaluation of deep learning models for skin lesion classification using dermoscopic images from the ISIC dataset. The project compares a baseline Convolutional Neural Network (CNN) trained from scratch with a transfer learning approach using a pretrained ResNet-18 model. An optional exploratory analysis of transformer-inspired concepts is also included.

The goal is to understand architectural differences in feature extraction and evaluate model performance using medically relevant metrics.

---

## Dataset
- ISIC Dermoscopic Image Dataset
- High-resolution RGB images
- Multi-class diagnostic labels


---

## Repository Structure
skin-lesion-classification/

├── notebooks/

│ ├── task1_baseline_cnn.ipynb

│ ├── task2_resnet_transfer.ipynb

│ └── task4_evaluation.ipynb

│

├── models/

│ ├── baseline_cnn_isic.pth

│ └── resnet18_isic.pth

│

├── README.md

└── requirements.txt

---

## Tasks Completed

### Task 1: Baseline CNN
- Designed and trained a CNN from scratch
- Used convolutional layers, max pooling, and fully connected layers
- Monitored training and validation performance
- Saved trained model weights

### Task 2: Transfer Learning with ResNet
- Used a pretrained ResNet-18 model
- Replaced the final classification layer to match ISIC categories
- Fine-tuned the model on the dataset
- Compared performance with the baseline CNN

### Task 4: Evaluation and Comparison
- Computed accuracy, precision, recall, F1-score, and ROC-AUC
- Generated confusion matrices using matplotlib
- Analyzed class imbalance effects on evaluation metrics
- Compared CNN vs ResNet performance

*Note:* Due to class imbalance and random splitting, one class was absent in the validation set. ROC-AUC was therefore computed in a one-vs-rest binary setting.

---

## Requirements
- Python 3.x
- PyTorch
- torchvision
- numpy
- matplotlib
- scikit-learn

Install dependencies using:
```bash```
pip install -r requirements.txt


## Results Summary

- The baseline CNN captures local texture features but has limited generalization.
- ResNet-18 benefits from pretrained hierarchical features and shows improved recall and ROC-AUC.
- Evaluation metrics beyond accuracy are crucial for medical imaging tasks.

---


## Dataset

This project uses the ISIC (International Skin Imaging Collaboration) dataset for skin lesion classification.
