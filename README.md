# 🫁 Pneumonia Detection System (P2M Project)

## 📌 Overview

This project is a **multi-agent AI system** designed for the detection and interpretation of pneumonia from chest X-ray images.  
It combines deep learning (DenseNet-121) and explainable AI (Grad-CAM) with a clinical reasoning module to improve diagnostic accuracy and interpretability.

The system is based on a **Chest X-Ray Pneumonia dataset from Kaggle** containing approximately 5,856 images labeled as:
- NORMAL
- PNEUMONIA

---

## 🧠 System Architecture

The project uses a **multi-agent approach**:

### 🧑‍⚕️ Agent 1: Vision Expert
- Analyzes chest X-ray images using **DenseNet-121**
- Predicts probability of pneumonia
- Generates a structured radiology report
- Uses **Grad-CAM** to highlight important regions in the image
- Identifies affected lung quadrant (if applicable)

### 🧑‍⚕️ Agent 2: Diagnostic Expert
- Receives the radiology report from Agent 1
- Combines image-based findings with patient symptoms
- Produces the final medical diagnosis

---

## 🧪 Dataset

- Source: Kaggle Chest X-Ray Pneumonia Dataset
- Total images: ~5,856
- Classes:
  - NORMAL (27%)
  - PNEUMONIA (73%)
- Split:
  - Training set
  - Validation set
  - Test set

---

## ⚙️ Preprocessing

- Resize images to **224 × 224**
- Normalize using **ImageNet statistics**
- Data augmentation (training only):
  - Horizontal flipping
  - Brightness adjustment

---

## 🏗️ Model Details

- Architecture: **DenseNet-121**
- Optimizer: **Adam**
- Training: Supervised learning
- Regularization:
  - Early stopping
  - Dropout layers
  - Class weighting (to handle imbalance)

---

## 📊 Evaluation Metrics

Due to dataset imbalance, the following metrics are used instead of accuracy:
- Recall
- F1-score
- AUC (Area Under Curve)

---

## 🔍 Explainability

The system uses **Grad-CAM** to:
- Visualize regions influencing predictions
- Improve model interpretability
- Assist clinical understanding

---

## 🚀 Key Features

- Multi-agent medical reasoning system
- Deep learning-based pneumonia detection
- Explainable AI using Grad-CAM
- Anatomical localization of infection
- Robust handling of class imbalance

---

## 📦 Installation

```bash
git clone https://github.com/your-username/p2m-project.git
cd p2m-project
pip install -r requirements.txt
