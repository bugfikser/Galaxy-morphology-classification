# Galaxy Morphology Classification with Deep Learning

This project focuses on classifying **spiral vs elliptical galaxies** using deep learning on real astronomical survey data.

Cosmology and physics are my core interests, and this project was built to explore how modern machine learning techniques can be applied to real astrophysical problems — end to end, without shortcuts.

---

## 🚀 Project Overview

- Binary classification: **Spiral vs Elliptical galaxies**
- Real sky images from **DECaLS**
- Labels derived from **Galaxy Zoo probabilistic annotations**
- CNN built **from scratch in PyTorch**
- Proper ML workflow with strict train / validation / test separation

Final **test accuracy: ~91–92% on unseen data**

---

## 📊 Data & Labeling

- Galaxy coordinates (RA/DEC) sourced from curated Galaxy Zoo tables
- Labels constructed using **probability thresholds**, not raw votes
- Only high-confidence galaxies used to reduce label noise
- Image cutouts retrieved dynamically via RA/DEC queries

Broken images and failed downloads were explicitly handled and excluded.

---

## 🧠 Model Architecture

- Custom Convolutional Neural Network (CNN)
- Three convolution blocks with ReLU + MaxPooling
- Sigmoid output for binary classification
- Trained using Binary Cross-Entropy loss

No pre-trained models were used in the baseline.

---

## 🏗️ Training Pipeline

- Image resizing to 256×256
- Train / validation split (80 / 20)
- Early stopping with patience
- Accuracy and loss tracking per epoch
- Best model checkpoint saved based on validation accuracy

Training speed is fast due to dataset size and task simplicity — convergence occurs early, as expected.

---

## 🧪 Evaluation Strategy

- **Strictly unseen test set**
- Test data never used for training or tuning
- Final evaluation performed once
- Confusion matrix and error analysis used to understand failure modes

Observed errors mostly occur for:
- Edge-on disk galaxies
- Weak spiral arms
- Low signal-to-noise images
- Ambiguous morphologies

---

## 📈 Results

| Metric | Value |
|------|------|
| Validation Accuracy | ~0.89–0.91 |
| Test Accuracy | ~0.91–0.92 |

Validation and test performance closely match, indicating good generalization and no data leakage.

---

## 🔍 Key Learnings

- Accuracy alone is insufficient — error analysis is critical
- Data quality and labeling decisions matter more than model depth
- Small, clean datasets can outperform larger noisy ones
- ML performance must be interpreted in physical context

This project reinforced my interest in working at the intersection of **cosmology, physics, and machine learning**.

---

## ▶️ How to Run

### Install dependencies
```bash
pip install -r requirements.txt

Train model
python train.py

Evaluate on unseen test data
python test.py


