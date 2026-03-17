# Final Assignment – Machine Learning
**Multiclass Facial Identification with CNNs and Transfer Learning (CelebA)**

## Problem Description

This project addresses a **supervised image classification** problem focused on **facial identity recognition** using a subset of the **CelebA (CelebFaces Attributes Dataset)**.

The objective is to correctly identify the **identity** associated with a facial image among a large number of possible classes. Unlike the partial assignment, which relied on classical feature extraction techniques, this final project explores **deep learning approaches**, specifically **Convolutional Neural Networks (CNNs)** and **transfer learning**.

The task is formulated as a **multiclass classification problem**, where each class corresponds to a distinct individual.

---

## Dataset

- **Dataset:** https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- **Subset configuration:**
  - 2000 identities
  - 30 images per identity
- **Total images used:** ~60,000

### Annotations used:
- `identity_CelebA.txt` (image → identity mapping)

A custom script was developed to:
- Automatically group images by identity
- Select a balanced subset
- Split data into train/validation/test sets
- Generate a `split.csv` file for reproducibility

---

## Methodology

The pipeline consists of the following steps:

1. **Subset Creation**
   - Selection of K identities
   - Selection of M images per identity
   - Train/validation/test split

2. **Preprocessing**
   - Image resizing (224×224)
   - Normalization using ImageNet statistics
   - Data augmentation (horizontal flip)

3. **Modeling**
   - Baseline CNN (trained from scratch)
   - ResNet18 pre-trained on ImageNet

4. **Training Strategy**
   - Phase 1: Train classification head
   - Phase 2: Fine-tune deeper layers

5. **Evaluation**
   - Top-1 accuracy
   - Top-5 accuracy
   - Confusion matrix

---

## Implemented Models

### 1. Baseline CNN
- Simple architecture
- Trained from scratch
- Used as performance reference

### 2. ResNet18 (Transfer Learning)
- Pre-trained on ImageNet
- Final layer adapted to 2000 classes
- Partial fine-tuning

---

## Results

| Model | Test Loss | Top-1 Accuracy | Top-5 Accuracy |
|------|----------|---------------|---------------|
| CNN Baseline | ~7.60 | ~0.05% | — |
| ResNet18 | ~2.30 | ~55% | ~74% |

The baseline model failed to learn meaningful representations, while the transfer learning approach achieved strong performance.

---

## Key Insights

- Training from scratch is insufficient for large multiclass problems
- Transfer learning significantly improves performance
- Pre-trained models capture useful visual features
- Fine-tuning enhances task-specific adaptation

---

## Project Structure

```
.
├── src/
│   ├── make_subset.py
│   ├── train_cnn_torch.py
│   ├── train_resnet18_celeba.py
├── resources/
│   └── celebA_subset/
├── results/
│   ├── cnn_torch_baseline/
│   └── resnet18_pretrained/
└── README.md
```

---

## Reproducibility

- Fixed random seed
- Automatic dataset generation
- Saved models and logs
- CSV-based dataset split

---

## Dependencies

- Python ≥ 3.10
- PyTorch
- torchvision
- NumPy
- pandas
- scikit-learn
- matplotlib

---

## How to Run

### 1. Create subset
```bash
python src/make_subset.py
```

### 2. Train baseline CNN
```bash
python src/train_cnn_torch.py
```

### 3. Train ResNet18
```bash
python src/train_resnet18_celeba.py
```

---

## Notes

- GPU (CUDA) is highly recommended for training
- Training on CPU is possible but significantly slower
