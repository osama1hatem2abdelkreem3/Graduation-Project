# SVM Model for Solar Panel Image Classification

This directory contains a comprehensive SVM (Support Vector Machine) model implementation for classifying solar panel images into different categories based on their condition. The model includes full evaluation metrics, visualizations, and performance analysis.

---

## 📊 Dataset Summary

The model uses the **"Solar Panels Image Enhanced"** dataset from Kaggle.

### Dataset Specifications
- **Source**: [ossossh/solar-pannels-image-enhanced](https://www.kaggle.com/ossossh/solar-pannels-image-enhanced)
- **Image Classes** (6 categories):
  1. **Clean** - Solar panels in clean condition
  2. **Dusty** - Solar panels with dust accumulation
  3. **Physical-Damage** - Panels with visible physical damage
  4. **Snow-Covered** - Panels covered with snow
  5. **Electrical-damage** - Panels with electrical faults
  6. **Bird-drop** - Panels affected by bird droppings

### Data Characteristics
- **Image Size**: 150 × 150 pixels with 3 RGB channels
- **Features Per Image**: 67,500 (150 × 150 × 3)
- **Train-Test Split**: 80% training, 20% testing (stratified split via `train_test_split`)
- **Supported Image Formats**: JPG, JPEG, PNG, BMP, TIFF, TIF (case-insensitive)
- **Data Loading**: Parallel image loading using `ThreadPoolExecutor` (6 worker threads)
- **Class Distribution**: Maintained through stratified sampling

---

## 🔧 Model Architecture

### Implementation Architecture

The model uses scikit-learn's `Pipeline` API for a reproducible, end-to-end processing workflow:

```
Raw Images (150×150×3)
    ↓
StandardScaler (Feature Normalization)
    ↓
PCA (Dimensionality Reduction: 67,500 → 340 components)
    ↓
SVM Classifier (RBF Kernel)
    ↓
Classification Output
```

### Pipeline Components

1. **StandardScaler** (Normalization)
   - Removes mean and scales features to unit variance
   - Essential for SVM performance optimization
   - Prevents features with larger scales from dominating the decision boundary

2. **PCA (Principal Component Analysis)**
   - Dimensionality reduction: 67,500 → 340 components
   - Captures 95%+ of variance with significantly reduced feature space
   - Benefits:
     - Computational efficiency and faster training
     - Reduced overfitting risk
     - Training time reduction: ~60-70%
   - Implemented with scikit-learn's `PCA(n_components=340)`

3. **SVM Classifier**
   - **Algorithm**: Support Vector Machine
   - **Kernel**: Radial Basis Function (RBF)
   - **Configuration**:
     - `C=100`: Regularization parameter (controls margin-misclassification trade-off)
     - `gamma='scale'`: Kernel coefficient (1/(n_features × X.var()))
     - `probability=True`: Enables probability estimates for ROC curve generation
     - `random_state=42`: Ensures reproducible results

## 📈 Performance Results

### Overall Accuracy
| Metric | Value |
|--------|-------|
| **Training Accuracy** | 99%+ |
| **Testing Accuracy** | ~85% |
| **Cross-Validation (5-Fold)** | Mean: ~85% (±std) |
