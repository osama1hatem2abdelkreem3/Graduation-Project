# SVM Model for Solar Panel Image Classification

This directory contains an SVM (Support Vector Machine) model implementation for classifying solar panel images into different categories based on their condition (e.g., Clean, Dusty, Physical-Damage, etc.).

## Dataset

The model uses the "Solar Panels Image Enhanced" dataset from Kaggle

- **Source**: [ossossh/solar-pannels-image-enhanced](https://www.kaggle.com/ossossh/solar-pannels-image-enhanced)
- **Categories**: Physical-Damage, Snow-Covered, Electrical-damage, Clean, Dusty, Bird-drop

## Pipeline Overview

The model follows a preprocessing and classification pipeline:

1. **Data Loading**:
   - Images are loaded from the dataset directories.
   - Each image is resized to 150x150 pixels with 3 color channels (RGB).
   - Images are flattened into 1D arrays (67500 features per image).
   - Supported formats: JPG, JPEG, PNG, BMP, TIFF, TIF (case-insensitive).

2. **Data Preparation**:
   - Images are converted into a Pandas DataFrame with features and target labels.
   - Target labels are integer-encoded based on category index.

3. **Train-Test Split**:
   - Data is split into training (80%) and testing (20%) sets using stratified sampling to maintain class distribution.
   - Random state: 77

4. **Preprocessing Pipeline**:
   - **StandardScaler**: Standardizes features by removing the mean and scaling to unit variance.
   - **PCA (Principal Component Analysis)**: Reduces dimensionality from 67500 to 340 components to improve computational efficiency and reduce overfitting.

5. **Model Training**:
   - **SVM Classifier**: Support Vector Machine with RBF kernel.
   - Parameters:
     - C: 100 (regularization parameter)
     - gamma: 'scale' (kernel coefficient, scaled by 1/(n_features * X.var()))
     - kernel: 'rbf' (Radial Basis Function)
     - probability: True (enables probability estimation)
     - random_state: 42

6. **Evaluation**:
   - Accuracy score on the test set.

## Performance

- Test accuracy: 85% 
