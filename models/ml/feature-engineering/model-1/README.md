# Summary of Work: SVM with GLCM Features

## Overview
This project focuses on classifying solar panel defects using texture-based features extracted from images. The approach combines Gray Level Co-occurrence Matrix (GLCM) for feature extraction and Support Vector Machine (SVM) for classification.

## Dataset

The model uses the "Solar Panels Image Enhanced" dataset from Kaggle

- **Source**: [ossossh/solar-pannels-image-enhanced](https://www.kaggle.com/ossossh/solar-pannels-image-enhanced)
- **Categories**: Physical-Damage, Snow-Covered, Electrical-damage, Clean, Dusty, Bird-drop

## GLCM Features
GLCM is a statistical method that analyzes the spatial relationship between pixels in an image. From the GLCM, six texture properties are calculated:

1. **Contrast**: Measures the intensity contrast between a pixel and its neighbors. High values indicate sharp edges or distinct textures.
2. **Dissimilarity**: Measures the variation in gray-level pairs. High values indicate more variation in texture.
3. **Homogeneity**: Measures the closeness of the distribution of elements in the GLCM to the GLCM diagonal. High values indicate smoother textures.
4. **Energy**: Represents the sum of squared elements in the GLCM (uniformity). High values indicate uniform textures.
5. **Correlation**: Measures the linear dependency of gray levels in the GLCM. High values indicate strong linear relationships in pixel intensities.
6. **ASM (Angular Second Moment)**: Measures the uniformity of the GLCM. High values indicate regular, repetitive patterns.

These six properties are computed for four angles (0°, 45°, 90°, 135°) and one distance, resulting in **24 features per image**.

## Pipeline Overview
1. **Data Preparation**:
   - Images are resized to a consistent shape.
   - Converted to grayscale for texture analysis.

2. **Feature Extraction**:
   - GLCM features are extracted for each image using the `extract_glcm_features` function.
   - The extracted features are stored in a dataset along with their corresponding labels.

3. **Model Training**:
   - The dataset is split into training and testing sets.
   - An SVM classifier with an RBF kernel is trained on the GLCM features.

4. **Evaluation**:
   - The model is evaluated on the test set, achieving a high accuracy.

## Performance
- **Test Accuracy**: 68%
