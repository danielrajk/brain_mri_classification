🧠 NeuroScan – Brain MRI Disease Classification with Explainable AI
Project Overview

NeuroScan is an intelligent medical image analysis system designed to automatically detect and classify brain diseases from Magnetic Resonance Imaging (MRI) scans using deep learning techniques. The system leverages transfer learning with EfficientNet to achieve accurate multi-class classification and integrates Explainable AI (XAI) using Grad-CAM to improve model interpretability and clinical trust.

This project aims to assist radiologists and medical professionals by providing a reliable, fast, and explainable decision-support tool for brain MRI analysis.

Objectives

To design a deep learning–based system for automated brain MRI disease classification
To apply transfer learning for improved accuracy on limited medical datasets
To handle class imbalance using class weighting
To provide explainable predictions using Grad-CAM visualization
To deploy the trained model using a Streamlit web application

Dataset Description

The project uses a publicly available Brain MRI dataset consisting of four classes:

Glioma
Meningioma
Pituitary Tumor
No Tumor

All images were consolidated into a raw dataset and manually partitioned into training, validation, and testing sets using a 70–15–15 split to avoid data leakage and ensure unbiased evaluation.

Dataset structure:

dataset/
├── train/
├── val/
└── test/


Each split contains class-wise folders with MRI images.

Methodology

Data Preprocessing

Image resizing to 224×224

Pixel normalization

Data augmentation (rotation, brightness variation, horizontal flip)

Model Architecture

EfficientNetB0 as the base feature extractor

Global Average Pooling

Batch Normalization

Fully connected layers with dropout

Softmax output for multi-class classification

Training Strategy

Transfer learning with partial layer freezing

Adam optimizer

Categorical cross-entropy loss

Early stopping and model checkpointing

Class-weighted training to handle imbalance

Evaluation

Accuracy

Precision, Recall, F1-Score

Confusion Matrix

Explainable AI

Grad-CAM used to highlight discriminative regions in MRI scans influencing model predictions

Deployment

Streamlit-based web interface for real-time inference and visualization

Project Structure
Brain_MRI_Classification/
│
├── app.py                  # Streamlit deployment with Grad-CAM
├── main.py                 # Training and evaluation script
├── requirements.txt        # Project dependencies
│
├── dataset/
│   ├── train/
│   ├── val/
│   └── test/
│
├── models/
│   └── brain_mri_v2.keras  # Trained deep learning model
│
├── outputs/
│   ├── confusion_matrix.png
│   └── gradcam/
│
└── README.md

Installation & Setup
1. Create virtual environment
python -m venv venv

2. Activate environment
# Windows
.\venv\Scripts\Activate.ps1

3. Install dependencies
pip install -r requirements.txt

Training the Model

Run the training script:

python main.py


The trained model will be saved as:

models/brain_mri_v2.keras


Evaluation results and confusion matrix will be stored in the outputs/ folder.

Running the Streamlit Application

Launch the web application:

streamlit run app.py


The app allows:

Uploading a brain MRI image

Viewing predicted disease class and confidence

Visualizing Grad-CAM heatmaps for explainability

Results

The model demonstrates reliable classification performance across four brain disease categories. Grad-CAM visualizations confirm that the network focuses on clinically relevant regions of the MRI scans, enhancing transparency and trust in predictions.

Key Features

Deep learning–based medical image classification

Transfer learning using EfficientNet

Explainable AI with Grad-CAM

Streamlit-based interactive deployment

Windows-stable .keras model format

Applications

Clinical decision support systems

Medical imaging research

AI-assisted radiology tools

Educational and academic demonstrations

Conclusion

This project demonstrates the effectiveness of deep learning and explainable AI in automated brain MRI analysis. By combining high-performance classification with interpretability, NeuroScan provides a practical and trustworthy solution for medical image analysis.

Future Enhancements

Integration with real-time hospital systems

Multi-modal MRI sequence analysis

Federated learning for privacy-preserving training

Lightweight model optimization for edge deployment