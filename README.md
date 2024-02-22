# NeuroGuard-BrainTumorDetection

Project aimed at automating brain tumor detection using OpenCV and VGG-16. Developed by the MedTech startup NeuroGuard.

## Introduction

This project focuses on automating the detection of brain tumors using the VGG-16 model. The VGG-16 model is a deep convolutional neural network that has been pre-trained on a large dataset. By fine-tuning this model and training it on brain tumor images, we aim to achieve accurate and efficient tumor detection.

## Getting Started

To get started with this project, follow the steps below:

1. Clone the repository: `git clone https://github.com/Loke-60000/NeuroGuard-BrainTumorDetection`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Train the model: `main.ipynb`


## Model Architecture

The VGG-16 model is used as the base model for this project. It is a deep convolutional neural network that has been widely used for image classification tasks. We add a few additional layers on top of the base model to adapt it to our specific task of brain tumor detection.

## Training

The model is trained using the training set. Data augmentation techniques are applied to increase the diversity of the training data and improve the model's generalization ability. The model is trained for 10 epochs with a batch size of 32.

## Evaluation

The trained model is evaluated on the testing set. The test loss and accuracy are calculated to assess the performance of the model in detecting brain tumors.