# Self-Supervised Image Representation Learning using Masked Autoencoders (MAE)

## 📌 Objective
This repository contains a complete, from-scratch PyTorch implementation of a Masked Autoencoder (MAE). The objective of this project is to design a self-supervised system that learns meaningful visual representations by reconstructing images with **75% of their input patches masked**.

## 🧠 Architecture
This model follows an asymmetric transformer-based encoder-decoder design based on the original MAE paper by FAIR:
* **Encoder:** A large Vision Transformer (ViT-Base, ~86M parameters). It only processes the 25% visible patches, making it highly memory-efficient.
* **Decoder:** A lightweight Vision Transformer (ViT-Small, ~22M parameters). It takes the encoder's latent representations and learnable mask tokens to reconstruct the original pixel values.

## ⚙️ Implementation Details
* **Dataset:** TinyImageNet (upsampled to 224x224)
* **Environment:** Dual Kaggle T4 GPUs using `nn.DataParallel`
* **Patchification:** 16x16 patches
* **Training Techniques:** Mixed Precision (`torch.amp`), Cosine Annealing Scheduler, AdamW optimizer with Weight Decay, and Gradient Clipping.
* **Loss Function:** Mean Squared Error (MSE) computed *strictly* on masked patches.

## 📊 Results & Evaluation
The model was evaluated using qualitative visual reconstructions and quantitative metrics:
* **PSNR (Peak Signal-to-Noise Ratio)**
* **SSIM (Structural Similarity Index)**

## 🚀 Live Demo
A real-time Gradio web application is included in the notebook, allowing users to upload custom images and adjust the masking ratio dynamically.
