**Exploring Vision Transformer and Swin Transformer for Image Classificaiton**

This repository contains two key modules: `VIT_SWIM` and `VIT_MAE_CLS_InfoNCE`. Each module focuses on deep learning concepts, including Vision Transformers (ViT), Swin Transformers, Masked Autoencoders (MAE), CLS tokens, and InfoNCE loss. 
The project aims to explore and compare various transformer-based architectures and loss functions for image classification tasks.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Modules Overview](#modules-overview)
  - [VIT_SWIM](#vit_swim)
  - [VIT_MAE_CLS_InfoNCE](#vit_mae_cls_infonce)
- [Concepts Explained](#concepts-explained)
  - [Vision Transformer (ViT)](#vision-transformer-vit)
  - [Swin Transformer (SWIM)](#swin-transformer-swim)
  - [Masked Autoencoder (MAE)](#masked-autoencoder-mae)
  - [CLS Token](#cls-token)
  - [InfoNCE Loss](#infonce-loss)
---

## Project Overview

The project leverages MNIST data for classification using advanced transformer-based architectures. The tasks are divided into two modules:

1. **VIT_SWIM:** Comparison of Vision Transformer and Swin Transformer with ablation studies and distillation techniques.
2. **VIT_MAE_CLS_InfoNCE:** Implementation of Masked Autoencoders (MAE), CLS token classification, and contrastive learning with InfoNCE loss.

---

## Modules Overview

### VIT_SWIM

#### Objective
- Train a Vision Transformer (ViT) and Swin Transformer for MNIST classification.
- Implemented the VIT and SWIM transformers blocks from scratch.
- Perform hyperparameter tuning and ablation studies.
- Use KL divergence loss with temperature scaling for knowledge distillation.

#### Tasks
1. **Vision Transformer (ViT)**
   - Use 2 transformer blocks.
   - Hyperparameter tuning: epoch count, learning rate, hidden dimension.
   - Patch size: 7x7.
   - Evaluate model using confusion matrix and classification accuracy.

2. **Swin Transformer (SWIM)**
   - Use 2 Swin Transformer blocks.
   - Train using KL distillation loss (temperature = 4).
   - No ground truth labels are used in the training process.

3. **Regularization and Augmentation**
   - Implement augmentation techniques and dropout for better generalization.
   - In regularization i used Weight decay and Dropout.
   - In Augmentation i used Mixup and RandAugment techniques.

4. **InfoNCE Loss Application**
   - Contrastive learning approach to enhance representation learning.

---

### VIT_MAE_CLS_InfoNCE

#### Objective
- Implement and fine-tune MAE, CLS token, and InfoNCE loss-based training.
- Compare various loss combinations and evaluate performance.

#### Tasks
1. **MAE (Masked Autoencoder) Training**
   - Use 2 ViT blocks.
   - Patch size: 7x7 with 50% masked tokens.
   - Loss: L2 loss applied only to masked tokens.
   - Train for at least 10 epochs.
   - And this MAE trained VIT is then used for the downstream task of classification by adding the classification layer instead of the linear layer in the end.
   - And by this model learn more local and contextual information because it is forced to reconstruct the missing patches by leveraging the surrounding context, which further helps in the classification task which requires both of local and global informations.
   - MAE + L2 loss, focuses on detailed local feature reconstruction, making it useful for tasks requiring fine spatial resolution and transfer learning adaptability.

2. **CLS Token with InfoNCE Loss**
   - In this approach, the CLS token is appended to the sequence of image patches and is used to aggregate information from the entire image through self-attention layers.
     The InfoNCE loss is applied to the CLS token to learn an embedding space where:

      * Similar samples (positive pairs) are pulled together.
      * Dissimilar samples (negative pairs) are pushed apart.
    - CLS token + InfoNCE loss is more focused on global feature aggregation and is excellent for tasks that require strong instance discrimination, such as classification.


3. **Combined Training with MAE + InfoNCE**
   - Weighted combination: 80% MAE + 20% InfoNCE.
   - MAE + L2 for local and CLS + InfoNCE (Learning both global and local representaion). 
   -  Which improve robustness and transferability.
   -  It enhances the model's ability to handle varied data distributions, occlusions, and noise, making it more suitable for real-world applications.

4. **Video MAE Implementation**
   - Video MAE (Masked Autoencoder for Video) is an extension of image-based MAE designed to learn temporal and spatial features simultaneously by processing multiple frames.
     It adopts a self-supervised learning paradigm where a portion of video tokens is masked, and the model learns to reconstruct them.
     This approach is particularly useful for video understanding tasks such as action recognition and classification.
     
   - Processes multiple frames simultaneously (in this case, 3 frames at a time), meaning the model captures both spatial and temporal features.
   - Video MAE allows the model to learn the relationship between consecutive frames, helping it capture motion dynamics and temporal dependencies.
   - This is essential for tasks like action recognition, where changes across time carry important contextual information.
   - In image processing, patch tokens are extracted from one frame.
   - In video processing, patches are extracted from each frame and concatenated to form a 3D token sequence, effectively capturing the sequence of events.
   - The key difference is that image MAE focuses purely on spatial features, while video MAE incorporates temporal context by processing multiple frames at once.
   - Tokenization from consecutive frames.
   - Random token shuffling and masking.
   - Training with L1 loss.

---

## Concepts Explained

### Vision Transformer (ViT)
The Vision Transformer (ViT) applies self-attention mechanisms to image patches instead of convolutional layers. It divides images into fixed-size patches, projects them into an embedding space, and processes them with transformer encoders.

### Swin Transformer (SWIM)
The Swin Transformer introduces hierarchical feature representation using a shifted windowing mechanism. This approach enhances the efficiency of self-attention by limiting computations to local windows.

### Masked Autoencoder (MAE)
MAE is a self-supervised learning framework where a significant portion of image patches is masked during training, and the model learns to reconstruct the missing information.

### CLS Token
The CLS (classification) token is a learnable parameter added to the input sequence, designed to aggregate information from the entire sequence and used for downstream classification tasks.

### InfoNCE Loss
InfoNCE (Information Noise Contrastive Estimation) is a contrastive loss function that maximizes the similarity between positive pairs while minimizing it for negative samples, commonly used in self-supervised learning.


