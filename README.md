# Hybrid Autoregressive-Diffusion Model for Real-Time Streaming Sign Language Production

This repository contains the code for the HybridSign model and inference code.



## 1. About

Earlier Sign Language Production (SLP) models typically relied on autoregressive methods that generate output tokens one by one, which inherently provide temporal alignment. Although techniques like Teacher Forcing can prevent model collapse during training, they often suffer from error accumulation during inference. In contrast, more recent approaches based on diffusion models leverage step-by-step denoising to enable high-quality generation. However, the iterative nature of these models and the requirement to denoise entire sequences limit their applicability in real-time tasks like SLP. To address this, we propose a novel hybrid autoregressive-diffusion model for real-time streaming generation. Our approach leverages the strengths of autoregressive models in sequential dependency modeling and diffusion models in refining high-quality outputs. To capture fine-grained body movements, we design a Multi-Scale Pose Representation module that separately extracts detailed features from distinct body parts and integrates them via a Multi-Scale Fusion module. Furthermore, we introduce a Confidence-Aware Causal Attention mechanism that utilizes joint-level confidence scores to dynamically guide the pose generation process, improving accuracy and robustness. Extensive experiments on the PHOENIX14T and How2Sign datasets demonstrate the effectiveness of our method in both generation quality and real-time streaming efficiency.



## 2. Data
Phoenix14T data can be downloaded from https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/ 

How2Sign data can be downloaded from https://how2sign.github.io/

The data file storage format should be similar to the following.
```css
HybridSign/
├── data/
│   ├── train.pkl
│   ├── dev.pkl
│   ├── test.pkl
├─
```

**Note: All keypoints should be stacked together, maintaining the (x, y, confidence) format.**