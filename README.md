# LARS-MobileNet-V4

This repository contains the implementation of the lightweight convolutional neural network architecture described in the paper "Advancing Real-Time Crop Disease Detection on Edge Computing Devices using Lightweight Convolutional Neural Networks."

## Overview

This project introduces LARS-MobileNetV4, an optimized version of MobileNetV4 specifically designed for real-time crop disease detection on resource-constrained edge devices such as Raspberry Pi. Our implementation achieves 97.84% accuracy on the Paddy Doctor dataset while maintaining fast inference times (88.91ms on Raspberry Pi 5), making it suitable for deployment in agricultural field settings.

## Key Features

- **Optimized MobileNetV4 Architecture**: Enhanced with Squeeze-and-Excitation (SE) blocks and Efficient Channel Attention (ECA) mechanisms
- **Resource-Efficient Design**: Significantly reduced model size (10.2MB) compared to ResNet34 (85.3MB)
- **Real-time Performance**: Average inference time of 39ms on CPU and 88.91ms on Raspberry Pi 5
- **High Accuracy**: 97.84% detection accuracy across 12 common rice diseases
- **Custom Loss Function**: Combination of Focal Loss and Label Smoothing for better handling of class imbalance
- **Comprehensive Data Augmentation**: Robust augmentation pipeline to improve model generalization
- **Deployment-Ready**: Optimized for TFLite deployment on edge devices

## Model Architecture

LARS-MobileNetV4 builds upon the recently introduced MobileNetV4 architecture with several key optimizations:

1. **Universal Inverted Bottleneck (UIB)**: Merges features of Inverted Bottlenecks, ConvNext, and Feed Forward Networks to enhance flexibility in spatial and channel mixing
2. **Mobile Multi-Query Attention (MQA)**: An accelerator-optimized attention mechanism that reduces memory bandwidth bottlenecks
3. **Squeeze-and-Excitation Blocks**: Added to adaptively recalibrate channel-wise feature responses
4. **Efficient Channel Attention**: Captures cross-channel interactions with minimal computational overhead
5. **Neural Architecture Search (NAS)**: Tailored architecture for specific hardware

## Performance Comparison

| Model                 | Parameters (M) | Accuracy (%) | Model Size (MB) | Inference Time on CPU (ms) | Inference Time on Raspberry Pi 5 (ms) |
| --------------------- | -------------- | ------------ | --------------- | -------------------------- | ------------------------------------- |
| ResNet34              | 21.79          | 97.50        | 85.3            | 148.93                     | 264.50                                |
| MobileNet-V2          | 3.5            | 92.42        | 9.2             | 40.00                      | 73.09                                 |
| MobileNet-V3          | 2.5            | 95.62        | 10.3            | N/A                        | N/A                                   |
| MobileNet-V4          | 3.8            | 97.17        | 10.2            | 39.20                      | 88.91                                 |
| **LARS-MobileNet-V4** | **3.8**        | **97.84**    | **10.2**        | **39.20**                  | **88.91**                             |

## Training Strategies

Our implementation includes several optimization techniques:

| Model Variation                                                                  | Train Accuracy (%) | Test Accuracy (%) |
| -------------------------------------------------------------------------------- | ------------------ | ----------------- |
| MobileNet-V4 Baseline                                                            | 99.93              | 97.17             |
| MobileNet-V4 (Augmentations)                                                     | 99.60              | 97.21             |
| MobileNet-V4 (FocalLabelSmoothingLoss)                                           | 99.71              | 97.79             |
| MobileNet-V4 (Augmentations, FocalLabelSmoothingLoss, Squeeze-Excitation Blocks) | 99.68              | **97.84**         |

### Custom Loss Function

We implement a combination of Focal Loss and Label Smoothing:

1. **Label Smoothing**: Redistributes confidence across classes

   $$y_{smooth} = (1 - ε)y + ε/C$$

   where ε is the smoothing factor and C is the total number of classes.

2. **Focal Loss**: Focuses on harder examples

   $$L_{focal}(pt) = -α(1 - pt)^γ log(pt)$$

   where pt is the predicted probability for the true class.

3. **Combined Loss (FLS)**:
   
   $$L_{FLS} = -α(1 - pt)^γ log(p_{smooth})$$

## Requirements

```
torch
torchvision
timm
numpy
pandas
Pillow
scikit-learn
tqdm
wandb
```

### Data Preparation

Organize your data as follows:

```
├── train_images/
│   ├── disease_class_1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── disease_class_2/
│   └── ...
├── test_images/
└── train.csv
```

The train.csv file should contain:

- `image_id`: Filename of the image
- `label`: Disease class name

### Configuration

Key hyperparameters can be modified at the top of the script:

```python
LEARNING_RATE = 0.0001
ARCHITECTURE = "MobileNetV4"
EPOCHS = 50
BATCH_SIZE = 64
OPTIMISER = "Adam"
LOSS_FUNCTION = "FocalLabelSmoothingComboLoss"
NUM_CLASSES = 13  # 12 disease classes + 1 normal class
PRETRAINED = True
```

## Citation

If you use this code in your research, please cite our paper:

```
@article{nanda2025advancing,
  title={Advancing Real-Time Crop Disease Detection on Edge Computing Devices using Lightweight Convolutional Neural Networks},
  author={Nanda, Tanmay Rai and Shukla, Ananya and Srinivasa, Tanay Raghunandan and Bhargava, Jia and Chauhan, Sunita},
  journal={},
  year={2025},
  volume={},
  pages={}
}
```

## Acknowledgements

- We use the Paddy Doctor dataset for training and evaluation [Petchiammal et al., 2022]
