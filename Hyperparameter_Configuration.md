# Hyperparameter Configuration Guide

## Overview

This document describes all hyperparameters used in the Fire Detection UNet model, including training, model architecture, and evaluation parameters.

## 1. Training Hyperparameters

### Current Configuration (Default)

```python
epochs = 10          # Number of training epochs
batch_size = 4       # Batch size for training
learning_rate = 1e-4 # Learning rate (0.0001)
optimizer = Adam     # Optimizer type
loss_function = DiceBCELoss  # Combined Dice + BCE loss
```

### Detailed Description

#### **Epochs**
- **Default**: `10`
- **Range**: Typically 10-50 for segmentation tasks
- **Description**: Number of complete passes through the training dataset
- **Recommendation**: 
  - Start with 10-20 epochs
  - Monitor validation loss to avoid overfitting
  - Use early stopping if validation loss plateaus

#### **Batch Size**
- **Default**: `4`
- **Range**: 2-16 (depends on GPU memory)
- **Description**: Number of samples processed before updating model weights
- **Trade-offs**:
  - **Smaller batch (2-4)**: More gradient updates, slower but potentially better convergence
  - **Larger batch (8-16)**: Faster training, more stable gradients, but requires more memory
- **Recommendation**: 
  - Use 4-8 for most cases
  - Increase if you have GPU memory available
  - Decrease if you get out-of-memory errors

#### **Learning Rate**
- **Default**: `1e-4` (0.0001)
- **Range**: `1e-5` to `1e-3`
- **Description**: Step size for weight updates during optimization
- **Recommendation**:
  - **1e-4**: Good starting point (current default)
  - **1e-3**: Faster but may be unstable
  - **1e-5**: Slower but more stable
  - Consider learning rate scheduling (reduce by 0.5 every 10 epochs)

#### **Optimizer**
- **Type**: Adam (Adaptive Moment Estimation)
- **Default Parameters**:
  - `lr = 1e-4`
  - `betas = (0.9, 0.999)` (default PyTorch)
  - `eps = 1e-8` (default PyTorch)
  - `weight_decay = 0` (no regularization)
- **Why Adam**: 
  - Adaptive learning rate per parameter
  - Good for segmentation tasks
  - Handles sparse gradients well
- **Alternatives**:
  - **SGD with momentum**: `torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)`
  - **AdamW**: Better weight decay handling

#### **Loss Function**
- **Type**: `DiceBCELoss` (Combined Dice Loss + Binary Cross-Entropy)
- **Components**:
  - **BCE Loss**: Binary Cross-Entropy with Logits
  - **Dice Loss**: Measures overlap between prediction and ground truth
- **Formula**:
  ```
  Total Loss = BCE Loss + Dice Loss
  Dice = (2 * intersection + smooth) / (union + smooth)
  Dice Loss = 1 - Dice
  ```
- **Smooth Parameter**: `1` (prevents division by zero)
- **Why Combined**: 
  - BCE handles pixel-wise classification
  - Dice handles class imbalance (fire pixels are rare)

## 2. Model Architecture Hyperparameters

### UNet Configuration

```python
in_channels = 4      # RGB (3) + NIR (1)
out_channels = 1     # Single channel output (fire probability)
pretrained = True    # Use pretrained EfficientNet-B0 encoder
```

### Encoder (EfficientNet-B0)
- **Backbone**: EfficientNet-B0 (pretrained on ImageNet)
- **Input Channels**: Modified to accept 4 channels (RGB + NIR)
- **Output Features**: 320 channels from last encoder layer

### Decoder
- **Layer 1**: ConvTranspose2d(320 → 128, stride=4)
- **Layer 2**: ConvTranspose2d(128 → 64, stride=2)
- **Layer 3**: ConvTranspose2d(64 → 32, stride=2)
- **Layer 4**: ConvTranspose2d(32 → 16, stride=2)
- **Output**: Conv2d(16 → 1, kernel_size=1)

## 3. Data Hyperparameters

### Dataset Configuration

```python
patch_size = 256     # Size of input patches (256x256)
mean = [0.485, 0.456, 0.406, 0.5]  # Normalization mean [R, G, B, NIR]
std = [0.229, 0.224, 0.225, 0.25]  # Normalization std [R, G, B, NIR]
```

#### **Patch Size**
- **Default**: `256`
- **Options**: 128, 256, 512
- **Trade-offs**:
  - **Larger (512)**: More context, slower training, more memory
  - **Smaller (128)**: Faster training, less context
  - **256**: Good balance

#### **Normalization Parameters**
- **Mean**: Per-channel mean for Z-score normalization
- **Std**: Per-channel standard deviation
- **Recommendation**: Calculate from your training data (see preprocessing guide)

## 4. Evaluation Hyperparameters

### F1 Score Threshold

```python
threshold = 0.5  # Probability threshold for binary classification
```

#### **Threshold Selection**
- **Default**: `0.5`
- **Range**: `0.01` to `0.9`
- **Description**: Probability value above which a pixel is classified as fire
- **Impact**: 
  - **Lower threshold (0.1-0.3)**: More fire pixels predicted (higher recall, lower precision)
  - **Higher threshold (0.7-0.9)**: Fewer fire pixels predicted (higher precision, lower recall)
- **Recommendation**: 
  - Start with 0.5
  - Tune based on your use case (precision vs recall trade-off)
  - Use validation set to find optimal threshold

## 5. XAI Hyperparameters

### GradCAM
- **Target Layer**: Last encoder layer (`model.encoder.features[7]`)
- **No additional hyperparameters**

### Integrated Gradients
```python
baseline = None      # Baseline (zeros if None)
steps = 50           # Number of interpolation steps
```

#### **Steps**
- **Default**: `50`
- **Range**: 20-100
- **Description**: Number of points along the path from baseline to input
- **Trade-off**: More steps = more accurate but slower

## 6. Recommended Hyperparameter Configurations

### Configuration 1: Fast Training (Quick Experiments)
```python
epochs = 5
batch_size = 8
learning_rate = 1e-3
```
- **Use case**: Quick experiments, debugging
- **Training time**: ~30 minutes
- **Expected F1**: Lower (0.3-0.5)

### Configuration 2: Balanced (Current Default)
```python
epochs = 10
batch_size = 4
learning_rate = 1e-4
```
- **Use case**: Standard training
- **Training time**: ~1-2 hours
- **Expected F1**: Moderate (0.5-0.7)

### Configuration 3: High Performance
```python
epochs = 20
batch_size = 4
learning_rate = 1e-4
# Add learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)
```
- **Use case**: Final model training
- **Training time**: ~3-4 hours
- **Expected F1**: Higher (0.6-0.8)

### Configuration 4: With Class Weighting
```python
epochs = 20
batch_size = 4
learning_rate = 1e-4
fire_weight = 10.0  # Weight fire pixels 10x more
```
- **Use case**: Severe class imbalance
- **Benefit**: Better fire pixel detection

## 7. Hyperparameter Tuning Strategy

### Step 1: Learning Rate
1. Start with `1e-4`
2. If loss doesn't decrease: try `1e-3`
3. If loss is unstable: try `1e-5`
4. Use learning rate finder if available

### Step 2: Batch Size
1. Start with `4`
2. Increase to `8` if memory allows
3. Monitor training speed vs memory usage

### Step 3: Epochs
1. Start with `10`
2. Monitor validation loss
3. Stop if validation loss plateaus (early stopping)
4. Increase to `20-30` if still improving

### Step 4: Threshold
1. Train model with default threshold `0.5`
2. Evaluate on validation set with multiple thresholds
3. Choose threshold that maximizes F1 score
4. Or choose based on precision/recall trade-off

## 8. Advanced Hyperparameters

### Learning Rate Scheduling
```python
# Reduce learning rate when loss plateaus
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',      # Minimize loss
    factor=0.5,      # Reduce by 50%
    patience=5,      # Wait 5 epochs
    verbose=True
)

# Or step-based scheduling
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,    # Every 10 epochs
    gamma=0.5        # Reduce by 50%
)
```

### Gradient Clipping
```python
# Prevent exploding gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Weight Decay (L2 Regularization)
```python
optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=1e-4,
    weight_decay=1e-5  # L2 regularization
)
```

## 9. Hyperparameter Summary Table

| Hyperparameter | Default | Range | Impact |
|---------------|---------|-------|-------|
| **Epochs** | 10 | 5-50 | Training time, model performance |
| **Batch Size** | 4 | 2-16 | Training speed, memory usage |
| **Learning Rate** | 1e-4 | 1e-5 to 1e-3 | Convergence speed, stability |
| **Optimizer** | Adam | Adam/SGD/AdamW | Optimization efficiency |
| **Loss Function** | DiceBCE | DiceBCE/Weighted | Class imbalance handling |
| **Patch Size** | 256 | 128-512 | Context, memory usage |
| **Threshold** | 0.5 | 0.01-0.9 | Precision/recall trade-off |
| **Fire Weight** | 1.0 | 1.0-20.0 | Fire pixel importance |

## 10. Example: Complete Training Configuration

```python
# Hyperparameters
config = {
    # Training
    'epochs': 20,
    'batch_size': 4,
    'learning_rate': 1e-4,
    'optimizer': 'Adam',
    'weight_decay': 1e-5,
    
    # Model
    'in_channels': 4,
    'out_channels': 1,
    'pretrained': True,
    
    # Data
    'patch_size': 256,
    'mean': [0.485, 0.456, 0.406, 0.5],
    'std': [0.229, 0.224, 0.225, 0.25],
    
    # Loss
    'loss_type': 'DiceBCE',
    'dice_smooth': 1.0,
    
    # Evaluation
    'threshold': 0.5,
    
    # Advanced
    'use_scheduler': True,
    'scheduler_patience': 5,
    'gradient_clip': 1.0,
}

# Usage
model = train_unet(
    dataset=train_dataset,
    epochs=config['epochs'],
    batch_size=config['batch_size'],
    lr=config['learning_rate'],
    device=device,
    save_path="unet_fire_model"
)
```

## 11. Tips for Hyperparameter Tuning

1. **Start Simple**: Use default values first
2. **One at a Time**: Change one hyperparameter at a time
3. **Monitor Metrics**: Track loss, F1 score, precision, recall
4. **Use Validation Set**: Don't tune on test set
5. **Document Changes**: Keep track of what works
6. **Be Patient**: Training can take time, let it complete
7. **Early Stopping**: Stop if validation loss doesn't improve

## Summary

The default hyperparameters are:
- **Epochs**: 10
- **Batch Size**: 4
- **Learning Rate**: 1e-4
- **Optimizer**: Adam
- **Loss**: DiceBCE
- **Threshold**: 0.5

These provide a good starting point for fire detection. Adjust based on your specific dataset and requirements!
