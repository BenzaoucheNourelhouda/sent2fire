# Data Preprocessing Guide for Fire Detection

## Overview

The data preprocessing pipeline transforms raw satellite imagery (RGB + NIR) into normalized tensors suitable for the UNet model.

## Preprocessing Steps

### 1. **Data Loading**
- Load RGB and NIR channels from `.npz` files
- Files are organized by scenes (scene1, scene2, etc.)
- Each scene has:
  - Original directory: Contains labels
  - `_NS` directory: Contains RGB + synthetic NIR data

### 2. **Pixel Value Normalization**
```python
rgb = ns["rgb"].astype(np.float32) / 255.0  # Scale to [0, 1]
nir = ns["nir"].astype(np.float32) / 255.0  # Scale to [0, 1]
```
- **Purpose**: Convert uint8 [0, 255] to float32 [0, 1]
- **Why**: Neural networks work better with normalized inputs

### 3. **NIR Channel Handling**
```python
if nir.ndim == 3:
    nir = nir.mean(axis=0)  # Average if multiple channels
nir = nir[None, ...]  # Add channel dimension: (H, W) -> (1, H, W)
```
- **Purpose**: Ensure NIR is single channel
- **Why**: Some data may have multi-channel NIR, we need single channel

### 4. **Channel Concatenation**
```python
x = np.concatenate([rgb, nir], axis=0)  # (3, H, W) + (1, H, W) -> (4, H, W)
```
- **Result**: 4-channel input [R, G, B, NIR]
- **Why**: UNet expects 4-channel input

### 5. **Z-Score Standardization**
```python
x = (x - mean[:, None, None]) / std[:, None, None]
```
- **Formula**: `normalized = (value - mean) / std`
- **Purpose**: Center data around 0 with unit variance
- **Why**: Helps training stability and convergence

## Mean and Standard Deviation Calculation

### Option 1: Calculate from Your Data (Recommended)
```python
def calculate_mean_std(base_dir, scenes, sample_size=None):
    """Calculate mean and std from your dataset."""
    all_pixels = {0: [], 1: [], 2: [], 3: []}  # R, G, B, NIR
    
    for scene in scenes:
        orig_dir = os.path.join(base_dir, scene)
        ns_dir = os.path.join(base_dir, f"{scene}_NS")
        
        files = sorted(glob.glob(os.path.join(orig_dir, "*.npz")))
        if sample_size:
            files = files[:sample_size]
        
        for f in files:
            ns_path = os.path.join(ns_dir, os.path.basename(f))
            if not os.path.exists(ns_path):
                continue
            
            ns = np.load(ns_path)
            rgb = ns["rgb"].astype(np.float32) / 255.0
            nir = ns["nir"].astype(np.float32) / 255.0
            
            if nir.ndim == 3:
                nir = nir.mean(axis=0)
            nir = nir[None, ...]
            
            x = np.concatenate([rgb, nir], axis=0)
            
            # Collect pixel values
            for c in range(4):
                all_pixels[c].append(x[c].flatten())
    
    # Calculate statistics
    mean = np.array([np.concatenate(all_pixels[c]).mean() for c in range(4)])
    std = np.array([np.concatenate(all_pixels[c]).std() for c in range(4)])
    
    return mean, std

# Usage
base = r"C:\Users\BAB AL SAFA\Documents\projects\computer vision\s2f"
mean, std = calculate_mean_std(base, scenes=["scene1", "scene2"], sample_size=100)
print(f"Mean: {mean}")
print(f"Std:  {std}")
```

### Option 2: Use Default Values (ImageNet-like)
```python
# Default values (approximate, calculate from your data for better results)
mean = np.array([0.485, 0.456, 0.406, 0.5])  # [R, G, B, NIR]
std = np.array([0.229, 0.224, 0.225, 0.25])   # [R, G, B, NIR]
```

## Complete Preprocessing Pipeline

```python
# Step 1: Load data
orig = np.load(orig_path)  # Contains label
ns = np.load(ns_path)      # Contains RGB + NIR

# Step 2: Extract and normalize
label = orig["label"].astype(np.float32)  # Ground truth mask
rgb = ns["rgb"].astype(np.float32) / 255.0  # [0, 1]
nir = ns["nir"].astype(np.float32) / 255.0  # [0, 1]

# Step 3: Handle NIR
if nir.ndim == 3:
    nir = nir.mean(axis=0)  # Average if multi-channel
nir = nir[None, ...]  # Add channel dim

# Step 4: Concatenate channels
x = np.concatenate([rgb, nir], axis=0)  # (4, H, W)

# Step 5: Standardize
x = (x - mean[:, None, None]) / std[:, None, None]

# Step 6: Convert to tensor
x = torch.tensor(x, dtype=torch.float32)  # (4, H, W)
y = torch.tensor(label, dtype=torch.float32).unsqueeze(0)  # (1, H, W)
```

## Data Format

### Input (x)
- **Shape**: `(4, H, W)` where H, W are image dimensions
- **Channels**: [R, G, B, NIR]
- **Type**: `torch.float32`
- **Range**: Normalized (approximately [-2, 2] after standardization)

### Label (y)
- **Shape**: `(1, H, W)`
- **Values**: Binary mask (0 = no fire, 1 = fire)
- **Type**: `torch.float32`

## Important Notes

1. **Calculate mean/std from your data**: Don't use ImageNet values blindly
2. **Consistent preprocessing**: Use same mean/std for training and evaluation
3. **NIR handling**: Always check if NIR needs averaging
4. **Data type**: Use `float32` for both input and labels
5. **Channel order**: RGB first, then NIR (4th channel)

## Example: Full Preprocessing Function

```python
def preprocess_sample(orig_path, ns_path, mean, std):
    """
    Complete preprocessing for a single sample.
    
    Args:
        orig_path: Path to original .npz (contains label)
        ns_path: Path to NS .npz (contains RGB + NIR)
        mean: Array of means [R, G, B, NIR]
        std: Array of stds [R, G, B, NIR]
    
    Returns:
        x: Preprocessed input tensor (4, H, W)
        y: Label tensor (1, H, W)
    """
    # Load
    orig = np.load(orig_path)
    ns = np.load(ns_path)
    
    # Extract
    label = orig["label"].astype(np.float32)
    rgb = ns["rgb"].astype(np.float32) / 255.0
    nir = ns["nir"].astype(np.float32) / 255.0
    
    # Handle NIR
    if nir.ndim == 3:
        nir = nir.mean(axis=0)
    nir = nir[None, ...]
    
    # Concatenate
    x = np.concatenate([rgb, nir], axis=0)
    
    # Standardize
    x = (x - mean[:, None, None]) / std[:, None, None]
    
    # Convert to tensors
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
    
    return x, y
```

## Summary

The preprocessing pipeline:
1. ✅ Loads RGB + NIR from .npz files
2. ✅ Normalizes pixel values to [0, 1]
3. ✅ Handles NIR channel (averages if multi-channel)
4. ✅ Concatenates to 4-channel input
5. ✅ Applies Z-score standardization
6. ✅ Converts to PyTorch tensors

This ensures the data is ready for the UNet model!
