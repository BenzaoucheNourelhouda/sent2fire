# Beyond the Visible: RGB vs RGB-NIR Fusion for Wildfire Segmentation
*Comparative study on the [Sen2Fire](https://doi.org/10.48550/arXiv.2403.17884) benchmark *

## Intro
Wildfire detection from RGB alone struggles with smoke, glare and low light,
and paired RGB-NIR datasets are scarce. This study asks: does adding a
near-infrared (NIR) channel, real or synthetic, improve pixel-level fire
segmentation on Sentinel-2 imagery?
Report: [link] · My role: [your part]

## Technologies
Python · PyTorch · [segmentation library if any] · Kaggle (P100 GPU)
Models: ResNet34 · U-Net (EfficientNet-B0) · SegFormer (MiT-B0)
XAI: Grad-CAM · Integrated Gradients

## Setup
- **Data:** Sen2Fire, 512×512 patches, RGB (B4/B3/B2) + NIR (B8)
- **Split by scene:** train on scenes 1-2, validate on 3, test on 4
  (no geographic overlap)
- **Inputs:** RGB · RGB + real NIR · RGB + synthetic NIR
- **Training:** Adam, lr 1e-4, Dice + BCE loss, batch size 8, up to 20 epochs
- **Challenge:** only 3.39% of pixels are fire, so accuracy is misleading;
  I evaluate with F1/Dice and IoU

## Results (test set, F1 %)
| Model | RGB | RGB + real NIR |
|---|---|---|
| ResNet34 | 3.28 | 2.00 |
| U-Net (EffNet-B0) | 3.77 | 3.11 |
| SegFormer (MiT-B0) | **4.11** | 3.30 |

- SegFormer performed best across input types.
- Adding real NIR by simple channel concatenation did not help; it lowered
  scores for all three models.
- Synthetic NIR with U-Net showed higher recall in a preliminary run
  (different training setup, so not directly comparable to the table above).

## Explainability
Grad-CAM and Integrated Gradients (U-Net, synthetic NIR) and prediction
confidence maps (SegFormer) show attention concentrated on fire regions.

## What I learned
- Under extreme class imbalance, high accuracy means nothing; overlap metrics do.
- More spectral data isn't automatically better; fusion strategy matters.
- instead of treating the lack of data as a breakpoint we took it as a research problem to elevate our work: can we train fire segmentation models with synthetic data. 

## What could be improved
Richer fusion than channel concatenation, a controlled synthetic NIR
comparison, larger and better-balanced data, and deeper architectures.

## Contact
nour339be@gmail.com
