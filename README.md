
# Image Segmentation with UNet and Fourier Neural Networks

This repository contains code for training and evaluating image segmentation models using various architectures, including UNet, ResUNet, and UNet with Fourier Neural Networks (FNNs). The implementation utilizes PyTorch and TensorBoard for monitoring training progress.

## Features

- **Model Architectures**: UNet, ResUNet, UNet with FNN, and MinimalUNet with FNN.
- **Loss Functions**: Dice Loss, IoU Loss, and Pixel-wise Binary Cross-Entropy Loss.
- **Data Handling**: Custom dataset class for loading and transforming human image segmentation data.
- **Training & Evaluation**: Training loop with model evaluation and checkpoint saving.

## Installation

1. Clone the repository:
   ```bash
   git clone [<repo_url>](https://github.com/Amar-Nath-Singh/Full-Body-Seg.git)
   cd Full-Body-Seg
   ```

2. Install dependencies:
   ```bash
   pip install torch torchvision tqdm
   ```

## Usage

1. **Prepare Data**: Place your dataset in `segmentation_full_body_mads_dataset_1192_img/` and ensure the CSV file for annotations is named `df.csv`.

2. **Configure Training**:
   - Adjust the configuration parameters in the `main` function as needed.

3. **Run Training**:
   ```bash
   python train.py
   ```

4. **Monitor Training**:
   - TensorBoard logs are saved for visualization. Run TensorBoard with:
     ```bash
     tensorboard --logdir=runs
     ```

![image](https://github.com/user-attachments/assets/d98a7234-a79a-4636-b7b2-064dab36ab05)
![image](https://github.com/user-attachments/assets/cd066cec-7d02-405d-8174-5e7c4f9f3fc9)


## Script Overview

- **DiceLoss & IoULoss**: Custom loss functions for segmentation tasks.
- **pixel_wise_loss**: Binary Cross-Entropy loss with logits.
- **train_one_epoch**: Function to train the model for one epoch.
- **evaluate_model**: Function to evaluate the model on validation data.
- **main**: Configuration, data loading, model training, and validation.

## Notes

- Ensure you have a CUDA-compatible GPU if you want to leverage GPU acceleration.
- Models are saved in the `models/` directory with the best model based on validation loss being saved separately.
