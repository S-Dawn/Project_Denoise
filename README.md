# Project Denoise

A deep learning project that implements a Convolutional Neural Network (CNN) to denoise 4D scientific data. The project focuses on removing noise from complex spatiotemporal data using modern deep learning techniques.

## Project Overview

This project provides a complete pipeline for:
1. Loading and processing 4D MATLAB (.mat) data files
2. Training a CNN-based denoising model
3. Applying the trained model to denoise new data

### Key Features

- **Data Processing**:
  - Handles 4D data in format (Time, Space, Channel, Samples)
  - Automatic data normalization and preprocessing
  - Conversion between different data formats (MAT/CSV)

- **Neural Network Architecture**:
  - U-Net style CNN architecture optimized for denoising
  - Skip connections for preserving fine details
  - Batch normalization for training stability
  - Dynamic input size handling

- **Training Features**:
  - Automatic train/validation/test split
  - Model checkpointing
  - Early stopping to prevent overfitting
  - Progress monitoring with multiple metrics (MSE, MAE, RMSE)

## Installation

This project uses Poetry for dependency management. To get started:

1. Install Poetry if you haven't already:
   ```
   pip install poetry
   ```

2. Install dependencies:
   ```
   poetry install
   ```

## Usage

### Data Preparation
```python
from src.mat_loader import MatLoader

# Load your data
loader = MatLoader("path/to/your/data.mat")
data = loader.get_tensorflow_format()  # Reshapes to (Samples, Channel, Height, Width)
```

### Training a Model
```python
from src.denoiser_cnn import DenoiserCNN

# Create and train the model
denoiser = DenoiserCNN(input_shape=blur_data.shape)
x_train, x_val, x_test, y_train, y_val, y_test = denoiser.prepare_data(
    blur_data, original_data
)

# Train the model
history = denoiser.train(
    x_train, y_train,
    x_val, y_val,
    epochs=100,
    batch_size=32,
    checkpoint_dir='checkpoints'
)
```

### Denoising New Data
```python
# Load a trained model
denoiser = DenoiserCNN.load_model('path/to/model.h5')

# Denoise new data
denoised_data = denoiser.predict(new_blur_data)
```

## Project Structure

```
Project_Denoise/
├── src/
│   ├── __init__.py
│   ├── mat_loader.py      # MATLAB file handling
│   └── denoiser_cnn.py    # CNN model implementation
├── Data/
│   ├── Blur_rho_16g_full_data.mat    # Input blurred data
│   ├── Original_rho_full_data.mat     # Target clean data
│   └── checkpoints/                   # Model checkpoints
├── poetry.lock
├── pyproject.toml         # Project dependencies
└── README.md
```

## Model Architecture

The denoising CNN uses a U-Net style architecture with:
- Encoder path for feature extraction
- Decoder path for reconstruction
- Skip connections to preserve spatial information
- Multiple convolution layers with ReLU activation
- Batch normalization for training stability

The model is optimized for 4D spatiotemporal data and includes:
- MSE loss function for optimization
- Adam optimizer with learning rate scheduling
- Multiple evaluation metrics (MAE, MSE, RMSE)
- Checkpoint saving for best models
