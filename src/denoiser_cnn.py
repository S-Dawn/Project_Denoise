"""
Module for implementing a Convolutional Neural Network for denoising data.
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Union
from .mat_loader import MatLoader


class DenoiserCNN:
    """A CNN-based denoiser model for 4D data."""
    
    def __init__(self, input_shape: Tuple[int, int, int, int]):
        """
        Initialize the denoiser CNN model.
        
        Args:
            input_shape: Tuple of (samples, channels, height, width)
        """
        self.input_shape = input_shape[1:]  # Remove batch dimension
        self.model = self._build_model()
        
    def _build_model(self) -> tf.keras.Model:
        """
        Build the CNN model architecture.
        
        Returns:
            tf.keras.Model: Compiled CNN model
        """
        inputs = tf.keras.layers.Input(shape=(
            self.input_shape[1],  # height (time)
            self.input_shape[2],  # width (space)
            self.input_shape[0]   # channels
        ))
        
        # Calculate padding to make dimensions divisible by 2
        height_pad = (2 - inputs.shape[1] % 2) % 2
        width_pad = (2 - inputs.shape[2] % 2) % 2
        
        # Add padding if needed
        if height_pad > 0 or width_pad > 0:
            padded = tf.keras.layers.ZeroPadding2D(
                padding=((0, height_pad), (0, width_pad))
            )(inputs)
        else:
            padded = inputs
        
        # Encoder path with fixed dimensions
        conv1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(padded)
        conv1 = tf.keras.layers.BatchNormalization()(conv1)
        conv1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
        skip1 = tf.keras.layers.BatchNormalization()(conv1)
        pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(skip1)
        
        # Middle path
        conv2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = tf.keras.layers.BatchNormalization()(conv2)
        conv2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
        conv2 = tf.keras.layers.BatchNormalization()(conv2)
        
        # Decoder path
        up1 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv2)
        
        # Ensure dimensions match before concatenation
        if up1.shape[1:3] != skip1.shape[1:3]:
            up1 = tf.keras.layers.Cropping2D(
                cropping=((0, up1.shape[1] - skip1.shape[1]), 
                         (0, up1.shape[2] - skip1.shape[2]))
            )(up1)
        
        # Add skip connection
        concat1 = tf.keras.layers.Concatenate()([up1, skip1])
        
        conv3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(concat1)
        conv3 = tf.keras.layers.BatchNormalization()(conv3)
        conv3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)
        conv3 = tf.keras.layers.BatchNormalization()(conv3)
        
        # Output layer with exact input size
        outputs = tf.keras.layers.Conv2D(self.input_shape[0], (1, 1), activation='linear')(conv3)
        
        # Remove padding if we added it
        if height_pad > 0 or width_pad > 0:
            outputs = tf.keras.layers.Cropping2D(
                cropping=((0, height_pad), (0, width_pad))
            )(outputs)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss='mse',
            metrics=['mae', 'mse', tf.keras.metrics.RootMeanSquaredError(name='rmse')]
        )
        
        return model
    
    def prepare_data(self, 
                    blur_data: np.ndarray, 
                    original_data: np.ndarray,
                    test_size: float = 0.2,
                    validation_split: float = 0.2) -> Tuple[np.ndarray, ...]:
        """
        Prepare the data for training by splitting into train/validation/test sets.
        
        Args:
            blur_data: Input blurred data
            original_data: Target clean data
            test_size: Proportion of data to use for testing
            validation_split: Proportion of training data to use for validation
            
        Returns:
            Tuple containing (x_train, x_val, x_test, y_train, y_val, y_test)
        """
        # Ensure data is in channels_last format
        blur_data = np.transpose(blur_data, (0, 2, 3, 1))
        original_data = np.transpose(original_data, (0, 2, 3, 1))
        
        # Normalize data to [0, 1] range
        blur_data = (blur_data - np.min(blur_data)) / (np.max(blur_data) - np.min(blur_data))
        original_data = (original_data - np.min(original_data)) / (np.max(original_data) - np.min(original_data))
        
        # Split into train and test
        x_train, x_test, y_train, y_test = train_test_split(
            blur_data, original_data, test_size=test_size, random_state=42
        )
        
        # Split training data into train and validation
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=validation_split, random_state=42
        )
        
        return x_train, x_val, x_test, y_train, y_val, y_test
    
    def train(self,
              x_train: np.ndarray,
              y_train: np.ndarray,
              x_val: np.ndarray,
              y_val: np.ndarray,
              epochs: int = 100,
              batch_size: int = 32,
              checkpoint_dir: Optional[Path] = None) -> tf.keras.callbacks.History:
        """
        Train the denoising model.
        
        Args:
            x_train: Training input data (blurred)
            y_train: Training target data (clean)
            x_val: Validation input data
            y_val: Validation target data
            epochs: Number of training epochs
            batch_size: Batch size for training
            checkpoint_dir: Directory to save model checkpoints
            
        Returns:
            Training history
        """
        callbacks = []
        
        # Add checkpointing if directory is provided
        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=str(checkpoint_dir / "model_{epoch:02d}_{val_loss:.4f}.h5"),
                    save_best_only=True,
                    monitor='val_loss',
                    mode='min'
                )
            )
        
        # Add early stopping
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
        )
        
        history = self.model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, blur_data: np.ndarray) -> np.ndarray:
        """
        Denoise the input blurred data.
        
        Args:
            blur_data: Input blurred data
            
        Returns:
            Denoised data
        """
        # Ensure data is in channels_last format
        blur_data = np.transpose(blur_data, (0, 2, 3, 1))
        
        # Make predictions
        denoised = self.model.predict(blur_data)
        
        # Convert back to channels_second format
        denoised = np.transpose(denoised, (0, 3, 1, 2))
        
        return denoised
    
    def save_model(self, filepath: Union[str, Path]) -> None:
        """Save the model to disk."""
        self.model.save(str(filepath))
    
    @classmethod
    def load_model(cls, filepath: Union[str, Path]) -> 'DenoiserCNN':
        """Load a saved model from disk."""
        model = tf.keras.models.load_model(str(filepath))
        instance = cls.__new__(cls)
        instance.model = model
        instance.input_shape = model.input_shape[1:]
        return instance


def main():
    # Example usage
    data_dir = Path(__file__).parent.parent / 'Data'
    
    # Load data
    blur_loader = MatLoader(data_dir / 'Blur_rho_16g_full_data.mat')
    original_loader = MatLoader(data_dir / 'Original_rho_full_data.mat')
    
    blur_data = blur_loader.get_tensorflow_format()
    original_data = original_loader.get_tensorflow_format()
    
    # Create and train model
    denoiser = DenoiserCNN(input_shape=blur_data.shape)
    
    # Prepare data
    x_train, x_val, x_test, y_train, y_val, y_test = denoiser.prepare_data(
        blur_data, original_data
    )
    
    # Create checkpoint directory
    checkpoint_dir = data_dir / 'checkpoints'
    
    # Train model
    history = denoiser.train(
        x_train, y_train,
        x_val, y_val,
        epochs=100,
        batch_size=32,
        checkpoint_dir=checkpoint_dir
    )
    
    # Save final model
    denoiser.save_model(data_dir / 'final_model.h5')
    
    # Example prediction
    test_prediction = denoiser.predict(x_test[:1])
    print(f"Prediction shape: {test_prediction.shape}")

if __name__ == '__main__':
    main()
