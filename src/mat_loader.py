"""
Module for loading and processing MATLAB (.mat) files.
"""

import scipy.io
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Union, Optional

class MatLoader:
    """A class to load and process MATLAB files."""
    
    def __init__(self, file_path: Union[str, Path]):
        """
        Initialize the MatLoader with a file path.
        
        Args:
            file_path (Union[str, Path]): Path to the .mat file
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        if self.file_path.suffix != '.mat':
            raise ValueError(f"File must be a .mat file, got: {self.file_path}")
        
        self._data = None
    
    def load_mat(self) -> Dict:
        """
        Load the .mat file and return a cleaned dictionary of its contents.
        
        Returns:
            Dict: Dictionary containing the data from the .mat file,
                 excluding MATLAB's internal variables (starting with '_')
        """
        raw_data = scipy.io.loadmat(str(self.file_path))
        # Remove MATLAB's internal variables (keys starting with '_')
        self._data = {k: v for k, v in raw_data.items() if not k.startswith('_')}
        return self._data
    
    def to_dataframe(self, variable_name: Optional[str] = None) -> pd.DataFrame:
        """
        Convert the .mat file contents to a pandas DataFrame.
        
        Args:
            variable_name (str, optional): Specific variable name to convert to DataFrame.
                                        If None, tries to convert all variables.
        
        Returns:
            pd.DataFrame: DataFrame containing the data
        
        Raises:
            ValueError: If the data structure is not compatible with DataFrame conversion
        """
        if self._data is None:
            self.load_mat()
            
        if variable_name:
            if variable_name not in self._data:
                raise KeyError(f"Variable '{variable_name}' not found in .mat file")
            data = self._data[variable_name]
            return pd.DataFrame(data)
        
        # If no specific variable name is provided, try to convert all variables
        try:
            return pd.DataFrame({k: pd.Series(v.flatten()) 
                               for k, v in self._data.items()
                               if isinstance(v, np.ndarray)})
        except Exception as e:
            raise ValueError(f"Could not convert data to DataFrame: {str(e)}")
    
    def get_variables(self) -> Dict[str, tuple]:
        """
        Get information about variables in the .mat file.
        
        Returns:
            Dict[str, tuple]: Dictionary with variable names as keys and their shapes as values
        """
        if self._data is None:
            self.load_mat()
            
        return {name: array.shape for name, array in self._data.items()}
    
    def save_to_csv(self, output_path: Union[str, Path], 
                    variable_name: Optional[str] = None) -> None:
        """
        Save the data to a CSV file.
        
        Args:
            output_path (Union[str, Path]): Path where to save the CSV file
            variable_name (str, optional): Specific variable to save to CSV
        """
        df = self.to_dataframe(variable_name)
        output_path = Path(output_path)
        df.to_csv(output_path, index=False)
    
    def get_tensorflow_format(self, variable_name: Optional[str] = None) -> np.ndarray:
        """
        Reshape the 4D data into format suitable for TensorFlow CNN.
        Converts from (Time, Space, Channel, Samples) to (Samples, Channel, Height, Width)
        
        Args:
            variable_name (str, optional): Specific variable to reshape.
                                         If None, uses the first available variable.
        
        Returns:
            np.ndarray: Reshaped data in format (Samples, Channel, Height, Width)
        
        Raises:
            ValueError: If the data is not 4D or if the variable is not found
        """
        if self._data is None:
            self.load_mat()
            
        # Get the data to reshape
        if variable_name:
            if variable_name not in self._data:
                raise KeyError(f"Variable '{variable_name}' not found in .mat file")
            data = self._data[variable_name]
        else:
            # Use the first available variable if none specified
            data = next(iter(self._data.values()))
        
        # Check dimensions
        if len(data.shape) != 4:
            raise ValueError(f"Data must be 4D, got shape {data.shape}")
            
        time, space, channel, samples = data.shape
        
        # Reshape and transpose the data
        # From: (Time, Space, Channel, Samples)
        # To: (Samples, Channel, Height=Time, Width=Space)
        reshaped_data = np.transpose(data, (3, 2, 0, 1))
        
        return reshaped_data


# Example usage
def main():
    # Example usage of the MatLoader class
    data_dir = Path(__file__).parent.parent / 'Data'
    
    # Load and process blur data
    blur_file = data_dir / 'Blur_rho_16g_full_data.mat'
    blur_loader = MatLoader(blur_file)
    print("Variables in blur data:")
    print(blur_loader.get_variables())
    
    # Get data in TensorFlow format
    print("\nReshaping blur data for TensorFlow...")
    tf_data = blur_loader.get_tensorflow_format()
    print(f"TensorFlow format shape: {tf_data.shape}")
    print(f"Format interpretation:")
    print(f"- Number of samples: {tf_data.shape[0]}")
    print(f"- Number of channels: {tf_data.shape[1]}")
    print(f"- Height (Time dimension): {tf_data.shape[2]}")
    print(f"- Width (Space dimension): {tf_data.shape[3]}")
    
    # Load and process original data
    original_file = data_dir / 'Original_rho_full_data.mat'
    original_loader = MatLoader(original_file)
    print("\nVariables in original data:")
    print(original_loader.get_variables())
    
    # Get original data in TensorFlow format
    print("\nReshaping original data for TensorFlow...")
    tf_data_original = original_loader.get_tensorflow_format()
    print(f"TensorFlow format shape: {tf_data_original.shape}")

if __name__ == '__main__':
    main()
