# src/data_processing.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.utils import location_to_grid

class DataProcessor:
    def __init__(self, config):
        self.config = config

    def load_and_prepare_data(self):
        file_path = self.config['paths']['labeled_data']
        print(f"Loading and preparing data from {file_path}...")
        data = pd.read_csv(file_path)

        if 'location' not in data.columns:
            raise ValueError("CSV file must contain a 'location' column.")
        
        grid_coords = data['location'].apply(location_to_grid)
        data['grid_x'] = grid_coords.apply(lambda x: x[0])
        data['grid_y'] = grid_coords.apply(lambda x: x[1])
        data.dropna(subset=['grid_x', 'grid_y'], inplace=True)

        rssi_cols = [col for col in data.columns if col.startswith('b30')]
        X = data[rssi_cols].values.astype(np.float32)

        y_coords = data[['grid_x', 'grid_y']].values
        unique_coords, y_classes = np.unique(y_coords, axis=0, return_inverse=True)
        self.idx_to_coords = {idx: tuple(coord) for idx, coord in enumerate(unique_coords)}
        
        seq_len = self.config['data']['sequence_length']
        X_seq, y_seq = [], []
        for i in range(len(X) - seq_len + 1):
            X_seq.append(X[i:i + seq_len])
            y_seq.append(y_classes[i + seq_len - 1])
        
        X_seq, y_seq = np.array(X_seq, dtype=np.float32), np.array(y_seq)
        print(f"Data preparation complete. Shape: {X_seq.shape}")
        return X_seq, y_seq, self.idx_to_coords

    def split_data(self, X_seq, y_seq):
        test_r = self.config['data']['test_ratio']
        seed = self.config['seed']
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X_seq, y_seq, test_size=test_r, stratify=y_seq, random_state=seed
        )
        print(f"Data split complete: Train/Val set={X_train_val.shape[0]}, Test set={X_test.shape[0]}")
        return X_train_val, X_test, y_train_val, y_test