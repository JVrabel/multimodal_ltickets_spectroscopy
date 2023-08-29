"""
Contains functionality for creating PyTorch DataLoaders for 
LIBS benchmark classification dataset.
"""

import os
import torch
from torch.utils.data import DataLoader
from load_libs_data import load_h5_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, MinMaxScaler
import numpy as np


NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str, 
    #test_dir: str, 
    batch_size: int, 
    device: torch.device,
    num_workers: int=NUM_WORKERS, 
    split_rate: float=0.8,
    random_st: int=102
    ):
    """Creates training and validation DataLoaders.

    Takes in a training directory directory path and split the data
    to train/validation. After, it turns them into PyTorch Datasets and 
    then into PyTorch DataLoaders.

    Args:
    train_dir: Path to training directory.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.

    Returns:
    A tuple of (train_dataloader, test_dataloader, metadata).
    Example usage:
        train_dataloader, test_dataloader, val_metadata = \
        = create_dataloaders(train_dir=path/to/train_dir,
                                batch_size=32,
                                num_workers=4)
    """

    data, metadata = load_h5_data(train_dir)

    metadata = metadata.loc[:,'SiO2':'K2O']
    metadata = metadata.drop(['FeT', 'FeO', 'Fe2O3'] , axis = 1)
    metadata = metadata.drop(metadata.columns[[3]], axis=1)

    wavelengths = data.columns

    X_train, X_val, y_train, y_val = train_test_split(data, metadata, test_size=split_rate, random_state=random_st, stratify=samples, shuffle = True)
    del data, metadata

    if True:
      scaler =  Normalizer()
      X_train = scaler.fit_transform(X_train)
      X_val = scaler.fit_transform(X_val)

    # Convert data to torch tensors
    X_train = torch.from_numpy(X_train).unsqueeze(1).float() # Add extra dimension for channels
    X_val = torch.from_numpy(X_val).unsqueeze(1).float() # Add extra dimension for channels
    y_train = torch.from_numpy(np.array(y_train)).long()
    y_val = torch.from_numpy(np.array(y_val)).long()

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # If available, move data to the GPU
    X_train.to(device)
    X_val.to(device) 
    y_train.to(device)
    y_val.to(device)




    # Create PyTorch DataLoader objects for the training and validation sets
    train_dataloader = DataLoader(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(X_val, y_val, batch_size=batch_size, shuffle=False)


    return train_dataloader, val_dataloader, y_val
