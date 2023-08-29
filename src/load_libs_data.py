import pandas as pd
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm

def load_h5_data(path):
    """
    Function for loading h5 format of the ChemCam extended calibration dataset.

    Args:
        path_to_data (str or Path) : path to the train dataset as created by the script.

    Returns:
        pd.DataFrame : X
        pd.Series : y
        entry to the file it originated from. 
    """
    loaded_data={
        'metadata':pd.read_hdf(path,key='metadata'),
        'spectra':pd.read_hdf(path,key='spectra')
    }

    with h5py.File(
      path,
      'r'
    ) as file:
        loaded_data['wvl'] = file['wvl'][()]


    spectra = loaded_data['spectra']
    spectra = spectra.iloc[:,1:]
    metadata = loaded_data['metadata']
    wavelengths = loaded_data['wvl']
    wavelengths = wavelengths[1:]

    return pd.DataFrame(np.array(spectra), columns=wavelengths), metadata 
