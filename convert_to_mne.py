import numpy as np
from scipy.io import loadmat
import mne

def convert_to_mne_raw(file_path):
    """
    Takes a .mat file from the faces_basic data set and converts it to an MNE raw array

    Args:
        file_path (path-like):  path to the .mat file
            
    Returns:
        MNE raw object                
    """

    raw_data = loadmat(file_path)

    data    = raw_data["data"].T       
    sfreq   = raw_data["srate"]
    events  = raw_data["stim"].T

    data_stim = np.vstack([data, events])

    ch_names = [f"EEG{i}" for i in range(data_stim.shape[0])]
    ch_names[-1] = "STIM"
    ch_types = ["eeg"] * data_stim.shape[0]
    ch_types[-1] = "stim"

    info = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types=ch_types
    )

    raw = mne.io.RawArray(data_stim, info)

    return raw