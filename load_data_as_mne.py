# %% Dependencies
import mne

# Load conversion function from other script
from convert_to_mne import convert_to_mne_raw

# %% Load and Plot
# Absolute path to a faces_basic .mat file (edit for your machine)
data_path = r"C:\Users\Jonny\Neuromatch\Project\dataset\faces_basic\data\aa\aa_faceshouses.mat"

# Load and convert
raw = convert_to_mne_raw(data_path)

# Find stimulus events (used for epoching)
events = mne.find_events(
    raw,
    consecutive=True   
)

# Plot
raw.plot(
    picks="eeg",        # Don't plot stim chan, its scale overwhelms the eeg data
    scalings="auto"
)



# %%
