# %% Dependencies
import mne

# Load conversion function from other script
from convert_to_mne import convert_to_mne_raw

# %% Load and Plot
# Load and convert
raw = convert_to_mne_raw("faces_noise/faces_noise/data/aa/aa_faceshouses.mat")

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
