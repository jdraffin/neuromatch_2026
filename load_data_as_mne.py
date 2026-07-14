# %% Dependencies
import mne

# Load builder function from the pipeline
from preprocessing_pipeline import to_raw

# %% Load and Plot
# Absolute path to a faces_basic .mat file (edit for your machine)
data_path = r"C:\Users\Jonny\Neuromatch\Project\dataset\faces_basic\data\aa\aa_faceshouses.mat"

# Load and convert (unfiltered for viewing)
raw, _ = to_raw(data_path, notch=None, bandpass=None)

# Find stimulus events (used for epoching)
events = mne.find_events(
    raw,
    consecutive=True   
)

# Plot
raw.plot(
    picks="ecog",       # Don't plot stim chan, its scale overwhelms the ecog data
    scalings="auto"
)



# %%
