# %% Dependencies
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from mne.decoding import SlidingEstimator, cross_val_multiscore

# %% Load data
data = loadmat("faces_basic/faces_basic/data/aa/aa_faceshouses.mat")

# %% Get data
stims = data["stim"].squeeze()
time_series = data["data"]

# %% Labels
stims[(stims >= 1) & (stims <= 50)] = 1
stims[(stims >= 51) & (stims <= 100)] = 2
labs = [0 for _ in range(150)] + [1 for _ in range(150)]

# %% Extract Epochs
faces = []
houses = []

for i, event in enumerate(stims):

    if (event == 1) and (stims[i-1] == 101):
        house_epoch = time_series[i-200:i+400, :]
        houses.append(house_epoch)
        continue

    if (event == 2) and (stims[i-1] == 101):
        face_epoch = time_series[i-200:i+400, :]
        faces.append(face_epoch)


full_data = np.concatenate(
    [np.stack(faces), np.stack(houses)],
    axis=0
)
full_data = np.transpose(full_data, (0, 2, 1))

# %% Baseline correct
baseline = full_data[:, :, :200].mean(axis=2, keepdims=True)
full_data = full_data - baseline

# %% Model set up
clf = make_pipeline(
    StandardScaler(),         # Z-score the data
    LogisticRegression(
        penalty="l1",         # L1 (lasso) regularisation
        solver='liblinear'
    )
)

# Sliding estimator to calcualte model at each time point
sliding_clf = SlidingEstimator(
    clf,
    scoring='accuracy',
    n_jobs=1
)

# Folds for k-fold cross validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# %% Cross validate
scores = cross_val_multiscore(
    sliding_clf,
    full_data,           # shape (n_epochs, n_features, n_times)
    labs,                # shape (n_epochs,)
    cv=cv,
    n_jobs=1
)

# Average across CV folds
auc_timecourse = scores.mean(axis=0)

# %% Plot
plt.plot(auc_timecourse)
plt.axvline(200, color="black")