import librosa
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

# Define paths
AUDIO_DIR = "labled_data/zebraFinch/audio"
ANNOTATIONS_PATH = "labled_data/zebraFinch/annotations.csv"

# Load the annotations file
df = pd.read_csv(ANNOTATIONS_PATH)

# Get the list of valid filenames (assuming the column is named 'filename')
valid_files = df['fn'].tolist()

features = []

for file in tqdm(valid_files):
    filepath = os.path.join(AUDIO_DIR, file)
    if not os.path.isfile(filepath):
        print(f"Warning: file {filepath} not found. Skipping.")
        continue

    # Load audio file
    y, sr = librosa.load(filepath, sr=None)

    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Take mean over time axis
    mfcc_mean = mfcc.mean(axis=1)

    features.append(mfcc_mean)

# Stack features into a matrix
features = np.stack(features)

# Save features matrix to file
np.save("features.npy", features)

print(f"Feature matrix saved with shape {features.shape}")
