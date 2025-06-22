import os
import numpy as np
import torch
import torchaudio
import pandas as pd
from pathlib import Path
from aves.aves import load_feature_extractor

# === Settings ===
base_input_dir = "GAT/zebra_finch/audio"  # Path to audio recordings
annotations_file = "GAT/zebra_finch/annotations.csv"  # Path to the annotations file
base_output_dir = "embeddings"  # Output directory
config_path = "aves/config/default_cfg_aves-base-all.json"
model_path = "aves-base-all.torchaudio.pt"
device = "cpu"
sample_rate = 16000

# Load the model
model = load_feature_extractor(config_path=config_path, model_path=model_path, device=device)

# Load the annotations CSV
df_annotations = pd.read_csv(annotations_file)
# Assuming the column with filenames is called "filename"
file_list = df_annotations["filename"].tolist()

max_embedding_size = 0
embeddings = []

print(f"Processing {len(file_list)} files listed in {annotations_file}")

for file in sorted(file_list):
    file_path = os.path.join(base_input_dir, file)
    if not os.path.isfile(file_path):
        print(f"Warning: file not found {file_path}")
        continue

    print(f"Processing file: {file_path}")
    try:
        waveform, sr = torchaudio.load(file_path)
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
            waveform = resampler(waveform)

        with torch.no_grad():
            features = model.extract_features(waveform)[-1]
            embedding = features.mean(dim=1).squeeze().cpu().numpy()
            max_embedding_size = max(max_embedding_size, embedding.shape[0])
            embeddings.append(embedding)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        continue

# Padding
padded_embeddings = []
for emb in embeddings:
    if emb.shape[0] < max_embedding_size:
        padding = np.zeros(max_embedding_size - emb.shape[0])
        padded_embeddings.append(np.concatenate([emb, padding]))
    else:
        padded_embeddings.append(emb)

# Save
if padded_embeddings:
    species_name = Path(base_input_dir).stem
    output_file = os.path.join(base_output_dir, f"{species_name}.npy")
    os.makedirs(base_output_dir, exist_ok=True)
    np.save(output_file, np.stack(padded_embeddings))
    print(f"Saved {len(padded_embeddings)} embeddings to {output_file}")
else:
    print("No valid embeddings created.")
