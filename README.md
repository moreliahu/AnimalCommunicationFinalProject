# AnimalCommunicationFinalProject

This repository contains the final project for our research on animal communication, focusing on signal analysis, representation learning, and deep generative models applied to animal vocalizations.

---

## 📁 Project Structure

```
AnimalCommunicationFinalProject/
├── AVES/                  # AVES-based feature extraction and embeddings
├── CycleGan/              # CycleGAN training & testing for audio-to-audio translation
├── GAT/                   # Graph Attention Network-related code and experiments
├── README_feature_analysis.md
├── Eigenvalue Spectrum and Long-Tail Histogram Analysis.ipynb
└── README.md              # This file
```

---

## 📚 Components Overview

### 🔬 `AVES/`
Contains scripts and pretrained model config for extracting embeddings using the AVES audio encoder. Used for high-level representations of vocalizations.

### 🔁 `CycleGan/`
Includes scripts for training and testing CycleGAN models on spectrogram data to perform domain translation between animal species (e.g., bird → cow). Also includes inference and generation code.

### 🧠 `GAT/`
Implements a Graph Attention Network architecture applied to bird vocalization data. This part includes graph-based representations of syllables and classification experiments.

### 📊 `README_feature_analysis.md` + Notebook
Includes analysis tools to:
- Compute eigenvalues of feature matrices
- Plot log-log eigenvalue decay
- Visualize long-tail histograms of matrix values

Useful for comparing structure and compression effects between raw waveform, MFCC, and VAE-encoded features.

---

## 🚀 How to Get Started

### Requirements
Ensure the following Python packages are installed:
```bash
numpy
matplotlib
librosa
torch
torchaudio
scikit-learn
networkx
```

### Run AVES Feature Extraction
See scripts inside the `AVES/` directory for details on how to extract embeddings per file using the AVES model.

### Train CycleGAN
Inside `CycleGan/`, use the following command:
```bash
python train_spec.py --dataroot <path_to_data> --name <model_name> --model cycle_gan --gpu_ids <id> --dataset_mode unaligned_spec --serial_batches
```

See the `HOW_TO_TRAIN.txt` for a full explanation of parameters.

### Run Eigenvalue & Histogram Analysis
Use the script `feature_analysis.py` or the Jupyter Notebook to visualize feature structure.

---

## 👩‍🔬 Authors

Mor Eliahu & Emily Smetanov  
Supervised by Khen Cohen  
In collaboration with Prof. Yossi Yovel's lab at Tel Aviv University

---

## 📄 License

This project is for academic and research use only.
