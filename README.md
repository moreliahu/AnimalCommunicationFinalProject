# AnimalCommunicationFinalProject

This repository contains the final project for our research on animal communication, focusing on signal analysis, representation learning, and deep generative models applied to animal vocalizations.

---

## 📁 Project Structure

```
AnimalCommunicationFinalProject/
├── AVES/                            # AVES-based feature extraction and embeddings
├── CycleGan/                        # CycleGAN training & testing for audio-to-audio translation
├── CycleGAN Tests/                  # Inference, visualization & transformation accuracy tools
├── CrossSpeciesClustering/         # Cross-species UMAP/t-SNE clustering & evaluation of CycleGAN fakes
├── GAT/                             # Graph Attention Network-related code and experiments
├── Eigenvalue Spectrum and Long-Tail Histogram Analysis.ipynb
├── README_feature_analysis.md
└── README.md                        # This file
```

---

## 📚 Components Overview

### 🔬 `AVES/`
Contains scripts and pretrained config files for extracting AVES embeddings. These embeddings provide semantic audio representations for downstream tasks like classification and clustering.

### ↻ `CycleGan/`
Contains training scripts, data format examples, and generator/inference logic. Used to translate spectrograms between species (e.g., bird → cow or dog ↔ cat) using CycleGANs.

### 🧪 `CycleGAN Tests/`
A companion folder for visual and numerical evaluation of CycleGAN outputs. Includes:
- Side-by-side audio comparisons
- MSE / Cosine distance evaluation
- Synthetic spectrogram generation

### 🔍 `CrossSpeciesClustering/`
**New!** End-to-end pipeline for evaluating CycleGAN-generated fakes:
- Runs dimensionality reduction via **UMAP** and **t-SNE**
- Clusters real vs. fake embeddings using KMeans, Spectral, HDBSCAN, and Agglomerative
- Produces summary plots and tables:
  - Clustering accuracy (real calls)
  - GT vs. Fake Overlay Plots (colored, annotated)
  - Fake classification accuracy under **semantic** (e.g. fakeA ↔ A) and **acoustic** (e.g. fakeA ↔ B) hypotheses
- Can be used as a sanity check for learned transformation fidelity

> 📌 See the notebook or script in `CrossSpeciesClustering/` for full instructions and visual output samples.

### 🧠 `GAT/`
Graph Attention Network over bird syllables. Includes:
- Graph construction from MFCC/VAE embeddings
- Attention-weighted classification
- Comparison of spectral features vs. deep-learned features in GATs

### 📊 `README_feature_analysis.md` + Notebook
Includes eigenvalue spectrum and histogram tools for analyzing latent space compression and representational decay in:
- Raw waveform
- MFCC
- VAE encodings

---

## 🚀 Getting Started

### Requirements
Install required packages:
```bash
pip install numpy matplotlib librosa torch torchaudio scikit-learn networkx umap-learn hdbscan seaborn
```

### AVES Feature Extraction
```bash
cd AVES/
# Run embedding script on a folder of `.wav` files
python extract_embeddings.py --input_dir data/bird --output_dir saved_features/
```

### CycleGAN Training
```bash
cd CycleGan/
python train_spec.py --dataroot data/ --name bird2cow_model --model cycle_gan --dataset_mode unaligned_spec
```

### Cross-Species Clustering Analysis
```bash
cd CrossSpeciesClustering/
python analyze_clustering.py
```

- Outputs include:
  - UMAP/t-SNE plots of real calls
  - Clustered results by algorithm
  - Overlay of real & fakes with accuracy annotations
  - Summary table for fake calls under both semantic and acoustic similarity goals

---

## 👩‍💻 Authors

**Mor Eliahu & Emily Smetanov**  
Supervised by **Khen Cohen**  
In collaboration with **Prof. Yossi Yovel**, Tel Aviv University

---

## 📄 License

This project is for academic and non-commercial research purposes only.
