# AnimalCommunicationFinalProject

This repository contains the final project for our research on animal communication, focusing on signal analysis, representation learning, and deep generative models applied to animal vocalizations.

---

## 📁 Project Structure

```
AnimalCommunicationFinalProject/
├── AVES/                            # AVES-based feature extraction and embeddings
├── CycleGan/                        # CycleGAN training & testing for audio-to-audio translation
├── CycleGAN Tests/                  # Inference, visualization & transformation accuracy tools
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
- Full pipeline for evaluating CycleGAN-generated fakes:
  - Runs dimensionality reduction via **UMAP** and **t-SNE**
  - Clusters real vs. fake embeddings using KMeans, Spectral, HDBSCAN, and Agglomerative
  - Produces summary plots and tables:
    - Clustering accuracy for real-only calls
    - GT vs. Fake Overlay Plots (colored, annotated)
    - Fake classification accuracy under:
      - **Semantic** hypothesis: fakeA ↔ A, fakeB ↔ B
      - **Acoustic** hypothesis: fakeA ↔ B, fakeB ↔ A
  - Visualizations include per-zip tables and color-coded species names

> 📌 See the notebook or script in `CycleGAN Tests/` for full instructions and visual output samples.

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

### CycleGAN Evaluation
```bash
cd CycleGAN Tests/
# Evaluate generated fakes via similarity or metrics
python evaluate_fakes.py --real_dir testA --fake_dir results/modelAtoB
```

### Clustering & Fake Evaluation
```bash
cd CycleGAN Tests/
python analyze_clustering.py
```

- Outputs include:
  - UMAP/t-SNE plots of real calls
  - Clustered results by algorithm
  - Overlay of real & fakes with accuracy annotations
  - Summary tables per zip
  - Accuracy for both **semantic** and **acoustic** hypotheses

---

## 👩‍💻 Authors

**Mor Eliahu & Emily Smetanov**  
Supervised by **Khen Cohen**  
In collaboration with **Prof. Yossi Yovel**, Tel Aviv University

---

## 📄 License

This project is for academic and non-commercial research purposes only.
