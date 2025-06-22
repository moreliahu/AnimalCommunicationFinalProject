# Feature Matrix Analysis: Eigenvalues and Distributions

This script performs visual analysis of feature matrices extracted from audio recordings. It helps characterize different audio categories based on:

1. **Eigenvalue Spectra** — showing how variance is distributed across dimensions (log-log plot).
2. **Long-Tail Distributions** — showing the value distribution of all entries in each matrix (log-scaled histogram).

## 🧪 Supported Encoding Types

- `raw_waveform`: raw audio feature matrices
- `mfcc`: Mel-Frequency Cepstral Coefficients
- `vae_latent`: latent space vectors from a trained VAE

## 📂 Expected Folder Structure

```bash
/path/to/saved_features/
├── raw_waveform_matrix_category1.npy
├── mfcc_matrix_category1_128.npy
├── vae_latent_matrix_category1.npy
├── ...
```

> Update the `categories` list and `base_dir` path in the script as needed.

## 📈 Output

The script generates:
- A log-log plot of sorted eigenvalues for each category
- A set of histograms (one per category) showing the long-tail distribution of matrix values

These visualizations help evaluate:
- Information richness per feature type
- Compression effectiveness (e.g., via VAE)
- Distribution sparsity or redundancy

## 📝 Usage

Run the script in a Python environment with `numpy`, `matplotlib`, and other standard scientific packages installed.

```bash
python feature_analysis.py
```

Make sure to adjust paths and category names before running.

## 👩‍🔬 Authors

Created by [Your Name / Team], as part of the [Your Project Name] research project.
