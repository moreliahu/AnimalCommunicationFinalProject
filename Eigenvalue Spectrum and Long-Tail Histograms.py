"""
Feature Matrix Analysis: Eigenvalue Spectrum and Long-Tail Histograms
----------------------------------------------------------------------
This script loads feature matrices (raw waveform, MFCC, VAE latent)
for multiple categories, computes eigenvalues of the covariance matrices,
and visualizes:
1. Log-log eigenvalue decay curves per category
2. Log-frequency histograms of feature value distributions ("long-tail")

Adapt file paths and category names as needed.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# List of category names (replace with actual labels)
categories = [
    "category1", "category2", "category3", "category4",
    "category5", "category6", "category7", "category8"
]

# Filename patterns for each encoding type
file_patterns = {
    "raw_waveform": "raw_waveform_matrix_{}.npy",
    "mfcc": "mfcc_matrix_{}_128.npy",
    "vae_latent": "vae_latent_matrix_{}.npy"
}

# Base directory containing saved feature matrices
base_dir = "/path/to/saved_features"

def load_matrices(matrix_type):
    """Load feature matrices for the given matrix type."""
    matrices = {}
    pattern = file_patterns[matrix_type]
    for cat in categories:
        path = os.path.join(base_dir, pattern.format(cat))
        if os.path.exists(path):
            matrices[cat] = np.load(path)
    return matrices

def get_eigenvalues(matrix):
    """Compute sorted absolute eigenvalues of the covariance matrix."""
    cov = np.cov(matrix.T)
    eigvals = np.linalg.eigvals(cov)
    return np.sort(np.abs(eigvals))[::-1]

# Process each encoding type
for matrix_type in ["raw_waveform", "mfcc", "vae_latent"]:
    matrices = load_matrices(matrix_type)

    if not matrices:
        print(f"No matrices found for: {matrix_type}")
        continue

    # Plot eigenvalue spectrum (log-log)
    plt.figure(figsize=(12, 6))
    for cat, mat in matrices.items():
        try:
            eigvals = get_eigenvalues(mat)
            plt.plot(eigvals, label=cat)
        except Exception as e:
            print(f"Error with {cat}: {e}")
    plt.title(f"{matrix_type.upper()} - Eigenvalues (log-log)")
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot value histograms (long-tail), one subplot per category
    fig, axs = plt.subplots(len(matrices), 1, figsize=(10, 2.5 * len(matrices)), sharex=True)
    if len(matrices) == 1:
        axs = [axs]  # Ensure it's iterable
    for i, (cat, mat) in enumerate(matrices.items()):
        axs[i].hist(mat.flatten(), bins=100, log=True, alpha=0.6, color=plt.cm.tab10(i))
        axs[i].set_title(f"{cat} - {matrix_type.upper()} Distribution")
        axs[i].set_ylabel("Log Frequency")
    axs[-1].set_xlabel("Value")
    plt.tight_layout()
    plt.show()
