# Cross-Species Clustering Evaluation

This script compares how well generated (fake) animal sounds resemble real animal vocalizations using dimensionality reduction and clustering techniques.

## Features
- Uses UMAP and t-SNE for 2D projections of latent vectors
- Clusters real (A, B) and fake (A→B, B→A) samples using multiple clustering algorithms
- Computes accuracy of clustering using true labels
- Evaluates **fake clustering accuracy** using two interpretations:
  - **Acoustic**: fakeA should resemble B, and fakeB should resemble A
  - **Semantic**: fakeA should resemble A, and fakeB should resemble B

## How to Use

### Input Structure

Place your data in:
```
data/
  zip1/
    cow_bird/
      bc/testA/
      bc/testB/
      results/bc_model_AtoB/
      results/bc_model_BtoA/
```

Each folder must contain `.npy` latent vector files.

### Run the Script

If using Jupyter Notebook, open and run `cross_species_clustering.ipynb`.

If using script:
```bash
python cross_species_clustering.py
```

## Outputs
- Clustering plots for real samples with accuracy scores
- Overlay plots with best clustering result and fake samples
- Console summary of fake classification accuracy (acoustic vs semantic)
