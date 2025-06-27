"""
Generalized analysis script for evaluating CycleGAN-generated audio fakes using dimensionality reduction (UMAP/t-SNE)
and clustering (KMeans, Spectral, Agglomerative, HDBSCAN, DBSCAN).

Outputs:
- Ground truth clustering accuracy
- Overlay plots of GT vs. fakes
- Fake clustering accuracy under semantic and acoustic assumptions
- Per-experiment summary tables

Requirements:
- realA, realB, fakeA, fakeB folders for each experiment (under data/{exp_id}/)
- Each folder must contain .npy latent vectors
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, DBSCAN
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import umap
import hdbscan

# === CONFIGURATION ===
data_root = "data"  # root path to experiment folders

# User provides experiment info here
datasets = {
    'zip1': {
        'name': 'cow_bird',
        'realA': 'bc/testA',
        'realB': 'bc/testB',
        'fakeA': 'results/bc_model_BtoA',
        'fakeB': 'results/bc_model_AtoB',
        'realA_label': 'cow',
        'realB_label': 'bird'
    },
    # Add more zips as needed
}

# === HELPERS ===
def load_vectors(folder, label):
    if not os.path.exists(folder):
        return [], []
    files = sorted([f for f in os.listdir(folder) if f.endswith('.npy')])
    vectors, labels = [], []
    for f in files:
        vec = np.load(os.path.join(folder, f))
        vectors.append(vec.flatten())
        labels.append(label)
    return vectors, labels


def clustering_algorithms(n_clusters):
    return {
        'KMeans': KMeans(n_clusters=n_clusters, random_state=0),
        'Spectral': SpectralClustering(n_clusters=n_clusters, assign_labels='kmeans'),
        'Agglomerative': AgglomerativeClustering(n_clusters=n_clusters),
        'DBSCAN': DBSCAN(eps=0.5),
        'HDBSCAN': hdbscan.HDBSCAN(min_cluster_size=5)
    }


# === MAIN PIPELINE ===
all_results = []

for zip_id, cfg in datasets.items():
    print(f"\n📂 Processing {zip_id} ({cfg['name']})")

    paths = {k: os.path.join(data_root, zip_id, v) for k, v in cfg.items() if isinstance(v, str)}
    all_data = {}
    for k, path in paths.items():
        vecs, labels = load_vectors(path, k)
        all_data[k] = {'X': vecs, 'y': labels}
        print(f"  Loaded {k} from {path}: {len(vecs)} files")

    # Skip if missing data
    if any(len(all_data[k]['X']) == 0 for k in ['realA', 'realB']):
        print("⚠️ Missing realA or realB data. Skipping.")
        continue

    # Combine real only
    X_real = np.array(all_data['realA']['X'] + all_data['realB']['X'])
    y_real = np.array(all_data['realA']['y'] + all_data['realB']['y'])
    le = LabelEncoder().fit(y_real)
    y_true = le.transform(y_real)
    n_clusters = len(np.unique(y_true))

    reducers = {
        'UMAP': umap.UMAP(n_components=2, random_state=42),
        't-SNE': TSNE(n_components=2, perplexity=10, random_state=42)
    }

    for red_name, reducer in reducers.items():
        emb = reducer.fit_transform(X_real)

        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        axs = axs.flatten()
        sns.scatterplot(x=emb[:, 0], y=emb[:, 1], hue=[cfg['realA_label'] if x == 'realA' else cfg['realB_label'] for x in y_real], ax=axs[0], s=40)
        axs[0].set_title(f"{red_name} + Ground Truth")

        best_acc, best_algo, best_pred = 0.0, None, None
        for i, (name, algo) in enumerate(clustering_algorithms(n_clusters).items(), 1):
            try:
                pred = algo.fit_predict(emb)
                acc = accuracy_score(y_true, pred) if len(set(pred)) > 1 else 0.0
                axs[i].set_title(f"{red_name} + {name}\nAcc: {acc:.2%}")
                sns.scatterplot(x=emb[:, 0], y=emb[:, 1], hue=pred, ax=axs[i], s=40, legend=False)
                if acc > best_acc:
                    best_acc, best_algo, best_pred = acc, name, pred
            except Exception as e:
                axs[i].text(0.5, 0.5, 'Error', ha='center')
                axs[i].set_title(f"{red_name} + {name}: Failed")

        plt.suptitle(f"{cfg['name']} – {red_name} Clustering", fontsize=16)
        plt.tight_layout()
        plt.show()

        # Overlay GT + Fakes
        if all(len(all_data[k]['X']) for k in ['fakeA', 'fakeB']):
            X_all = np.array(all_data['realA']['X'] + all_data['realB']['X'] + all_data['fakeA']['X'] + all_data['fakeB']['X'])
            y_all = all_data['realA']['y'] + all_data['realB']['y'] + all_data['fakeA']['y'] + all_data['fakeB']['y']
            emb_all = reducer.fit_transform(X_all)

            true_labels = [0 if 'A' in lbl else 1 for lbl in y_all if 'fake' in lbl]
            pred_labels = [best_pred[i] for i, lbl in enumerate(y_all) if 'fake' in lbl]
            fake_acc = accuracy_score(true_labels, pred_labels) if len(set(pred_labels)) > 1 else 0.0

            # Plot with legend
            fig, ax = plt.subplots(figsize=(7, 6))
            for lbl in np.unique(y_all):
                idx = np.array(y_all) == lbl
                color = 'red' if 'fakeA' in lbl else 'magenta' if 'fakeB' in lbl else 'skyblue' if 'realA' in lbl else 'peachpuff'
                ax.scatter(emb_all[idx, 0], emb_all[idx, 1], label=lbl.replace('realA', cfg['realA_label']).replace('realB', cfg['realB_label']), s=60 if 'fake' in lbl else 30, alpha=1.0 if 'fake' in lbl else 0.4, c=color)

            ax.set_title(f"{cfg['name']}\n{red_name} + {best_algo} GT + Fakes\nFake Clustering Accuracy: {fake_acc:.2%}")
            ax.legend(fontsize='small')
            plt.tight_layout()
            plt.show()

            all_results.append((cfg['name'], red_name, best_algo, fake_acc))

# === Summary Table ===
print("\n📊 Fake Accuracy Summary Table:")
df = pd.DataFrame(all_results, columns=["Experiment", "Projection", "Clustering", "FakeAcc"])
display(df.sort_values(by="FakeAcc", ascending=False))
