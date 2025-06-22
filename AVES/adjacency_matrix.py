import numpy as np
from sklearn.neighbors import NearestNeighbors

# Load the feature matrix
features = np.load('features.npy')  # Shape: (2941, 13)

# Define the number of neighbors
k_neighbors = 5  # You can change this value if needed

# Fit Nearest Neighbors model using cosine distance
nn = NearestNeighbors(n_neighbors=k_neighbors + 1, metric='cosine')  # +1 to include self
nn.fit(features)

# Find neighbors
distances, indices = nn.kneighbors(features)

# Initialize adjacency matrix
num_nodes = features.shape[0]
adj_matrix = np.zeros((num_nodes, num_nodes))

# Fill adjacency matrix with similarity (1 - distance)
for i in range(num_nodes):
    for j, dist in zip(indices[i][1:], distances[i][1:]):  # Skip the first neighbor (itself)
        adj_matrix[i, j] = 1 - dist  # Convert cosine distance to similarity

# Save adjacency matrix to file
np.save('adjacency_matrix.npy', adj_matrix)

print("Adjacency matrix created and saved with shape:", adj_matrix.shape)
