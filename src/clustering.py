"""Fit K-Means, HDBSCAN, and Agglomerative Clustering on a given embedding matrix.

Why this module exists: the assignment requires comparing all three algorithms
on the same embeddings. Each wrapper takes only what that algorithm actually
needs (k for K-Means/Agglomerative, density params for HDBSCAN) and returns
cluster labels in the same format so evaluation.py can treat them identically.

TODO (brainstorm before implementing):
- run_kmeans(embeddings, k) -> labels
- run_hdbscan(embeddings, min_cluster_size) -> labels (k is discovered, not passed in)
- run_agglomerative(embeddings, k) -> labels
"""
