"""Estimate the number of clusters k directly from the embeddings, without looking at labels.

Why this module exists: K-Means and Agglomerative both need a k up front, and
we are not allowed to just read k off the label column (that would be cheating
the assignment's own stated ground truth). This module answers "what does the
data itself suggest?" via elbow/silhouette sweeps and dendrogram inspection;
HDBSCAN's discovered cluster count (from clustering.py) is used as a cross-check.

TODO (brainstorm before implementing):
- sweep_kmeans_k(embeddings, k_range) -> inertia + silhouette per k, for plotting an elbow
- suggest_k_from_silhouette(embeddings, k_range) -> best k by silhouette score
"""
