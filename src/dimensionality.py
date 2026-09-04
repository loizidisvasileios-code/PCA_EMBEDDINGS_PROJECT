"""Reduce embedding dimensionality with PCA before re-clustering.

Why this module exists: the assignment asks us to see whether PCA improves
clustering quality / brings cluster counts closer to the true k. Keeping this
separate lets us re-run the exact same k_selection + clustering + evaluation
steps on the reduced embeddings and compare against the raw-embedding results.

TODO (brainstorm before implementing):
- fit_pca(embeddings, n_components) -> reduced matrix
- explained_variance_curve(embeddings, max_components) -> for choosing n_components
"""
