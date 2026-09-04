"""Score cluster assignments against true labels, and collect results across runs.

Why this module exists: this is the ONLY place in src/ that is allowed to read
ground-truth labels. Every other module works purely on embeddings/cluster
labels, so this boundary is what keeps the whole pipeline from "cheating" by
leaking label information into how k is chosen or how clusters are fit.

TODO (brainstorm before implementing):
- score(true_labels, predicted_labels) -> dict with NMI, ARI, AMI
- silhouette(embeddings, predicted_labels) -> float (label-free, sanity check)
- results table: append one row per (dataset, embedding, algorithm, k_used, PCA?, scores)
"""
