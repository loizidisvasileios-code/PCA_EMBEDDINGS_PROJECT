"""Reusable pipeline code for the PCA & Embeddings clustering project (AUEB).

Each module here has one job: loading data, building embeddings, picking k,
clustering, reducing dimensionality, evaluating, or visualizing. scripts/run_pipeline.py
wires them together; nothing in src/ should read ground-truth labels except evaluation.py.
"""
