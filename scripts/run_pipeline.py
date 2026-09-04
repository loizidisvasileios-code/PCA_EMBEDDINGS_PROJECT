"""Command-line entry point: run the full clustering pipeline end to end.

Why this file exists: it's the one thing you actually execute (`python scripts/run_pipeline.py`).
It imports from src/ and orchestrates, in order: load both datasets -> build
embeddings -> discover k per dataset/embedding -> cluster with K-Means/HDBSCAN/
Agglomerative -> evaluate against true labels -> repeat with PCA-reduced
embeddings -> save the comparison table and plots to results/.

TODO (brainstorm before implementing): fill in the orchestration once each
src/ module has real logic behind it.
"""
