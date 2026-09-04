# PCA & Embeddings — Clustering (AUEB)

Exercise: cluster two text datasets using different embeddings, compare
clustering algorithms, evaluate against ground truth, and see whether PCA
improves results. Full task description: `docs/3. Clustering_with_PCA_Word_Embeddings (1).pdf`.

## Datasets

- **BBC News** (`data/bbc_news_test.csv`, not committed — see Setup): ~1,490
  articles across 5 categories (business, entertainment, politics, sport, tech).
- **20NewsGroups**: fetched at runtime via `scikit-learn` (`load_20newsgroups.py`),
  20 topics.

## Method

1. **Feature extraction** — embed text with TF-IDF (and optionally Word2Vec/FastText).
2. **k selection** — estimate the number of clusters *from the data alone*
   (elbow/silhouette sweep, HDBSCAN's own discovered count, dendrogram cut).
   The true number of categories (5 / 20) is never fed into this step —
   only used later, to score how close we got.
3. **Clustering** — K-Means, HDBSCAN, Agglomerative Clustering.
4. **Evaluation** — NMI, ARI, AMI against true labels; Silhouette Score as a
   label-free sanity check.
5. **PCA** — reduce embedding dimensionality, repeat steps 2–4, compare.
6. **Visualization** — bar charts comparing metrics across methods; bonus t-SNE 2D plot.

## Project structure

```
src/            reusable pipeline code (loading, features, k-selection, clustering,
                dimensionality reduction, evaluation, visualization)
scripts/        run_pipeline.py — the command-line entry point
data/           local only, gitignored (see Setup)
results/        metrics table + figures produced by the pipeline
docs/           original assignment PDF
```

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Place `bbc_news_test.csv` in `data/` (not tracked in git).

## Run

```
python scripts/run_pipeline.py
```

## Results

_(filled in once the pipeline runs end to end)_
