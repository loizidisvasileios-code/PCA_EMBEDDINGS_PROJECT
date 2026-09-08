"""Reduce embedding dimensionality and find the reduction that clusters best.

Why this module exists: the assignment asks us to use PCA to improve the
clustering results. Our own diagnostics made this load-bearing rather than
cosmetic: on raw TF-IDF (24-27k dimensions) all three algorithms failed to
find usable structure (near-zero silhouette for K-Means, singleton chaining
for Agglomerative, 0-2 clusters with huge noise for HDBSCAN) -- the classic
curse of dimensionality.

The question this module answers is purely practical: **at how many
dimensions does the clustering actually get better?** We do not care how
much variance a reduction retains -- only whether the clusters improve. So
sweep_n_components tries several reduction sizes, re-runs the label-free
k-sweeps on each, and reports the best silhouette each size achieved. The
winner is then simply the row with the best score.

Two reducers, chosen automatically by input type:
- sparse input (TF-IDF): TruncatedSVD. scikit-learn's PCA cannot accept
  sparse matrices at all, because centering the data (PCA's first step)
  would destroy sparsity and blow up memory. TruncatedSVD is the same
  underlying SVD without that centering step, and applied to TF-IDF it is
  the standard technique known as LSA (Latent Semantic Analysis).
- dense input (Word2Vec/FastText document vectors): ordinary PCA, since
  there is no sparsity to preserve and centering is free.

Note on ordering: the standard text pipeline is embed -> reduce ->
re-normalize -> cluster, because SVD/PCA output is no longer unit-length.
We do not re-normalize here: clustering.py and k_selection.py already call
utils.l2_normalize internally (and cosine distance is scale-invariant), so
the re-normalization happens automatically downstream.
"""

# pandas.DataFrame is the shared output shape for the sweep table, matching
# the sweep tables produced by k_selection.py and clustering.py.
import pandas as pd

# issparse tells us which reducer to dispatch to; it is the whole basis of
# the mixed sparse/dense strategy described in the module docstring.
from scipy.sparse import issparse

# PCA handles the dense case, TruncatedSVD the sparse one.
from sklearn.decomposition import PCA, TruncatedSVD

# The k-sweeps are how we judge whether a given reduction clusters better:
# each one returns a silhouette per k, and best_k_by_silhouette picks the
# winning row. Importing them here keeps the "which reduction is best?"
# question in one place instead of duplicating sweep logic.
from k_selection import best_k_by_silhouette, sweep_agglomerative_k, sweep_kmeans_k


def reduce_dimensions(embeddings, n_components: int, seed: int = 42) -> tuple:
    """Reduce embeddings to n_components dimensions, returning (reduced, model).

    Dispatches on input type: TruncatedSVD for sparse (TF-IDF), PCA for
    dense (Word2Vec/FastText). Callers do not need to know or care which one
    ran -- both return a dense (n_documents, n_components) matrix.
    """
    # issparse() is the single branch point between the two reducers; every
    # other line below is shared behaviour.
    if issparse(embeddings):
        # TruncatedSVD works directly on the sparse matrix without ever
        # densifying it, which is the entire reason we cannot just use PCA.
        model = TruncatedSVD(n_components=n_components, random_state=seed)
    else:
        # PCA centers the data first (subtracts the mean of each feature),
        # which is what makes it "true" PCA rather than plain SVD.
        model = PCA(n_components=n_components, random_state=seed)

    # fit_transform learns the projection and immediately applies it,
    # returning the reduced matrix in one call.
    reduced = model.fit_transform(embeddings)

    return reduced, model


def sweep_n_components(
    embeddings,
    component_values=(2, 5, 10, 20, 50, 100, 200, 300),
    k_range=range(2, 31),
    seed: int = 42,
) -> pd.DataFrame:
    """Try several reduction sizes and report which one clusters best.

    For each value in component_values: reduce the embeddings to that many
    dimensions, then re-run the K-Means and Agglomerative k-sweeps on the
    reduced data (k_range wide enough to cover both datasets' true category
    counts without ever being told them). Records the best silhouette each
    algorithm reached and at which k.

    Returns one row per reduction size, with columns:
    n_components, kmeans_best_k, kmeans_best_silhouette,
    agglomerative_best_k, agglomerative_best_silhouette.
    Reading the winner is then just "which row has the highest silhouette".
    """
    # A reduction can never produce more dimensions than the data has to
    # give: TruncatedSVD requires n_components < n_features, and PCA
    # requires n_components <= min(n_samples, n_features). Subtracting one
    # from the smaller of the two satisfies both rules, and any requested
    # size above it is skipped rather than crashing the whole sweep (this
    # matters for Word2Vec/FastText, which only have 100 dimensions to
    # start with, so 200 and 300 are simply not applicable there).
    max_allowed = min(embeddings.shape[0], embeddings.shape[1]) - 1

    rows = []
    for n_components in component_values:
        if n_components > max_allowed:
            continue

        # The reduced matrix is what every downstream measurement sees;
        # the fitted model itself is not needed here, hence the throwaway.
        reduced, _ = reduce_dimensions(embeddings, n_components=n_components, seed=seed)

        # Exactly the same sweeps we ran on the un-reduced embeddings, so
        # the numbers are directly comparable against the raw baseline.
        kmeans_sweep = sweep_kmeans_k(reduced, k_range=k_range, seed=seed)
        agglomerative_sweep = sweep_agglomerative_k(reduced, k_range=k_range)

        rows.append(
            {
                "n_components": n_components,
                # best_k_by_silhouette reports WHICH k won; .max() reports
                # HOW GOOD that winning k was. We need both: a good score
                # at an implausible k is a different story from a good
                # score at a plausible one.
                "kmeans_best_k": best_k_by_silhouette(kmeans_sweep),
                "kmeans_best_silhouette": kmeans_sweep["silhouette"].max(),
                "agglomerative_best_k": best_k_by_silhouette(agglomerative_sweep),
                "agglomerative_best_silhouette": agglomerative_sweep["silhouette"].max(),
            }
        )

    return pd.DataFrame(rows)


# Running this file directly answers the actual question for both datasets
# on TF-IDF (the embedding that failed worst without reduction): does PCA
# rescue the clustering, and at how many dimensions? Results are saved to
# results/ so they stay inspectable after the run.
if __name__ == "__main__":
    from pathlib import Path

    from data_loading import load_20newsgroups, load_bbc_news
    from features import build_tfidf

    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    for name, df in [("BBC", load_bbc_news()), ("20NG", load_20newsgroups())]:
        print(f"\n{name}")

        tfidf_matrix, _ = build_tfidf(df["text"])

        sweep = sweep_n_components(tfidf_matrix)

        sweep_path = results_dir / f"pca_sweep_{name.lower()}_tfidf.csv"
        sweep.to_csv(sweep_path, index=False)

        print(sweep.to_string(index=False))
        print(f"saved to {sweep_path}")
