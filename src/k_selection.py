"""Estimate the number of clusters k directly from the embeddings, without looking at labels.

Why this module exists: K-Means and Agglomerative both need a k up front, and
we are not allowed to just read k off the label column (that would be cheating
the assignment's own stated ground truth). This module answers "what does the
data itself suggest?" via silhouette-score sweeps (and inertia, for an elbow
plot) across a range of k. HDBSCAN needs no k and is therefore out of scope
here -- its own discovered cluster count is produced by clustering.py instead.
"""

# pandas.DataFrame is the shared output shape: one row per k tried, so
# visualization.py can plot elbow/silhouette curves straight from it.
import pandas as pd

# AgglomerativeClustering and KMeans are the two algorithms that need a k
# chosen before they can be fit at all.
from sklearn.cluster import AgglomerativeClustering, KMeans

# silhouette_score is our label-free way to judge a clustering's quality:
# how much closer, on average, is each point to its own cluster than to the
# next-nearest one. Works without ever touching ground-truth labels.
from sklearn.metrics import silhouette_score

# l2_normalize makes Euclidean distance behave like cosine distance -- see
# utils.py and our conversation for why that matters for text embeddings.
# cosine_distance_matrix builds the n x n distance matrix Agglomerative needs
# (see sweep_agglomerative_k for why: it cannot take sparse input) -- shared
# with clustering.py so both stages use the exact same distances.
from utils import cosine_distance_matrix, l2_normalize


def sweep_kmeans_k(embeddings, k_range=range(2, 21), seed: int = 42) -> pd.DataFrame:
    """Fit K-Means for every k in k_range and record inertia + silhouette.

    embeddings: TF-IDF/Word2Vec/FastText matrix (sparse or dense) for one
    dataset. Normalized internally, so callers can pass raw build_* output.
    Returns a DataFrame with columns k, inertia, silhouette -- one row per k,
    ready to plot as an elbow curve (inertia) or a silhouette curve.
    """
    # Normalize once up front rather than per-k: the embeddings do not change
    # between iterations, only k does.
    normalized = l2_normalize(embeddings)

    rows = []
    for k in k_range:
        # n_init="auto" lets scikit-learn pick a sensible number of random
        # restarts for us; random_state=seed makes the chosen centroids (and
        # therefore inertia/silhouette) reproducible across runs.
        model = KMeans(n_clusters=k, random_state=seed, n_init="auto")

        # fit_predict both fits the model and returns the resulting cluster
        # label for every document in one call.
        labels = model.fit_predict(normalized)

        # inertia_ is K-Means' own objective: sum of squared distances from
        # each point to its assigned centroid. It always decreases as k
        # grows, which is exactly what makes an elbow plot meaningful (we
        # look for where the decrease stops being worth the extra cluster).
        inertia = model.inertia_

        # silhouette_score needs at least 2 clusters and fewer clusters than
        # samples, both of which our k_range already guarantees.
        silhouette = silhouette_score(normalized, labels)

        rows.append({"k": k, "inertia": inertia, "silhouette": silhouette})

    return pd.DataFrame(rows)


def sweep_agglomerative_k(embeddings, k_range=range(2, 21), linkage: str = "average") -> pd.DataFrame:
    """Fit Agglomerative Clustering for every k in k_range and record silhouette.

    No inertia column here: Agglomerative Clustering has no equivalent
    single-objective value the way K-Means' inertia_ does, so silhouette is
    the only label-free signal we get for it.

    Unlike KMeans, scikit-learn's AgglomerativeClustering refuses sparse
    input entirely, and densifying a large TF-IDF matrix would be wasteful.
    Instead we compute the n x n cosine distance matrix once (small, since it
    scales with document count, not vocabulary size) and reuse it for every k
    via metric="precomputed" -- this also means the expensive pairwise-
    distance computation happens once per sweep, not once per k.
    """
    # cosine distance is scale-invariant, so this does not need l2_normalize
    # first; cosine_distance_matrix also accepts sparse input directly, so
    # the original (potentially sparse) embeddings never get densified.
    distance_matrix = cosine_distance_matrix(embeddings)

    rows = []
    for k in k_range:
        # linkage="average" (mean distance between every pair across two
        # clusters) works with any distance metric and is less prone than
        # "ward" to collapsing into one giant cluster plus singletons on
        # noisy, high-dimensional text data. "ward" is excluded by this
        # choice too: it only supports metric="euclidean", not precomputed
        # distances.
        model = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage=linkage)

        labels = model.fit_predict(distance_matrix)

        # metric="precomputed" here tells silhouette_score that
        # distance_matrix already contains distances, so it should not try
        # to recompute them from raw feature vectors.
        silhouette = silhouette_score(distance_matrix, labels, metric="precomputed")

        rows.append({"k": k, "silhouette": silhouette})

    return pd.DataFrame(rows)


def best_k_by_silhouette(sweep_df: pd.DataFrame) -> int:
    """Return the k with the highest silhouette score from a sweep table.

    Works for the output of either sweep_kmeans_k or sweep_agglomerative_k,
    since both always include a "silhouette" column.
    """
    # idxmax() finds the row-label (here, the row's position) of the maximum
    # silhouette value; .loc then reads that row's k back out.
    best_row = sweep_df.loc[sweep_df["silhouette"].idxmax()]

    # k came out of a range() of Python ints, but pandas may have upcast the
    # column to float64 when building the DataFrame; int() converts it back
    # to a plain integer for use as n_clusters elsewhere.
    return int(best_row["k"])


# Running this file directly sweeps k=2..30 (comfortably past both datasets'
# true category counts of 5 and 20, without ever telling the algorithms that
# number) using TF-IDF on both datasets, and -- unlike a plain print, which
# vanishes once the terminal scrolls -- saves every sweep table as a CSV
# under results/, so the numbers stay inspectable after the run ends.
if __name__ == "__main__":
    from pathlib import Path

    from data_loading import load_20newsgroups, load_bbc_news
    from features import build_tfidf

    # results/ lives at the project root, one level up from src/; created if
    # this is the first module run that needs it.
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    K_RANGE = range(2, 31)

    for name, df in [("BBC", load_bbc_news()), ("20NG", load_20newsgroups())]:
        print(f"\n{name}")

        tfidf_matrix, _ = build_tfidf(df["text"])

        kmeans_sweep = sweep_kmeans_k(tfidf_matrix, k_range=K_RANGE)
        kmeans_path = results_dir / f"k_selection_{name.lower()}_tfidf_kmeans.csv"
        kmeans_sweep.to_csv(kmeans_path, index=False)
        print(f"K-Means sweep saved to {kmeans_path}")
        print("best k (K-Means, silhouette):", best_k_by_silhouette(kmeans_sweep))

        agglo_sweep = sweep_agglomerative_k(tfidf_matrix, k_range=K_RANGE)
        agglo_path = results_dir / f"k_selection_{name.lower()}_tfidf_agglomerative.csv"
        agglo_sweep.to_csv(agglo_path, index=False)
        print(f"Agglomerative sweep saved to {agglo_path}")
        print("best k (Agglomerative, silhouette):", best_k_by_silhouette(agglo_sweep))
