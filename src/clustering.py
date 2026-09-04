"""Fit K-Means, HDBSCAN, and Agglomerative Clustering on a given embedding matrix.

Why this module exists: the assignment requires comparing all three algorithms
on the same embeddings. Each wrapper takes only what that algorithm actually
needs (k for K-Means/Agglomerative, density params for HDBSCAN) and returns
cluster labels in the same (labels, fitted_model) shape so evaluation.py can
treat them identically. HDBSCAN never receives k -- it discovers its own
cluster count from data density, which is exactly why it is our label-free
cross-check against the k chosen for the other two algorithms.
"""

# pandas.DataFrame is the shared output shape for sweep_hdbscan_min_cluster_size,
# matching k_selection.py's sweep tables.
import pandas as pd

# hdbscan.HDBSCAN is a separate package (not part of scikit-learn).
import hdbscan

# AgglomerativeClustering and KMeans do the actual K-Means/Agglomerative fitting.
from sklearn.cluster import AgglomerativeClustering, KMeans

# silhouette_score judges clustering quality without touching ground-truth labels.
from sklearn.metrics import silhouette_score

# cosine_distance_matrix / l2_normalize: the same shared distance handling
# used in k_selection.py, so the final fit here uses identical distances to
# whatever the k-sweep already explored.
from utils import cosine_distance_matrix, l2_normalize


def run_kmeans(embeddings, k: int, seed: int = 42) -> tuple:
    """Fit K-Means with a chosen k and return (labels, fitted_model).

    k is expected to come from k_selection.sweep_kmeans_k /
    best_k_by_silhouette -- this function does not choose k itself, only fits
    it, so the same k-selection logic stays in one place.
    """
    normalized = l2_normalize(embeddings)

    model = KMeans(n_clusters=k, random_state=seed, n_init="auto")

    labels = model.fit_predict(normalized)

    return labels, model


def run_agglomerative(embeddings, k: int, linkage: str = "average") -> tuple:
    """Fit Agglomerative Clustering with a chosen k and return (labels, fitted_model).

    Same precomputed-cosine-distance approach as
    k_selection.sweep_agglomerative_k, for the same reason: scikit-learn's
    AgglomerativeClustering refuses sparse input outright.
    """
    distance_matrix = cosine_distance_matrix(embeddings)

    model = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage=linkage)

    labels = model.fit_predict(distance_matrix)

    return labels, model


def run_hdbscan(embeddings, min_cluster_size: int = 5, min_samples: int | None = None) -> tuple:
    """Fit HDBSCAN and return (labels, fitted_model). Discovers its own k.

    HDBSCAN groups points by density rather than needing a target cluster
    count: min_cluster_size is the smallest group of points it is willing to
    call a cluster at all, and min_samples controls how conservative it is
    about calling a point "noise" (higher = more points get labeled noise).
    Like Agglomerative, it is given the same precomputed cosine distance
    matrix rather than the raw (possibly sparse) embeddings.
    """
    distance_matrix = cosine_distance_matrix(embeddings)

    model = hdbscan.HDBSCAN(
        metric="precomputed",
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )

    labels = model.fit_predict(distance_matrix)

    return labels, model


def count_hdbscan_clusters(labels) -> tuple:
    """Return (n_clusters, n_noise) from a run_hdbscan labels array.

    HDBSCAN marks points it considers noise (not part of any cluster) with
    label -1, so a plain len(set(labels)) would overcount by one whenever
    noise is present. This is the "k HDBSCAN found on its own" figure we
    compare against the k chosen for K-Means/Agglomerative.
    """
    unique_labels = set(labels)

    has_noise = -1 in unique_labels

    n_clusters = len(unique_labels) - (1 if has_noise else 0)

    n_noise = int((labels == -1).sum())

    return n_clusters, n_noise


def sweep_hdbscan_min_cluster_size(embeddings, size_range=range(3, 51)) -> pd.DataFrame:
    """Try a range of min_cluster_size values and record what HDBSCAN finds.

    Mirrors k_selection.py's sweeps, but the parameter being swept is
    min_cluster_size (HDBSCAN's own tuning knob) rather than k directly,
    since HDBSCAN has no k parameter to sweep. Silhouette is computed on
    non-noise points only, and left as NaN when HDBSCAN finds fewer than 2
    real clusters (silhouette is undefined below that, and would otherwise
    crash the sweep).
    """
    distance_matrix = cosine_distance_matrix(embeddings)

    rows = []
    for min_cluster_size in size_range:
        model = hdbscan.HDBSCAN(metric="precomputed", min_cluster_size=min_cluster_size)

        labels = model.fit_predict(distance_matrix)

        n_clusters, n_noise = count_hdbscan_clusters(labels)

        # A silhouette score needs at least 2 real clusters; with 0 or 1,
        # there is nothing meaningful to compare, so we record NaN instead
        # of letting silhouette_score raise.
        if n_clusters >= 2:
            non_noise_mask = labels != -1
            silhouette = silhouette_score(
                distance_matrix[non_noise_mask][:, non_noise_mask],
                labels[non_noise_mask],
                metric="precomputed",
            )
        else:
            silhouette = float("nan")

        rows.append(
            {
                "min_cluster_size": min_cluster_size,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "silhouette": silhouette,
            }
        )

    return pd.DataFrame(rows)


# Running this file directly does a quick sanity check: fit all three
# algorithms on TF-IDF for both datasets using a placeholder k (5 for BBC, 20
# for 20NG -- used ONLY here, as a fixed smoke-test value to confirm the code
# runs end-to-end; this is not a claim about the "right" k, which is
# k_selection.py's job), and sweep HDBSCAN's min_cluster_size to see what
# cluster count it finds on its own.
if __name__ == "__main__":
    from pathlib import Path

    from data_loading import load_20newsgroups, load_bbc_news
    from features import build_tfidf

    # Same results/ folder k_selection.py writes to, so every module's
    # diagnostic output lands in one predictable, inspectable place.
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    for name, df, smoke_test_k in [
        ("BBC", load_bbc_news(), 5),
        ("20NG", load_20newsgroups(), 20),
    ]:
        print(f"\n{name}")

        tfidf_matrix, _ = build_tfidf(df["text"])

        kmeans_labels, _ = run_kmeans(tfidf_matrix, k=smoke_test_k)
        print(f"K-Means (k={smoke_test_k}): {len(set(kmeans_labels))} labels produced")

        agglo_labels, _ = run_agglomerative(tfidf_matrix, k=smoke_test_k)
        print(f"Agglomerative (k={smoke_test_k}): {len(set(agglo_labels))} labels produced")

        hdbscan_labels, _ = run_hdbscan(tfidf_matrix)
        n_clusters, n_noise = count_hdbscan_clusters(hdbscan_labels)
        print(f"HDBSCAN (default params): {n_clusters} clusters found, {n_noise} points marked noise")

        hdbscan_sweep = sweep_hdbscan_min_cluster_size(tfidf_matrix, size_range=range(3, 51))
        sweep_path = results_dir / f"clustering_{name.lower()}_tfidf_hdbscan_sweep.csv"
        hdbscan_sweep.to_csv(sweep_path, index=False)
        print(f"HDBSCAN min_cluster_size sweep (3..50) saved to {sweep_path}")
