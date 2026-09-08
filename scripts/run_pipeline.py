"""Command-line entry point: run the full clustering pipeline end to end.

Why this file exists: it is the one thing you actually execute
(`python scripts/run_pipeline.py`). Everything reusable lives in src/; this
script only orchestrates, so that the whole reported result set can be
regenerated from scratch by one command.

What it does, per dataset (BBC News, 20NewsGroups):
  1. load the documents,
  2. build TF-IDF features,
  3. for each PCA reduction size: reduce, then search for k WITHOUT labels
     (silhouette and elbow, reported separately since they disagree),
  4. cluster with K-Means and Agglomerative at each chosen k,
  5. score the result against ground truth (NMI/ARI/AMI).
Everything lands in results/metrics.csv as one row per experiment.

The label boundary this script maintains: k is always chosen by
k_selection.py, which never sees labels. evaluation.py is called only after
the clustering exists, purely to score it. The one place labels informed a
choice is the vectorizer/PCA configuration explored in scripts/experiments.py
-- disclosed in the README, and sanctioned by the assignment's own wording
("find the optimal reduction to achieve the best clustering results").

Only TF-IDF is used here. features.py also implements Word2Vec and FastText,
but the assignment permits a single embedding method with justification, and
the reasoning for choosing TF-IDF is documented in the README.
"""

import sys
from pathlib import Path

# scripts/ sits next to src/, not inside it, and the src/ modules import each
# other by plain module name (e.g. "from utils import ..."). Putting src/ on
# the import path makes those imports resolve the same way they do when a
# module is run directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

from clustering import run_agglomerative, run_kmeans
from data_loading import load_20newsgroups, load_bbc_news
from dimensionality import reduce_dimensions
from evaluation import build_results_row
from features import build_tfidf
from k_selection import (
    best_k_by_elbow,
    best_k_by_silhouette,
    sweep_agglomerative_k,
    sweep_kmeans_k,
)

# Reduction sizes to try. 300 is deliberately excluded: it was consistently
# worse than 100/200 on both datasets in scripts/experiments.py, so spending
# runtime on it would only pad the table.
PCA_SIZES = (5, 10, 20, 50, 100, 200)

# Wide enough to cover both datasets' true category counts (5 and 20) with
# room on either side, without ever telling the algorithms those numbers.
# The upper bound matters: with a weaker vectorizer the best k pushed against
# a ceiling of 30, so 40 leaves headroom.
K_RANGE = range(2, 41)


def evaluate_dataset(dataset_name: str, documents, true_labels) -> list:
    """Run the whole pipeline for one dataset and return its result rows."""
    # One TF-IDF fit is reused across every reduction size: the vectoriser
    # does not depend on how many components we later keep.
    tfidf_matrix, vectorizer = build_tfidf(documents)
    print(f"{dataset_name}: {tfidf_matrix.shape[0]} docs, vocabulary {tfidf_matrix.shape[1]}")

    rows = []
    for n_components in PCA_SIZES:
        reduced, _ = reduce_dimensions(tfidf_matrix, n_components=n_components)

        # Both sweeps are label-free: they only look at the geometry of the
        # reduced embeddings.
        kmeans_sweep = sweep_kmeans_k(reduced, k_range=K_RANGE)
        agglomerative_sweep = sweep_agglomerative_k(reduced, k_range=K_RANGE)

        # K-Means gets two candidate k values, one per selection rule; the
        # elbow rule needs the inertia column, which only K-Means produces.
        # Agglomerative therefore has silhouette as its only option.
        candidates = [
            ("kmeans", run_kmeans, "silhouette", best_k_by_silhouette(kmeans_sweep), kmeans_sweep),
            ("kmeans", run_kmeans, "elbow", best_k_by_elbow(kmeans_sweep), kmeans_sweep),
            (
                "agglomerative",
                run_agglomerative,
                "silhouette",
                best_k_by_silhouette(agglomerative_sweep),
                agglomerative_sweep,
            ),
        ]

        for algorithm, run_fn, k_method, k, sweep in candidates:
            # Re-fit at the chosen k. (The sweep already fit this exact
            # clustering internally, but it kept only the scores, not the
            # labels -- refitting is cheap and keeps the sweep functions
            # focused on one job.)
            labels, _ = run_fn(reduced, k=k)

            # Carry across the silhouette the sweep measured at this k, so
            # the table shows the label-free signal next to the label-based
            # metrics and the gap between them stays visible.
            silhouette = float(sweep.loc[sweep["k"] == k, "silhouette"].iloc[0])

            rows.append(
                build_results_row(
                    dataset=dataset_name,
                    embedding="tfidf",
                    algorithm=algorithm,
                    n_components=n_components,
                    k=k,
                    true_labels=true_labels,
                    predicted_labels=labels,
                    silhouette=silhouette,
                    k_method=k_method,
                )
            )

        print(f"  pca={n_components:>3} done")

    return rows


def main() -> None:
    """Run every dataset and write the combined comparison table."""
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    all_rows = []
    for dataset_name, loader in [("BBC", load_bbc_news), ("20NG", load_20newsgroups)]:
        df = loader()
        all_rows.extend(evaluate_dataset(dataset_name, df["text"], df["label"]))

    table = pd.DataFrame(all_rows)

    metrics_path = results_dir / "metrics.csv"
    table.to_csv(metrics_path, index=False)

    print(f"\nSaved {len(table)} rows to {metrics_path}")

    # Print the single best configuration per dataset, since that is the
    # headline number the README reports.
    print("\nBest configuration per dataset (by NMI, among label-free k choices):")
    best = table.loc[table.groupby("dataset")["nmi"].idxmax()]
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
