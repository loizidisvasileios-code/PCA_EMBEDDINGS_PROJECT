"""Score cluster assignments against true labels, and collect results across runs.
"""


import pandas as pd


import numpy as np


from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)


def score_clustering(true_labels, predicted_labels) -> dict:
    """Compare a clustering against ground truth. Returns {"nmi", "ari", "ami"}.

    HDBSCAN's noise points (label -1) are deliberately left in and treated as
    one more group rather than being dropped.
    """
    return {
       
        "nmi": normalized_mutual_info_score(true_labels, predicted_labels),
        "ari": adjusted_rand_score(true_labels, predicted_labels),
        "ami": adjusted_mutual_info_score(true_labels, predicted_labels),
    }


def noise_fraction(predicted_labels) -> float:
    """Fraction of documents an algorithm refused to assign to any cluster.

    Only HDBSCAN produces these (label -1); K-Means and Agglomerative always
    assign every point, so this returns 0.0 for them. Reported alongside the
    scores because a high value changes how those scores should be read.
    """
    # np.asarray lets this accept plain lists as well as numpy arrays.
    labels = np.asarray(predicted_labels)

    # -1 is HDBSCAN's noise marker; every other label is a real cluster.
    return float((labels == -1).sum() / len(labels))


def build_results_row(
    dataset: str,
    embedding: str,
    algorithm: str,
    n_components,
    k,
    true_labels,
    predicted_labels,
    silhouette=None,
    k_method: str | None = None,
) -> dict:
    
    row = {
        "dataset": dataset,
        "embedding": embedding,
        "algorithm": algorithm,
        # None here reads as "no PCA applied", which pandas renders as an
        # empty cell -- distinguishable from an actual component count.
        "n_components": n_components,
        "k_method": k_method,
        "k": k,
    }

    # Merge the three metric values in as their own columns.
    row.update(score_clustering(true_labels, predicted_labels))

    row["silhouette"] = silhouette
    row["noise_fraction"] = noise_fraction(predicted_labels)

    return row



if __name__ == "__main__":
    from pathlib import Path

    # Part 1: check the metrics behave as expected on cases where the right
    # answer is known in advance. This is what makes the self-test a test
    # rather than a demo -- if these three ever stop holding, something in
    # the scoring is wrong and every number in metrics.csv is suspect.
    truth = [0, 0, 1, 1, 2, 2]

    # A clustering identical to the truth up to renaming: all three metrics
    # must be 1.0, since they measure structure, not label values.
    renamed = [5, 5, 3, 3, 9, 9]
    print("perfect clustering (renamed labels):", score_clustering(truth, renamed))

    # Everything in one cluster shares no information with the truth, so the
    # chance-corrected metrics (ARI/AMI) must sit at ~0.
    single = [0, 0, 0, 0, 0, 0]
    print("single-cluster clustering:          ", score_clustering(truth, single))

    # noise_fraction counts only HDBSCAN's -1 marker: 2 of 6 here, and 0.0
    # for any algorithm that assigns every point.
    print("noise fraction of [-1,-1,0,0,1,1]:  ", noise_fraction([-1, -1, 0, 0, 1, 1]))

    # Part 2: summarise the comparison table this module helped build, so
    # running the file also answers "what did we actually achieve?".
    metrics_path = Path(__file__).resolve().parent.parent / "results" / "metrics.csv"
    if not metrics_path.exists():
        print(f"\n{metrics_path.name} not found -- run scripts/run_pipeline.py first")
    else:
        metrics = pd.read_csv(metrics_path)

        print(f"\n{len(metrics)} experiments in {metrics_path.name}")
        print("\nBest configuration per dataset (by NMI, k chosen without labels):")

        best = metrics.loc[metrics.groupby("dataset")["nmi"].idxmax()]
        print(best.to_string(index=False))
