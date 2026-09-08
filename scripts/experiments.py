"""Diagnostic experiments that shaped the pipeline's configuration.

Why this file exists separately from run_pipeline.py: these experiments are
not part of the "official" pipeline -- several of them deliberately look at
ground-truth labels in order to measure what was *achievable*, which is not
something the pipeline itself is allowed to do when choosing k. Keeping them
here makes the reasoning behind our configuration reproducible instead of
existing only as numbers in results/ with no code behind them, while keeping
run_pipeline.py honest and label-free in its k selection.

The four experiments, in the order they were originally run:

1. nmi_vs_k_20ng
   Established that 20NewsGroups' problem was not PCA but k selection:
   silhouette was picking k=2 (NMI 0.049) while NMI actually peaked around
   k=27 at 0.265 -- a 5x gap thrown away by the selection rule.

2. tfidf_variants_20ng
   Established that the real bottleneck was the vectorizer, not the
   selection rule. scikit-learn's defaults (min_df=1, raw term frequency)
   left ~81% of the vocabulary as near-unique noise tokens. Adding
   min_df=5 + max_df=0.5 + sublinear_tf raised the achievable NMI from
   0.265 to 0.367.

3. labelfree_on_best_config_20ng
   The honest follow-up: with that cleaner vectorizer, do the LABEL-FREE
   rules now find a sensible k? They do -- silhouette picked k=31 and elbow
   picked k=21 (true: 20), both scoring NMI ~0.354, i.e. 96% of the ceiling.

4. tfidf_variants_bbc
   Checked the same vectorizer change on BBC, where results were already
   good: NMI rose from 0.716 to 0.847, with silhouette still picking
   exactly k=5.

Run all of them with `python scripts/experiments.py`. Each writes its own CSV
under results/.
"""

import sys
from pathlib import Path

# Same import-path handling as run_pipeline.py: src/ modules import each
# other by bare module name.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from clustering import run_kmeans
from data_loading import load_20newsgroups, load_bbc_news
from dimensionality import reduce_dimensions
from evaluation import score_clustering
from k_selection import best_k_by_elbow, best_k_by_silhouette
from utils import l2_normalize

RESULTS_DIR = PROJECT_ROOT / "results"

# The vectorizer settings compared in experiments 2 and 4. "default" is
# scikit-learn's out-of-the-box behaviour, which is what the project used
# before these experiments; the last entry is what features.build_tfidf now
# defaults to.
TFIDF_VARIANTS = {
    "default": dict(min_df=1, max_df=1.0, sublinear_tf=False),
    "min_df5": dict(min_df=5, max_df=1.0, sublinear_tf=False),
    "min_df5_sublinear": dict(min_df=5, max_df=1.0, sublinear_tf=True),
    "min_df5_maxdf05_sublinear": dict(min_df=5, max_df=0.5, sublinear_tf=True),
}


def _sweep_k_with_scores(reduced, true_labels, k_range) -> pd.DataFrame:
    """Cluster at every k in k_range and record both label-free and label-based scores.

    This is the shared engine behind all four experiments: it is the only
    place that computes silhouette (label-free) and NMI/ARI/AMI (label-based)
    side by side, which is exactly what makes the gap between them visible.
    """
    # Silhouette is measured on normalized vectors for the same reason
    # clustering uses them: so Euclidean distance behaves like cosine.
    normalized = l2_normalize(reduced)

    rows = []
    for k in k_range:
        labels, model = run_kmeans(reduced, k=k)
        scores = score_clustering(true_labels, labels)

        rows.append(
            {
                "k": k,
                # inertia is what best_k_by_elbow needs.
                "inertia": model.inertia_,
                "silhouette": silhouette_score(normalized, labels),
                "nmi": scores["nmi"],
                "ari": scores["ari"],
                "ami": scores["ami"],
            }
        )

    return pd.DataFrame(rows)


def nmi_vs_k_20ng(k_range=range(2, 31), pca_sizes=(5, 20, 50, 100)) -> pd.DataFrame:
    """Experiment 1: is 20NG's problem the reduction, or the k selection rule?"""
    df = load_20newsgroups()
    # Deliberately the OLD default vectorizer, because this experiment is
    # what the project looked like before experiment 2 changed it.
    matrix = TfidfVectorizer(**TFIDF_VARIANTS["default"]).fit_transform(df["text"])

    tables = []
    for n_components in pca_sizes:
        reduced, _ = reduce_dimensions(matrix, n_components=n_components)
        table = _sweep_k_with_scores(reduced, df["label"], k_range)
        table["n_components"] = n_components
        tables.append(table)

        best_nmi = table.loc[table["nmi"].idxmax()]
        chosen_k = best_k_by_silhouette(table)
        chosen_nmi = float(table.loc[table["k"] == chosen_k, "nmi"].iloc[0])
        print(
            f"  pca={n_components:>3}: silhouette picks k={chosen_k:>2} (nmi={chosen_nmi:.3f}) "
            f"| best possible k={int(best_nmi['k']):>2} (nmi={best_nmi['nmi']:.3f})"
        )

    combined = pd.concat(tables, ignore_index=True)
    combined.to_csv(RESULTS_DIR / "diagnostic_20ng_nmi_vs_k.csv", index=False)
    return combined


def tfidf_variants(dataset_name: str, loader, pca_sizes, k_range=range(2, 41)) -> pd.DataFrame:
    """Experiments 2 and 4: which vectorizer settings raise the achievable NMI?"""
    df = loader()
    true_labels = df["label"]

    tables = []
    for variant_name, kwargs in TFIDF_VARIANTS.items():
        matrix = TfidfVectorizer(**kwargs).fit_transform(df["text"])

        for n_components in pca_sizes:
            reduced, _ = reduce_dimensions(matrix, n_components=n_components)
            table = _sweep_k_with_scores(reduced, true_labels, k_range)
            table["variant"] = variant_name
            table["vocabulary"] = matrix.shape[1]
            table["n_components"] = n_components
            tables.append(table)

            best = table.loc[table["nmi"].idxmax()]
            print(
                f"  {variant_name:>26} vocab={matrix.shape[1]:>6} pca={n_components:>3}: "
                f"best k={int(best['k']):>2} nmi={best['nmi']:.3f}"
            )

    combined = pd.concat(tables, ignore_index=True)
    combined.to_csv(RESULTS_DIR / f"diagnostic_{dataset_name.lower()}_tfidf_variants.csv", index=False)
    return combined


def labelfree_on_best_config_20ng(k_range=range(2, 41)) -> pd.DataFrame:
    """Experiment 3: with the better vectorizer, can label-free rules find a good k?

    This is the experiment that produced the number the README reports for
    20NewsGroups, because unlike the other three it makes its choice without
    looking at labels -- the ground-truth scores are only read afterwards, to
    report how well that blind choice did.
    """
    df = load_20newsgroups()
    matrix = TfidfVectorizer(**TFIDF_VARIANTS["min_df5_maxdf05_sublinear"]).fit_transform(df["text"])
    reduced, _ = reduce_dimensions(matrix, n_components=100)

    table = _sweep_k_with_scores(reduced, df["label"], k_range)
    table.to_csv(RESULTS_DIR / "labelfree_20ng_best_config.csv", index=False)

    # The two label-free rules, then the ceiling for comparison.
    for label, k in [
        ("silhouette", best_k_by_silhouette(table)),
        ("elbow", best_k_by_elbow(table)),
        ("best possible (uses labels)", int(table.loc[table["nmi"].idxmax(), "k"])),
    ]:
        row = table[table["k"] == k].iloc[0]
        print(f"  {label:>28}: k={k:>2} nmi={row['nmi']:.3f} ari={row['ari']:.3f}")

    return table


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    print("1. 20NG: is the problem PCA or k selection?")
    nmi_vs_k_20ng()

    print("\n2. 20NG: which TF-IDF settings raise the ceiling?")
    tfidf_variants("20NG", load_20newsgroups, pca_sizes=(100, 200, 300))

    print("\n3. 20NG: can label-free rules find a good k on the best config?")
    labelfree_on_best_config_20ng()

    print("\n4. BBC: does the same vectorizer change help there too?")
    tfidf_variants("BBC", load_bbc_news, pca_sizes=(5, 10, 20, 100))


if __name__ == "__main__":
    main()
