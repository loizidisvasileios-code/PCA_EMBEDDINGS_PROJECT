"""Plot comparison charts and a 2D t-SNE view of the clusters."""


import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

# ticker is imported explicitly rather than relied on as a side effect of
# importing pyplot: it is used for the axis formatter in
# plot_nmi_vs_components, and an implicit import can disappear between
# matplotlib versions.
import matplotlib.ticker
import numpy as np
import pandas as pd

# t-SNE projects the high-dimensional embeddings down to 2D
from sklearn.manifold import TSNE


SURFACE = "#fcfcfb"      # chart background
INK_PRIMARY = "#0b0b0b"  # titles
INK_SECONDARY = "#52514e"  # axis labels
INK_MUTED = "#898781"    # tick labels
GRIDLINE = "#e1e0d9"     # hairline grid
BASELINE = "#c3c2b7"     # axis line
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # blue, orange, aqua -- in fixed order
CONTEXT_GREY = "#d8d7d1"  # de-emphasised points in the small multiples


NEAR_ZERO = 0.02


def _style_axes(ax) -> None:
    
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)

    
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)

    
    ax.tick_params(colors=INK_MUTED, labelsize=9)


def plot_metric_comparison(metrics: pd.DataFrame, output_path) -> None:
    
    # One bar group per (dataset, algorithm): pick that combination's best row.
    best = metrics.loc[metrics.groupby(["dataset", "algorithm"])["nmi"].idxmax()]
    best = best.sort_values(["dataset", "algorithm"], ascending=[False, True])

    labels = [
        f"{r.dataset}\n{r.algorithm}\nPCA {int(r.n_components)}, k={int(r.k)}"
        for r in best.itertuples()
    ]

    metric_names = ["nmi", "ari", "ami"]
    positions = np.arange(len(best))
    # Three bars per group
    bar_width = 0.26

    figure, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    for index, metric in enumerate(metric_names):
        offsets = positions + (index - 1) * bar_width
        values = best[metric].to_numpy()

        bars = ax.bar(
            offsets,
            values,
            bar_width * 0.92,  
            label=metric.upper(),
            color=SERIES[index],
        )

      
        ax.bar_label(
            bars,
            labels=[f"{value:.2f}" if value >= NEAR_ZERO else "" for value in values],
            fontsize=8,
            color=INK_SECONDARY,
            padding=2,
        )

    
    for position, row in zip(positions, best.itertuples()):
        if all(getattr(row, metric) < NEAR_ZERO for metric in metric_names):
            ax.annotate(
                "all ≈ 0\n(single degenerate cluster)",
                xy=(position, 0),
                xytext=(0, 26),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=INK_SECONDARY,
            )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9, color=INK_SECONDARY)
    ax.set_ylabel("score", fontsize=10, color=INK_SECONDARY)
    ax.set_ylim(0, 1)
    ax.set_title(
        "Best clustering result per dataset and algorithm",
        fontsize=12,
        color=INK_PRIMARY,
        pad=14,
    )
    
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, ncols=3)
    _style_axes(ax)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def plot_nmi_vs_components(metrics: pd.DataFrame, output_path) -> None:
    
    # K-Means only: it is the one algorithm both selection rules can drive
    # (elbow needs inertia, which Agglomerative has no equivalent of).
    kmeans_rows = metrics[metrics["algorithm"] == "kmeans"]

    datasets = ["BBC", "20NG"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=SURFACE, sharey=True)

    for ax, dataset in zip(axes, datasets):
        ax.set_facecolor(SURFACE)
        subset = kmeans_rows[kmeans_rows["dataset"] == dataset]

        for index, method in enumerate(["silhouette", "elbow"]):
            method_rows = subset[subset["k_method"] == method].sort_values("n_components")
            ax.plot(
                method_rows["n_components"],
                method_rows["nmi"],
                marker="o",
                markersize=6,
                linewidth=2,
                color=SERIES[index],
                label=method,
            )

        
        ax.set_xscale("log")
        ax.set_xticks(sorted(subset["n_components"].unique()))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("PCA components", fontsize=10, color=INK_SECONDARY)
        ax.set_title(dataset, fontsize=11, color=INK_PRIMARY)
        _style_axes(ax)

    axes[0].set_ylabel("NMI", fontsize=10, color=INK_SECONDARY)
    axes[0].set_ylim(0, 1)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, title="k chosen by")

    figure.suptitle(
        "Label-free k selection: silhouette and elbow fail in opposite places",
        fontsize=12,
        color=INK_PRIMARY,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def plot_k_selection_curves(sweep: pd.DataFrame, dataset_name: str, output_path) -> None:
    
    
    figure, (ax_inertia, ax_silhouette) = plt.subplots(
        2, 1, figsize=(8, 6), facecolor=SURFACE, sharex=True
    )

    for ax in (ax_inertia, ax_silhouette):
        ax.set_facecolor(SURFACE)

  
    ax_inertia.plot(sweep["k"], sweep["inertia"], linewidth=2, color=SERIES[0])
    ax_inertia.set_ylabel("inertia", fontsize=10, color=INK_SECONDARY)
    ax_inertia.set_title(
        f"{dataset_name}: k selection",
        fontsize=12,
        color=INK_PRIMARY,
        pad=12,
    )
    _style_axes(ax_inertia)

    # Bottom panel: silhouette on the same k axis.
    ax_silhouette.plot(sweep["k"], sweep["silhouette"], linewidth=2, color=SERIES[1])
    ax_silhouette.set_ylabel("silhouette", fontsize=10, color=INK_SECONDARY)
    ax_silhouette.set_xlabel("number of clusters (k)", fontsize=10, color=INK_SECONDARY)
    _style_axes(ax_silhouette)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def plot_tsne_small_multiples(
    embeddings, true_label_names, dataset_name: str, output_path, seed: int = 42
) -> None:
    """Project to 2D with t-SNE, then show one small panel per true category.
    """
    
    projection = TSNE(n_components=2, random_state=seed, perplexity=30, init="pca")
    points = projection.fit_transform(np.asarray(embeddings))

    categories = sorted(pd.unique(true_label_names))

    # Lay the panels out in a grid roughly as wide as it is tall.
    n_columns = min(5, len(categories))
    n_rows = int(np.ceil(len(categories) / n_columns))

    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(2.4 * n_columns, 2.4 * n_rows),
        facecolor=SURFACE,
    )
    # np.atleast_1d keeps indexing uniform when there is only one row.
    axes = np.atleast_1d(axes).ravel()

    labels_array = np.asarray(true_label_names)

    for ax, category in zip(axes, categories):
        ax.set_facecolor(SURFACE)

        
        ax.scatter(points[:, 0], points[:, 1], s=3, color=CONTEXT_GREY, linewidths=0)

        
        mask = labels_array == category
        ax.scatter(points[mask, 0], points[mask, 1], s=4, color=SERIES[0], linewidths=0)

        ax.set_title(str(category), fontsize=8, color=INK_SECONDARY)
        
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_color(GRIDLINE)

    
    for unused_ax in axes[len(categories):]:
        unused_ax.set_visible(False)

    figure.suptitle(
        f"{dataset_name} t-sne",
        fontsize=12,
        color=INK_PRIMARY,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, facecolor=SURFACE)
    plt.close(figure)



if __name__ == "__main__":
    from pathlib import Path

    from data_loading import load_20newsgroups, load_bbc_news
    from dimensionality import reduce_dimensions
    from features import build_tfidf
    from k_selection import sweep_kmeans_k

    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / "results"
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(results_dir / "metrics.csv")

    plot_metric_comparison(metrics, figures_dir / "metric_comparison.png")
    print("saved metric_comparison.png")

    plot_nmi_vs_components(metrics, figures_dir / "nmi_vs_components.png")
    print("saved nmi_vs_components.png")

   
    for dataset_name, loader, n_components in [
        ("BBC", load_bbc_news, 5),
        ("20NG", load_20newsgroups, 100),
    ]:
        df = loader()
        tfidf_matrix, _ = build_tfidf(df["text"])
        reduced, _ = reduce_dimensions(tfidf_matrix, n_components=n_components)

        sweep = sweep_kmeans_k(reduced, k_range=range(2, 41))
        plot_k_selection_curves(
            sweep, dataset_name, figures_dir / f"k_selection_{dataset_name.lower()}.png"
        )
        print(f"saved k_selection_{dataset_name.lower()}.png")

        plot_tsne_small_multiples(
            reduced, df["label_name"], dataset_name, figures_dir / f"tsne_{dataset_name.lower()}.png"
        )
        print(f"saved tsne_{dataset_name.lower()}.png")
