"""Plot comparison bar charts and a 2D t-SNE view of the clusters.

Why this module exists: the assignment asks for visual comparison of metrics
across methods/datasets, plus a bonus 2D view of how clusters look in space.
Plots get saved to results/figures/ so the pipeline is reproducible from the
command line, without needing a notebook to view anything interactively.

TODO (brainstorm before implementing):
- plot_metric_comparison(results_df, metric) -> bar chart, saved to results/figures/
- plot_tsne(embeddings, cluster_labels, true_labels) -> side-by-side 2D scatter
"""
