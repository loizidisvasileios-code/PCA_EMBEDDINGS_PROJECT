"""Small helpers shared by more than one pipeline stage.

Why this module exists: L2 normalization and the cosine distance matrix are
both needed by k_selection.py (to sweep k) and clustering.py (to fit the
final models) -- keeping them in one place means every stage treats
"distance" identically instead of drifting apart.
"""

# sklearn's normalize does the actual L2 row-normalization for both sparse
# and dense input (it never densifies a sparse matrix internally), so we do
# not need to hand-roll the division ourselves or branch on input type.
# pairwise_distances computes the cosine distance matrix, also directly on
# sparse input, without ever densifying the original embeddings.
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize


def l2_normalize(embeddings):
    """Scale every document vector (row) to unit length.

    Why: K-Means/Agglomerative/HDBSCAN all reason about Euclidean distance
    internally, but for text we care about direction (topic), not magnitude
    (document length / word frequency). Euclidean distance between two
    unit-length vectors is a monotonic function of their cosine distance, so
    normalizing first makes plain Euclidean clustering behave like cosine
    clustering, without needing cosine-aware algorithm variants.

    TF-IDF vectors from features.build_tfidf are already unit-normalized
    (TfidfVectorizer's default norm="l2"), so this is a no-op for them; it
    matters for Word2Vec/FastText's averaged document vectors, which are not
    normalized by gensim. Applying it uniformly avoids special-casing by
    embedding type.
    """
    # normalize() works directly on scipy sparse matrices without densifying
    # them, and on dense numpy arrays the same way -- one call handles both
    # TF-IDF's sparse matrix and Word2Vec/FastText's dense matrix.
    return normalize(embeddings, norm="l2", axis=1)


def cosine_distance_matrix(embeddings):
    """Compute the n x n pairwise cosine distance matrix for a set of documents.

    Why: AgglomerativeClustering and HDBSCAN both need this precomputed form
    instead of the raw embeddings -- AgglomerativeClustering refuses sparse
    input outright (see k_selection.sweep_agglomerative_k), and HDBSCAN has
    the same practical limitation for TF-IDF-sized sparse input. Using
    metric="precomputed" downstream sidesteps both, and this distance matrix
    is small regardless of vocabulary size: it scales with the number of
    documents (n), not the number of features.
    """
    # pairwise_distances accepts sparse input directly and returns a dense
    # n x n array -- cosine distance is scale-invariant, so this does not
    # need l2_normalize first (unlike the Euclidean-based KMeans path).
    return pairwise_distances(embeddings, metric="cosine")
