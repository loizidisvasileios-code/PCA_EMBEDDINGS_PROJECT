"""Small helpers shared by more than one pipeline stage.

Why this module exists: L2 normalization is needed by both k_selection.py
(to sweep k) and clustering.py (to fit the final models) -- keeping it in one
place means both stages treat "distance" identically instead of drifting apart.
"""

# sklearn's normalize does the actual L2 row-normalization for both sparse
# and dense input (it never densifies a sparse matrix internally), so we do
# not need to hand-roll the division ourselves or branch on input type.
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
