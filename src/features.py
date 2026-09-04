"""Turn raw text into numeric embeddings (TF-IDF, Word2Vec, FastText).

Why this module exists: clustering algorithms need vectors, not strings. The
choice of embedding changes what "distance between two documents" even means,
so we keep every method isolated behind its own build_* function, all callable
with different hyperparameters, so run_pipeline.py can sweep and compare them
fairly instead of us guessing which one "should" work best.

No preprocessing is applied anywhere in this module (per the assignment's own
note): TF-IDF tokenizes with scikit-learn's default pattern, Word2Vec/FastText
are tokenized by a plain whitespace split. Punctuation stays glued to words
("good." is a different token from "good") -- that is a deliberate choice, not
an oversight.
"""

# numpy backs the dense document-vector matrices Word2Vec/FastText produce.
import numpy as np

# pandas.Series is the input type for every build_* function (one string per
# document), matching the "text" column from data_loading.py.
import pandas as pd

# TfidfVectorizer builds the sparse TF-IDF doc-term matrix.
from sklearn.feature_extraction.text import TfidfVectorizer

# gensim's Word2Vec and FastText train word embeddings directly on our own
# corpus (no pretrained/external vectors -- see our conversation: staying
# self-contained and reproducible from just this project's data).
from gensim.models import FastText, Word2Vec


def build_tfidf(
    texts: pd.Series,
    max_features: int | None = None,
    ngram_range: tuple[int, int] = (1, 1),
) -> tuple:
    """Fit a TF-IDF vectorizer on texts and return (doc-term matrix, vectorizer).

    max_features / ngram_range are exposed (rather than hardcoded) so
    run_pipeline.py can sweep them if the default unigram, unlimited-vocabulary
    TF-IDF underperforms. The fitted vectorizer is returned alongside the
    matrix so callers can inspect vocabulary/feature names later if useful
    for interpretation.
    """
    # TfidfVectorizer handles tokenization, vocabulary building, term-frequency
    # counting, and inverse-document-frequency weighting all in one fit_transform.
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)

    # fit_transform learns the vocabulary from texts and immediately encodes
    # every document as a sparse row in the returned matrix.
    matrix = vectorizer.fit_transform(texts)

    return matrix, vectorizer


def _tokenize(texts: pd.Series) -> list[list[str]]:
    """Split each document into tokens on whitespace only -- no lowercasing,
    no punctuation stripping, no stopword removal. This feeds Word2Vec/FastText
    training and keeps us honest to "no preprocessing required".
    """
    # str(t) guards against any non-string values slipping through; .split()
    # with no argument splits on any run of whitespace and drops empty strings.
    return [str(t).split() for t in texts]


def _average_document_vectors(
    tokenized_docs: list[list[str]], model, vector_size: int
) -> np.ndarray:
    """Represent each document as the mean of its tokens' word vectors.

    A document whose tokens are all unknown to the model (or that has zero
    tokens) gets a zero-vector -- an intentional, agreed-on edge case rather
    than a crash or a silently dropped row, so every document still gets a
    row in the output matrix that clustering.py can consume.
    """
    # Pre-allocate a dense (n_docs, vector_size) matrix, defaulting every row
    # to zeros; rows for empty/all-unknown-word documents are left untouched.
    doc_vectors = np.zeros((len(tokenized_docs), vector_size), dtype=np.float32)

    # model.wv is gensim's KeyedVectors: `token in model.wv` checks whether a
    # vector exists for that token. For FastText this is true even for
    # out-of-vocabulary words (it builds a vector from character n-grams),
    # which is exactly FastText's practical advantage on rare/misspelled
    # tokens; for Word2Vec it is only true for tokens seen >= min_count times.
    for row_index, tokens in enumerate(tokenized_docs):
        known_vectors = [model.wv[token] for token in tokens if token in model.wv]

        # Only overwrite the zero-initialized row if we found at least one
        # known token; otherwise this document keeps its zero-vector.
        if known_vectors:
            doc_vectors[row_index] = np.mean(known_vectors, axis=0)

    return doc_vectors


def build_word2vec(
    texts: pd.Series,
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 2,
    epochs: int = 10,
    seed: int = 42,
) -> tuple:
    """Train Word2Vec from scratch on texts, then average each doc's word
    vectors into one document vector. Returns (doc-vector matrix, fitted model).

    workers=1 alongside a fixed seed makes training fully deterministic
    (gensim's multi-threaded training is not reproducible run-to-run, since
    thread scheduling affects the order vectors get updated in).
    """
    tokenized = _tokenize(texts)

    # Word2Vec trains directly on our tokenized corpus; vector_size/window/
    # min_count/epochs are all exposed for later sweeping if results are weak.
    model = Word2Vec(
        sentences=tokenized,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        epochs=epochs,
        seed=seed,
        workers=1,
    )

    doc_vectors = _average_document_vectors(tokenized, model, vector_size)

    return doc_vectors, model


def build_fasttext(
    texts: pd.Series,
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 2,
    epochs: int = 10,
    seed: int = 42,
) -> tuple:
    """Same idea as build_word2vec, but with gensim's FastText, which also
    learns character n-gram (subword) vectors. That lets it produce a
    reasonable vector even for words it never saw in training -- relevant for
    20NewsGroups' informal, typo- and jargon-heavy short posts.
    """
    tokenized = _tokenize(texts)

    model = FastText(
        sentences=tokenized,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        epochs=epochs,
        seed=seed,
        workers=1,
    )

    doc_vectors = _average_document_vectors(tokenized, model, vector_size)

    return doc_vectors, model


# Running this file directly does a quick sanity check: build all three
# embeddings on both datasets and print their resulting shapes, so we can
# confirm everything trains end-to-end before wiring it into clustering.
if __name__ == "__main__":
    from data_loading import load_bbc_news, load_20newsgroups

    for name, df in [("BBC", load_bbc_news()), ("20NG", load_20newsgroups())]:
        print(f"\n{name}: {len(df)} docs")

        tfidf_matrix, tfidf_vectorizer = build_tfidf(df["text"])
        print("  TF-IDF matrix:", tfidf_matrix.shape, "vocab size:", len(tfidf_vectorizer.vocabulary_))

        w2v_vectors, w2v_model = build_word2vec(df["text"])
        print("  Word2Vec matrix:", w2v_vectors.shape, "vocab size:", len(w2v_model.wv))

        ft_vectors, ft_model = build_fasttext(df["text"])
        print("  FastText matrix:", ft_vectors.shape, "vocab size:", len(ft_model.wv))
