"""Load the two raw datasets (BBC News, 20NewsGroups) into a common shape.

Why this module exists: every downstream step (features, clustering, evaluation)
just needs a text column and a held-aside label column, regardless of which
dataset it came from or how that dataset's source data is structured on disk.
Both loaders below return a DataFrame with the same three columns:
text (str), label (int), label_name (str).
"""

# pathlib gives us OS-agnostic paths (this project runs on Windows) and lets
# us build the data/ path relative to this file, so it works no matter where
# the caller's current working directory is.
from pathlib import Path

# pandas.DataFrame is the common return shape every other module will consume.
import pandas as pd

# fetch_20newsgroups downloads/caches the 20NewsGroups dataset via scikit-learn.
from sklearn.datasets import fetch_20newsgroups

# train_test_split is used to reproduce the exact 10% stratified split that
# load_20newsgroups.py specifies ("use the x_test_news"), so our subset matches
# what the assignment snippet already prepared.
from sklearn.model_selection import train_test_split

# This file lives at <project_root>/src/data_loading.py, so its parent's
# parent is the project root; DATA_DIR then points at <project_root>/data.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def load_bbc_news(path: Path = DATA_DIR / "bbc_news_test.csv") -> pd.DataFrame:
    """Load the BBC News CSV and return it as (text, label, label_name).

    path: location of bbc_news_test.csv; defaults to data/bbc_news_test.csv
    under the project root. The file itself is not committed to git (see
    the project .gitignore), so this will raise FileNotFoundError if it's
    missing locally.
    """
    # Fail with a clear message rather than a confusing pandas traceback if
    # the (gitignored, locally-provided) data file hasn't been placed yet.
    if not path.exists():
        raise FileNotFoundError(
            f"BBC News data not found at {path}. "
            "Place bbc_news_test.csv under data/ (it is not tracked in git)."
        )

    # Read the raw CSV as-is; columns are ArticleId, Text, Category.
    raw = pd.read_csv(path)

    # factorize() turns the Category strings into integer codes (label) and
    # returns the sorted unique strings that those codes index into
    # (categories); this gives us both an int label and its readable name
    # without hand-writing a category -> int mapping.
    codes, categories = pd.factorize(raw["Category"], sort=True)

    # Build the common output shape: rename Text -> text, attach the integer
    # label and its readable label_name, and drop ArticleId/Category since
    # downstream code never needs them.
    df = pd.DataFrame(
        {
            "text": raw["Text"],
            "label": codes,
            "label_name": raw["Category"],
        }
    )

    # categories is unused beyond factorize() itself here, since label_name
    # already carries the readable value per row; keeping the variable name
    # documents what codes actually means.
    del categories

    return df


def load_20newsgroups() -> pd.DataFrame:
    """Fetch 20NewsGroups and return its 10% stratified test split as (text, label, label_name).

    Mirrors load_20newsgroups.py exactly (same fetch args, same split args)
    so the subset we cluster on is the one the assignment snippet points at,
    not a different sample we picked ourselves.
    """
    # Download (or use the local sklearn cache of) all 20NewsGroups documents,
    # stripping headers/footers/quotes so the model can't shortcut on
    # metadata like sender addresses instead of actual article content.
    bunch = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))

    # bunch.data is the list of raw document strings; bunch.target is the
    # matching list of integer topic ids; bunch.target_names maps each
    # integer id to its readable topic name (e.g. "sci.space").
    documents = bunch.data
    labels = bunch.target
    label_names = bunch.target_names

    # Assemble a DataFrame with one row per document: its integer label, and
    # its text (named "comment" here to mirror the original snippet before
    # we rename it below).
    news_df = pd.DataFrame({"label": labels, "comment": documents})

    # Attach the readable topic name for each row by looking up its integer
    # label in label_names.
    news_df["label_name"] = news_df["label"].apply(lambda x: label_names[x])

    # Split off a 10% stratified test set, stratified by label so all 20
    # topics stay proportionally represented in the subset. We only keep the
    # test half (x_test_news); the 90% train half is discarded, matching
    # "use the x_test_news" in load_20newsgroups.py. random_state=42 makes
    # this split reproducible across runs.
    _, x_test_news, _, _ = train_test_split(
        news_df,
        news_df["label"],
        test_size=0.1,
        random_state=42,
        stratify=news_df["label"],
    )

    # Rename comment -> text to match the common output shape, and reset the
    # index (it currently holds the original news_df row numbers from before
    # the split) so downstream code can rely on a clean 0..n-1 index.
    x_test_news = x_test_news.rename(columns={"comment": "text"}).reset_index(drop=True)

    return x_test_news


# Running this file directly (rather than importing it) does a quick sanity
# check: load both datasets and print their shape and label distribution, so
# we can eyeball that everything loaded correctly before wiring it into the
# rest of the pipeline.
if __name__ == "__main__":
    bbc = load_bbc_news()
    print("BBC News:", bbc.shape)
    print(bbc["label_name"].value_counts())

    news20 = load_20newsgroups()
    print("\n20NewsGroups:", news20.shape)
    print(news20["label_name"].value_counts())
