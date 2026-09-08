"""Load the two raw datasets (BBC News, 20NewsGroups) into a common shape.
"""

from pathlib import Path


import pandas as pd

# fetch_20newsgroups downloads the 20NewsGroups dataset via scikit-learn.
from sklearn.datasets import fetch_20newsgroups


from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def load_bbc_news(path: Path = DATA_DIR / "bbc_news_test.csv") -> pd.DataFrame:
    """Load the BBC News CSV and return it as (text, label, label_name).
    """
    # Fail with a clear message
    if not path.exists():
        raise FileNotFoundError(
            f"BBC News data not found at {path}. "
            "Place bbc_news_test.csv under data/ "
        )

    
    raw = pd.read_csv(path)

    # factorize() turns the Category strings into integer codes (label) and
    # returns the sorted unique strings that those codes index into
    # (categories); this gives us both an int label and its readable name
    
    codes, categories = pd.factorize(raw["Category"], sort=True)

    
    df = pd.DataFrame(
        {
            "text": raw["Text"],
            "label": codes,
            "label_name": raw["Category"],
        }
    )

    
    del categories

    return df


def load_20newsgroups() -> pd.DataFrame:
    """Fetch 20NewsGroups and return its 10% stratified test split as (text, label, label_name).
    """

    
    bunch = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))

    # bunch.data is the list of raw document strings; bunch.target is the
    # matching list of integer topic ids; bunch.target_names maps each
    # integer id to its readable topic name.
    documents = bunch.data
    labels = bunch.target
    label_names = bunch.target_names


    news_df = pd.DataFrame({"label": labels, "comment": documents})

    # Attach the readable topic name for each row by looking up its integer
    # label in label_names.
    news_df["label_name"] = news_df["label"].apply(lambda x: label_names[x])

    # Split off a 10% stratified test set

    _, x_test_news, _, _ = train_test_split(
        news_df,
        news_df["label"],
        test_size=0.1,
        random_state=42,
        stratify=news_df["label"],
    )

    
    x_test_news = x_test_news.rename(columns={"comment": "text"}).reset_index(drop=True)

    return x_test_news


if __name__ == "__main__":
    bbc = load_bbc_news()
    print("BBC News:", bbc.shape)
    print(bbc["label_name"].value_counts())

    news20 = load_20newsgroups()
    print("\n20NewsGroups:", news20.shape)
    print(news20["label_name"].value_counts())
