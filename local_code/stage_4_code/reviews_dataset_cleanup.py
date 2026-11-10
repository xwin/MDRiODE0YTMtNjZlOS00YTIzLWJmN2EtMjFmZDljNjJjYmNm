import re
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

DATA_DIR = Path("../../data/stage_4_data/text_classification")       # folder containing train/ and test/
OUT_FILE = DATA_DIR / "reviews_clean.csv"
REMOVE_STOPWORDS = True
USE_STEMMING = False

KEEP = {"no", "nor", "not"}      # negations: never drop these
STOP = set(stopwords.words("english")) - KEEP
stemmer = PorterStemmer()


def clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)           # <br />, other HTML tags
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = ["not" if t == "n't" else t for t in tokens]  # keep negation
    tokens = [t for t in tokens if t.isalpha()]    # drops punctuation, numbers
    if REMOVE_STOPWORDS:
        tokens = [t for t in tokens if t not in STOP]
    if USE_STEMMING:
        tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)


rows = []
for split in ("train", "test"):
    for label_name, label in (("pos", 1), ("neg", 0)):
        for path in (DATA_DIR / split / label_name).glob("*.txt"):
            raw = path.read_text(encoding="utf-8")
            rows.append((split, label, clean(raw)))

df = pd.DataFrame(rows, columns=["split", "label", "text"])
df = df[df["text"].str.len() > 0]                  # drop reviews that became empty
df.to_csv(OUT_FILE, index=False)
print(df.groupby(["split", "label"]).size())