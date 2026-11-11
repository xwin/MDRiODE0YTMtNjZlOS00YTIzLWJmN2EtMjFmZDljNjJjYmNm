import re
from collections import Counter
from pathlib import Path

import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

DATA_FILE = Path("../../data/stage_4_data/text_generation/data")

def strip_markup(t):
    t = re.sub(r"\[([^\]]*)\]\(https?://[^)\s]*\)", r"\1", t)   # [text](url) -> text
    t = re.sub(r"https?://\S+", " ", t)                          # bare URLs
    return re.sub(r"\s+", " ", t.replace("*", "")).strip()

df = pd.read_csv(DATA_FILE, encoding="utf-8")
print("shape:", df.shape, "| columns:", list(df.columns))
jokes = df["Joke"]
print("null:", jokes.isna().sum(), "| empty after strip:", (jokes.fillna("").str.strip() == "").sum())
print("duplicate jokes:", jokes.duplicated().sum())
print("with newline/tab:", jokes.str.contains(r"[\n\r\t]").sum())
print("leading/trailing whitespace:", (jokes != jokes.str.strip()).sum())
print("double spaces:", jokes.str.contains(r"\s{2,}").sum())
print("contains HTML-like tags:", jokes.str.contains(r"<[^>]+>").sum())

# Non-ASCII characters (curly quotes, odd symbols)
non_ascii = Counter(ch for j in jokes.dropna() for ch in j if ord(ch) > 127)
print("non-ASCII chars:", non_ascii.most_common(20))

# Tokenize (lowercase, keep punctuation)
jokes = jokes.dropna().str.strip()
tokens = [word_tokenize(j.lower()) for j in jokes]
lens = pd.Series([len(t) for t in tokens])
print("\ntoken length stats:")
print(lens.describe())
print(lens.quantile([0.9, 0.95, 0.99]))
print("jokes with < 3 tokens:", (lens < 3).sum())

counter = Counter(t for ts in tokens for t in ts)
print("\nunique tokens:", len(counter))
print("seen once:", sum(1 for c in counter.values() if c == 1))
print("seen >= 2:", sum(1 for c in counter.values() if c >= 2))
print("top 25:", counter.most_common(25))

# Detokenizer round trip
detok = TreebankWordDetokenizer()
norm = lambda s: re.sub(r"\s+", " ", s.lower()).strip()
bad = [(j, detok.detokenize(t)) for j, t in zip(jokes, tokens)
       if norm(j) != detok.detokenize(t)]
print(f"\nround-trip mismatches: {len(bad)} / {len(jokes)}")
for orig, back in bad[:10]:
    print("ORIG:", norm(orig))
    print("BACK:", back, "\n")

# Weird tokens worth eyeballing (quote tokens, contraction pieces)
print("quote tokens:", {k: v for k, v in counter.items() if k in ("``", "''", '"')})
print("tokens starting with apostrophe:", [(k, v) for k, v in counter.most_common() if k.startswith("'")][:15])

df["stripped"] = df["Joke"].map(strip_markup)
df["n_words"] = df["stripped"].str.findall(r"[A-Za-z0-9]+").str.len()

has_url = df["Joke"].str.contains(r"https?://")
print("with URL:", has_url.sum())
for _, r in df[has_url].iterrows():
    print(r["ID"], "|", r["n_words"], "words after stripping |", r["Joke"][:160])

print("\nshortest 15 after stripping:")
for _, r in df.nsmallest(15, "n_words").iterrows():
    print(r["ID"], "|", r["n_words"], "|", r["Joke"][:160])

print("\nlongest 5 by word count:")
for _, r in df.nlargest(5, "n_words").iterrows():
    print(r["ID"], "|", r["n_words"], "|", r["Joke"][:200])