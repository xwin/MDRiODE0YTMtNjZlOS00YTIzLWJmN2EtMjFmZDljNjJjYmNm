import random
from collections import Counter
from pathlib import Path

import pandas as pd

DATA_DIR = Path("../../data/stage_4_data/text_generation")
IN_FILE = DATA_DIR / "jokes_clean.csv"

df = pd.read_csv(IN_FILE)
print(df.shape)
tokens = df["text"].str.split()
lens = tokens.str.len()
print(lens.describe())
print(lens.quantile([0.9, 0.95, 0.99]))

counter = Counter(t for ts in tokens for t in ts)
print("\nunique tokens:", len(counter))
for k in (1, 2, 3, 5):
    print(f"seen >= {k}:", sum(1 for c in counter.values() if c >= k))
print("top 30:", counter.most_common(30))
print("non-word tokens:", [(t, c) for t, c in counter.most_common() if not t[0].isalnum()])
rare = [t for t, c in counter.items() if c == 1]
print("random sample of tokens seen once:", random.Random(0).sample(rare, 40))

starts = Counter(" ".join(ts[:3]) for ts in tokens)
print("\ndistinct 3-word starts:", len(starts))
print("top 10 starts:", starts.most_common(10))
bad = [" ".join(ts[:6]) for ts in tokens if not all(t[0].isalnum() for t in ts[:3])]
print("jokes whose first 3 tokens are not all words:", len(bad), bad[:5])