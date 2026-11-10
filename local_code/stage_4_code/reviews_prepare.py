import json
from collections import Counter
from pathlib import Path

import pandas as pd
import torch

DATA_DIR = Path("../../data/stage_4_data/text_classification")
CSV_FILE = DATA_DIR / "reviews_clean.csv"
ENCODED_FILE = DATA_DIR / "reviews_encoded.pt"
VOCAB_FILE = DATA_DIR / "vocab.json"

VAL_FRAC = 0.15
MIN_FREQ = 5          # drop words seen fewer times in the training portion
MAX_VOCAB = 20000     # including <pad> and <unk>
MAX_LEN = 300         # adjust using your token-length stats
SEED = 42

PAD, UNK = "<pad>", "<unk>"

# --- Step 1: load and split ---
df = pd.read_csv(CSV_FILE).dropna(subset=["text"])
train_all = df[df["split"] == "train"]
test_df = df[df["split"] == "test"]

val_df = train_all.sample(frac=VAL_FRAC, random_state=SEED)
train_df = train_all.drop(val_df.index)

# --- Step 2: build vocabulary from the training portion only ---
counter = Counter()
for text in train_df["text"]:
    counter.update(text.split())

vocab = {PAD: 0, UNK: 1}
for word, count in counter.most_common(MAX_VOCAB - len(vocab)):
    if count >= MIN_FREQ:
        vocab[word] = len(vocab)

# --- Step 3: encode and save ---
def encode(text):
    return [vocab.get(w, vocab[UNK]) for w in text.split()[:MAX_LEN]]

def encode_df(d):
    return {"ids": [encode(t) for t in d["text"]],
            "labels": d["label"].tolist()}

data = {
    "train": encode_df(train_df),
    "val": encode_df(val_df),
    "test": encode_df(test_df),
    "pad_idx": vocab[PAD],
    "unk_idx": vocab[UNK],
    "vocab_size": len(vocab),
}
torch.save(data, ENCODED_FILE)

with open(VOCAB_FILE, "w", encoding="utf-8") as f:
    json.dump(vocab, f)

print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")
print(f"vocab size={len(vocab)}")