import json
import random
from collections import Counter
from pathlib import Path

import pandas as pd
import torch

DATA_DIR = Path("../../data/stage_4_data/text_generation")
CSV_FILE = DATA_DIR / "jokes_clean.csv"
ENCODED_FILE = DATA_DIR / "jokes_encoded.pt"
VOCAB_FILE = DATA_DIR / "jokes_vocab.json"

VAL_FRAC = 0.10
TEST_FRAC = 0.10
MIN_FREQ = 1          # drop tokens seen fewer times in the training split
MAX_LEN = 60          # safety cap, above the longest joke (53 tokens + <bos>/<eos> = 55)
SEED = 42

PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"

# --- Step 1: load, shuffle, split 80/10/10 ---
df = pd.read_csv(CSV_FILE).dropna(subset=["text"])
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
n_test = int(len(df) * TEST_FRAC)
n_val = int(len(df) * VAL_FRAC)
test_df = df.iloc[:n_test]
val_df = df.iloc[n_test:n_test + n_val]
train_df = df.iloc[n_test + n_val:]

# --- Step 2: vocabulary from the training split only ---
counter = Counter()
for text in train_df["text"]:
    counter.update(text.split())

vocab = {PAD: 0, UNK: 1, BOS: 2, EOS: 3}
for word, count in counter.most_common():
    if count >= MIN_FREQ:
        vocab[word] = len(vocab)

# --- Step 3: encode as [<bos>] + tokens + [<eos>] ---
def encode(text):
    ids = [vocab.get(w, vocab[UNK]) for w in text.split()]
    return ([vocab[BOS]] + ids + [vocab[EOS]])[:MAX_LEN]

def encode_df(d):
    return [encode(t) for t in d["text"]]

data = {
    "train": encode_df(train_df),
    "val": encode_df(val_df),
    "test": encode_df(test_df),
    "vocab": vocab,
    "pad_idx": vocab[PAD],
    "unk_idx": vocab[UNK],
    "bos_idx": vocab[BOS],
    "eos_idx": vocab[EOS],
    "vocab_size": len(vocab),
}
torch.save(data, ENCODED_FILE)
with open(VOCAB_FILE, "w", encoding="utf-8") as f:
    json.dump(vocab, f)

# --- Checks ---
print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")
print(f"vocab size={len(vocab)} (MIN_FREQ={MIN_FREQ})")
for name in ("train", "val", "test"):
    ids = [i for seq in data[name] for i in seq]
    unk = sum(1 for i in ids if i == vocab[UNK])
    print(f"{name}: {len(ids)} ids, <unk> rate = {unk / len(ids):.3%}")
print("longest encoded sequence:", max(len(s) for s in data["train"] + data["val"] + data["test"]))
print("truncated sequences:", sum(1 for s in data["train"] + data["val"] + data["test"]
                                  if len(s) == MAX_LEN and s[-1] != vocab[EOS]))

full = Counter(w for t in df["text"] for w in t.split())
print("entity debris counts (amp, gt, lt, quot):", [full[w] for w in ("amp", "gt", "lt", "quot")])

# Round trip: ids -> words must reproduce the text (apart from <unk>)
inv = {i: w for w, i in vocab.items()}
bad = 0
for text, ids in zip(train_df["text"], data["train"]):
    expected = [w if w in vocab else UNK for w in text.split()]
    if [inv[i] for i in ids[1:-1]] != expected:
        bad += 1
print("round-trip mismatches (train):", bad, "/", len(train_df))
# random.Random(0).shuffle(data["test"])