import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root; adjust if needed
from local_code.stage_4_code.Dataset_Loader_Jokes import Dataset_Loader_Jokes

loader = Dataset_Loader_Jokes("jokes", "floor")
loader.dataset_source_folder_path = "../../data/stage_4_data/text_generation"
train = loader.load()["train"]

idx = random.Random(0).sample(range(len(train)), 32)   # same 32 jokes as the overfit check
seqs = [train.ids[i] for i in idx]

nxt = defaultdict(Counter)                              # prefix -> counts of the next token
for s in seqs:
    for t in range(len(s) - 1):
        nxt[tuple(s[:t + 1])][s[t + 1]] += 1

n = sum(sum(c.values()) for c in nxt.values())
floor = -sum(k * math.log(k / sum(c.values())) for c in nxt.values() for k in c.values()) / n
print("targets:", n, "| loss floor:", round(floor, 4), "| perplexity floor:", round(math.exp(floor), 4))