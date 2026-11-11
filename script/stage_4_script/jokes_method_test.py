import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root; adjust if needed
from local_code.stage_4_code.Dataset_Loader_Jokes import Dataset_Loader_Jokes, JokeDataset
from local_code.stage_4_code.Method_RNN_Jokes import Method_RNN_Jokes

loader = Dataset_Loader_Jokes("jokes", "smoke")
loader.dataset_source_folder_path = "../../data/stage_4_data/text_generation"
sets = loader.load()


def subset(ds, n, unk_prob=None):
    idx = random.Random(0).sample(range(len(ds)), n)
    return JokeDataset([ds.ids[i] for i in idx], ds.pad_idx, ds.unk_idx, ds.vocab_size,
                       rare_mask=ds.rare_mask,
                       unk_prob=ds.unk_prob if unk_prob is None else unk_prob)


def run(data, **hp):
    m = Method_RNN_Jokes("rnn", "jokes")
    for k, v in hp.items():
        setattr(m, k, v)
    m.data = data
    return m.run()


print("\n=== smoke run: small subsets, 2 epochs ===")
print(run({"train": subset(sets["train"], 200), "val": subset(sets["val"], 40),
           "test": subset(sets["test"], 40)}, max_epoch=2))

print("\n=== overfit check: 32 jokes, no dropout, no <unk> simulation ===")
tiny = subset(sets["train"], 32, unk_prob=0.0)
print(run({"train": tiny, "val": tiny, "test": tiny},
          max_epoch=300, dropout=0.0, batch_size=8))