import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root; adjust if your imports differ
from local_code.stage_4_code.Dataset_Loader_Jokes import Dataset_Loader_Jokes

loader = Dataset_Loader_Jokes("jokes", "test")
loader.dataset_source_folder_path = "../../data/stage_4_data/text_generation"
sets = loader.load()
PAD, UNK, BOS, EOS = loader.pad_idx, loader.unk_idx, loader.bos_idx, loader.eos_idx
V = loader.vocab_size

for name, ds in sets.items():
    batch = [ds[i] for i in range(len(ds))]
    x, lengths, y = ds.collate(batch)
    print(f"\n[{name}] x{tuple(x.shape)} y{tuple(y.shape)} lengths{tuple(lengths.shape)}")
    assert x.shape == y.shape and x.dtype == y.dtype == lengths.dtype == torch.long
    assert x.max() < V and y.max() < V and x.min() >= 0 and y.min() >= 0
    assert (x[:, 0] == BOS).all(), "every input must start with <bos>"
    assert (lengths == (x != PAD).sum(1)).all(), "lengths must match non-pad inputs"
    # every non-pad target is an <eos> exactly at the last position of each joke
    last = y[torch.arange(len(y)), lengths - 1]
    assert (last == EOS).all(), "last target must be <eos>"
    # shift check on the raw sequences (val/test have no random swaps)
    if name != "train":
        for i in range(0, len(ds), 17):
            n = int(lengths[i])
            raw = ds[i]
            assert torch.equal(x[i, :n], raw[:-1])
            kept = raw[1:] != UNK
            assert torch.equal(y[i, :n][kept], raw[1:][kept])
            assert (y[i, :n][~kept] == PAD).all()
        print("val/test unk targets masked:", int((y == PAD).sum() - (x == PAD).sum()))
    # padding positions in y are PAD
    assert (y[x == PAD] == PAD).all()

# train: simulated <unk> only in x, not in y, and the rate is sensible
tr = sets["train"]
batch = [tr[i] for i in range(len(tr))]
x, lengths, y = tr.collate(batch)
raw_x = torch.nn.utils.rnn.pad_sequence([s[:-1] for s in batch], batch_first=True, padding_value=PAD)
changed = (x != raw_x)
print("\ntrain tokens swapped to <unk>:", int(changed.sum()), "of", int((raw_x != PAD).sum()))
assert (x[changed] == UNK).all()
assert not (y == UNK).any()
assert (raw_x[changed] >= 4).all(), "specials must never be swapped"
x2, _, _ = tr.collate(batch)
print("two draws differ (should be True):", not torch.equal(x, x2))
print("\nall checks passed")