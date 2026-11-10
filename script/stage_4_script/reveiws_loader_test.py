from pathlib import Path
import torch
from torch.utils.data import DataLoader

from local_code.stage_4_code.Dataset_Loader_Reviews import Dataset_Loader_Reviews  # adjust import path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / 'data' / 'stage_4_data' / 'text_classification'
FOLDER = DATA_DIR   # folder containing reviews_encoded.pt
BATCH_SIZE = 8

loader = Dataset_Loader_Reviews('reviews', 'IMDB reviews')
loader.dataset_source_folder_path = FOLDER
data = loader.load()

for split in ('train', 'val', 'test'):
    ds = data[split]
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=ds.collate)
    x, lengths, y = next(iter(dl))

    # shapes and dtypes
    assert x.shape == (BATCH_SIZE, lengths.max().item()), x.shape
    assert x.dtype == torch.long and lengths.dtype == torch.long and y.dtype == torch.long
    assert lengths.shape == y.shape == (BATCH_SIZE,)

    # padding: positions beyond each length are pad_idx, positions within are not
    for i in range(BATCH_SIZE):
        n = lengths[i].item()
        assert (x[i, n:] == ds.pad_idx).all(), f'{split}: non-pad after end, row {i}'
        assert (x[i, :n] != ds.pad_idx).all(), f'{split}: pad inside review, row {i}'

    # round trip: unpadded rows equal the original encoded ids (shuffle=False -> first 8 items)
    for i in range(BATCH_SIZE):
        assert x[i, :lengths[i]].tolist() == ds.ids[i], f'{split}: mismatch at row {i}'
        assert y[i].item() == ds.labels[i]

    # ranges
    assert x.min() >= 0 and x.max() < ds.vocab_size
    assert set(y.tolist()) <= {0, 1}

    print(f'{split}: OK  x={tuple(x.shape)}  lengths={lengths.tolist()}  y={y.tolist()}')

# whole-dataset checks: no empty reviews, lengths within MAX_LEN, balanced labels
for split in ('train', 'val', 'test'):
    ds = data[split]
    lens = [len(s) for s in ds.ids]
    print(f'{split}: n={len(ds)}  min_len={min(lens)}  max_len={max(lens)}  '
          f'pos_frac={sum(ds.labels) / len(ds):.3f}')