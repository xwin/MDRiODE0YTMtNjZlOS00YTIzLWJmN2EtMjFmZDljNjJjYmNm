from pathlib import Path
import random
import torch

from local_code.stage_4_code.Dataset_Loader_Reviews import Dataset_Loader_Reviews  # adjust paths
from local_code.stage_4_code.Method_RNN_Reviews import Method_RNN_Reviews

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / 'data' / 'stage_4_data' / 'text_classification'
FOLDER = DATA_DIR           # folder containing reviews_encoded.pt
SMOKE = False                # True: small random subsets and 2 epochs, False: full run

torch.manual_seed(0)
random.seed(0)

loader = Dataset_Loader_Reviews('reviews', 'IMDB reviews')
loader.dataset_source_folder_path = FOLDER
data = loader.load()


def shrink(ds, n):
    '''keep n random items; a random sample is needed because the data is ordered pos-first'''
    idx = random.sample(range(len(ds)), min(n, len(ds)))
    ds.ids = [ds.ids[i] for i in idx]
    ds.labels = [ds.labels[i] for i in idx]


if SMOKE:
    shrink(data['train'], 2000)
    shrink(data['val'], 500)
    shrink(data['test'], 500)

model = Method_RNN_Reviews('gru classifier', 'bi-GRU sentiment')
model.max_epoch = 2 if SMOKE else 10
model.batch_size = 64
model.data = data

result = model.run()

pred, true = result['pred_y'], result['true_y']
print('pred shape:', tuple(pred.shape), 'true shape:', tuple(true.shape))
print('pred values:', pred.unique().tolist(), 'true values:', true.unique().tolist())
print('test accuracy:', (pred == true).float().mean().item())