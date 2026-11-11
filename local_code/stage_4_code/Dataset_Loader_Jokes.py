'''
Concrete IO class for the encoded joke dataset (next-token prediction)
'''

from local_code.base_class.dataset import dataset
import os
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

N_SPECIAL = 4  # <pad>, <unk>, <bos>, <eos> occupy ids 0..3


class JokeDataset(Dataset):
    '''Holds encoded jokes: [<bos>, w1, ..., wn, <eos>].
    collate -> (x, lengths, y) with x = seq[:-1], y = seq[1:].
    unk_prob > 0 (train only): input tokens that are rare words are replaced by <unk>.'''

    def __init__(self, ids, pad_idx, unk_idx, vocab_size, rare_mask=None, unk_prob=0.0):
        self.ids = ids
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx
        self.vocab_size = vocab_size
        self.rare_mask = rare_mask      # bool tensor [vocab_size]: words seen once in train
        self.unk_prob = unk_prob

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return torch.tensor(self.ids[i], dtype=torch.long)

    def collate(self, batch):
        '''pad to the longest joke in the batch -> (x, lengths, y)'''
        lengths = torch.tensor([len(s) - 1 for s in batch], dtype=torch.long)
        x = pad_sequence([s[:-1] for s in batch], batch_first=True, padding_value=self.pad_idx)
        y = pad_sequence([s[1:] for s in batch], batch_first=True, padding_value=self.pad_idx)

        # natural <unk> targets can't be predicted -> loss ignores them
        y = y.masked_fill(y == self.unk_idx, self.pad_idx)

        # simulate unknown words in the input only; targets keep the real words
        if self.unk_prob > 0 and self.rare_mask is not None:
            swap = self.rare_mask[x] & (torch.rand(x.shape) < self.unk_prob)
            x = x.masked_fill(swap, self.unk_idx)
        return x, lengths, y


class Dataset_Loader_Jokes(dataset):
    dataset_source_folder_path = None
    dataset_source_file_name = 'jokes_encoded.pt'   # point at the MIN_FREQ=1 file
    train_unk_prob = 0.5                            # guess, to tune

    # filled in by load(); generate() needs the vocabulary
    vocab = None
    inv_vocab = None
    vocab_size = None
    pad_idx = unk_idx = bos_idx = eos_idx = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading data...')
        path = os.path.join(self.dataset_source_folder_path, self.dataset_source_file_name)
        data = torch.load(path)

        self.vocab = data['vocab']
        self.inv_vocab = {i: w for w, i in self.vocab.items()}
        self.vocab_size = data['vocab_size']
        self.pad_idx, self.unk_idx = data['pad_idx'], data['unk_idx']
        self.bos_idx, self.eos_idx = data['bos_idx'], data['eos_idx']
        print(f'vocab size: {self.vocab_size}, '
              f'train/val/test: {len(data["train"])}/{len(data["val"])}/{len(data["test"])}')

        # words seen exactly once in the training split (specials excluded)
        counts = torch.zeros(self.vocab_size, dtype=torch.long)
        for seq in data['train']:
            counts += torch.bincount(torch.tensor(seq), minlength=self.vocab_size)
        rare_mask = counts == 1
        rare_mask[:N_SPECIAL] = False
        print(f'rare training words (count 1): {int(rare_mask.sum())}')

        def make(split, **kw):
            return JokeDataset(data[split], self.pad_idx, self.unk_idx, self.vocab_size, **kw)

        return {
            'train': make('train', rare_mask=rare_mask, unk_prob=self.train_unk_prob),
            'val': make('val'),
            'test': make('test'),
        }