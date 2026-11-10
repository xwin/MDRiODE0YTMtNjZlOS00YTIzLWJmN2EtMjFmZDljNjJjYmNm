'''
Concrete IO class for the encoded movie review dataset
'''

from local_code.base_class.dataset import dataset
import os
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class ReviewDataset(Dataset):
    '''Holds encoded reviews (lists of token ids) and labels.
    Use DataLoader(ds, batch_size=..., collate_fn=ds.collate).'''

    def __init__(self, ids, labels, pad_idx, vocab_size):
        self.ids = ids
        self.labels = labels
        self.pad_idx = pad_idx
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return torch.tensor(self.ids[i], dtype=torch.long), self.labels[i]

    def collate(self, batch):
        '''pad to the longest review in the batch -> (x, lengths, y)'''
        seqs, labels = zip(*batch)
        lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
        x = pad_sequence(seqs, batch_first=True, padding_value=self.pad_idx)
        y = torch.tensor(labels, dtype=torch.long)
        return x, lengths, y


class Dataset_Loader_Reviews(dataset):
    dataset_source_folder_path = None
    dataset_source_file_name = 'reviews_encoded.pt'

    # filled in by load(), so the model can be built from the data
    vocab_size = None
    pad_idx = None
    num_classes = 2

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading data...')
        path = os.path.join(self.dataset_source_folder_path, self.dataset_source_file_name)
        data = torch.load(path)

        self.vocab_size = data['vocab_size']
        self.pad_idx = data['pad_idx']
        print(f'vocab size: {self.vocab_size}, '
              f'train/val/test: {len(data["train"]["ids"])}/'
              f'{len(data["val"]["ids"])}/{len(data["test"]["ids"])}')

        return {
            split: ReviewDataset(data[split]['ids'], data[split]['labels'],
                                 self.pad_idx, self.vocab_size)
            for split in ('train', 'val', 'test')
        }