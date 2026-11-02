'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset
import numpy as np
import os
import pickle
import torch
from torch.utils.data import Dataset,TensorDataset
from torchvision.transforms import v2

class AugmentedDataset(TensorDataset):
    '''TensorDataset that applies a random transform to the image on every access.
    .tensors holds the original, unaugmented data.'''
    def __init__(self, *tensors, transform):
        super().__init__(*tensors)
        self.transform = transform

    def __getitem__(self, i):
        x, y = super().__getitem__(i)
        return self.transform(x), y

class Dataset_Loader_Pickle(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    # set to a channel index to keep only that channel (0 = R, 1 = G, 2 = B), e.g. for ORL
    # None keeps all channels
    channel_index = None
    # subtracted from every label so they start at 0, e.g. set to 1 for ORL (labels 1..40)
    label_offset = 0

    # filled in by load(), so the model can be built from the data
    num_channels = None
    num_classes = None
    image_size = None  # (H, W)
    # if True, the returned train set applies random transforms on the fly
    augment = False
    augment_transform = None   # None -> default below

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def _default_transform(self):
        return v2.Compose([
            v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            v2.RandomHorizontalFlip(),
        ])

    def _to_images(self, samples):
        '''list of samples -> float tensor (N, C, H, W) scaled to [0, 1]'''
        images = np.stack([np.asarray(s["image"]) for s in samples])
        if images.ndim == 3:                       # (N, H, W) -> (N, 1, H, W)
            images = images[:, None, :, :]
        elif images.ndim == 4:                     # (N, H, W, C) -> (N, C, H, W)
            images = images.transpose(0, 3, 1, 2)
        else:
            raise ValueError(f'Unexpected image array shape: {images.shape}')
        if self.channel_index is not None:
            if not 0 <= self.channel_index < images.shape[1]:
                raise ValueError(f'channel_index {self.channel_index} out of range for {images.shape[1]} channels')
            images = images[:, self.channel_index:self.channel_index + 1]
        images = images.astype(np.float32) / 255.0
        return torch.from_numpy(np.ascontiguousarray(images))

    def _to_labels(self, samples):
        '''shift raw labels by label_offset (e.g. ORL's 1..40 -> 0..39)'''
        labels = torch.tensor([s["label"] for s in samples], dtype=torch.long) - self.label_offset
        if labels.min() < 0:
            raise ValueError(f'Negative label after subtracting label_offset={self.label_offset}')
        return labels

    def load(self):
        print('loading data...')
        path = os.path.join(self.dataset_source_folder_path, self.dataset_source_file_name)
        with open(path, "rb") as f:
            data = pickle.load(f)

        train_x = self._to_images(data["train"])
        test_x = self._to_images(data["test"])
        train_y = self._to_labels(data["train"])
        test_y = self._to_labels(data["test"])

        # per-channel statistics from the training set only
        mean = train_x.mean(dim=(0, 2, 3), keepdim=True)
        std = train_x.std(dim=(0, 2, 3), keepdim=True)
        train_x = (train_x - mean) / std
        test_x = (test_x - mean) / std

        self.num_channels = train_x.shape[1]
        self.image_size = tuple(train_x.shape[2:])
        self.num_classes = int(max(train_y.max(), test_y.max())) + 1
        print(f'channels: {self.num_channels}, image size: {self.image_size}, classes: {self.num_classes}, augmentation: {self.augment}')

        if self.augment:
            transform = self.augment_transform or self._default_transform()
            train_ds = AugmentedDataset(train_x, train_y, transform=transform)
        else:
            train_ds = TensorDataset(train_x, train_y)
        return train_ds, TensorDataset(test_x, test_y)
