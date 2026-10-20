'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset
import numpy as np
import os

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    mean = None
    std = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):
        print('loading data...')
        data = np.loadtxt(os.path.join(self.dataset_source_folder_path, self.dataset_source_file_name), dtype=np.float32, delimiter=',')
        X = data[:, 1:]
        y = data[:, 0].astype(np.int32)
        print(f'before normalization: min of X: {np.min(X)}, max of X: {np.max(X)}')
        X = X / 255.0
        if self.mean is None or self.std is None:
            self.mean = np.mean(X)
            self.std = np.std(X)
        print(f'applying mean: {self.mean} and std: {self.std}')
        X = (X - self.mean) / self.std
        print(f'after normalization: min of X: {np.min(X)}, max of X: {np.max(X)}')
        return {'X': X, 'y': y}