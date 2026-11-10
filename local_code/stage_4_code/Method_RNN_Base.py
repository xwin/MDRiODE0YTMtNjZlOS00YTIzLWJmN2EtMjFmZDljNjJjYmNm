'''
Base class for RNN methods: shared hyperparameters and the train / validate / test loop.
Subclasses define _build_layers(), forward(), _compute_loss(), _evaluate() and _test().
'''

import copy
import torch
from torch import nn
from torch.utils.data import DataLoader

from local_code.base_class.method import method


class Method_RNN_Base(method, nn.Module):
    # expected format: {'train': Dataset, 'val': Dataset, 'test': Dataset}
    # each dataset provides .collate for DataLoader
    data = None
    history = None
    max_epoch = 10
    batch_size = 64
    learning_rate = 1e-3
    optimizer_name = 'Adam'      # 'Adam', 'SGD', 'RMSprop'
    loss_name = 'cross_entropy'  # 'cross_entropy', 'label_smoothing'
    clip_norm = 1.0              # None disables gradient clipping
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

    # ---- hooks for subclasses ----
    def _build_layers(self):
        '''create layers from self.data (vocab size etc.)'''
        raise NotImplementedError

    def _compute_loss(self, batch):
        '''forward pass + loss for one training batch -> (loss, n_items)'''
        raise NotImplementedError

    def _evaluate(self, dataset):
        '''no-grad pass over a dataset -> dict with at least 'Loss' (lower is better)'''
        raise NotImplementedError

    def _test(self, dataset):
        '''final pass on the test set -> whatever run() should return'''
        raise NotImplementedError

    # ---- shared machinery ----
    def _build_optimizer(self):
        name = self.optimizer_name.lower()
        if name == 'adam':
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif name == 'sgd':
            return torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        elif name == 'rmsprop':
            return torch.optim.RMSprop(self.parameters(), lr=self.learning_rate)
        raise ValueError('Optimizer not found')

    def _build_loss(self):
        name = self.loss_name.lower()
        if name == 'cross_entropy':
            return nn.CrossEntropyLoss()
        elif name == 'label_smoothing':
            return nn.CrossEntropyLoss(label_smoothing=0.1)
        raise ValueError('Loss not found')

    def _loader(self, dataset, shuffle):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle,
                          collate_fn=dataset.collate)

    def _to_device(self, batch):
        return tuple(t.to(self.device) for t in batch)

    def fit(self, train_dataset, val_dataset):
        optimizer = self._build_optimizer()
        self.loss_function = self._build_loss()
        loader = self._loader(train_dataset, shuffle=True)

        best_val, best_state = float('inf'), None

        for epoch in range(self.max_epoch):
            nn.Module.train(self, True)
            total_loss, total_n = 0.0, 0

            for batch in loader:
                loss, n = self._compute_loss(self._to_device(batch))

                optimizer.zero_grad()
                loss.backward()
                if self.clip_norm:
                    nn.utils.clip_grad_norm_(self.parameters(), self.clip_norm)
                optimizer.step()

                total_loss += loss.item() * n
                total_n += n

            train_loss = total_loss / total_n
            val = self._evaluate(val_dataset)

            if val['Loss'] < best_val:
                best_val = val['Loss']
                best_state = copy.deepcopy(self.state_dict())

            metrics = {'Epoch': epoch, 'Loss': train_loss}
            metrics.update({'Val' + k: v for k, v in val.items()})
            if self.history:
                self.history.record(**metrics)
            print(metrics)

        if self.history:
            self.history.save()
        if best_state is not None:
            self.load_state_dict(best_state)   # restore best-validation weights

    def run(self):
        print('method running...')
        self._build_layers()
        self.to(self.device)
        print('--start training...')
        self.fit(self.data['train'], self.data['val'])
        print('--start testing...')
        return self._test(self.data['test'])