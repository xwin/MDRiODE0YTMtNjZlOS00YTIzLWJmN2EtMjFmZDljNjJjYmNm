'''
Bidirectional GRU sentiment classifier (2 logits, CrossEntropyLoss)
'''

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from local_code.stage_4_code.Method_RNN_Base import Method_RNN_Base


class Method_RNN_Reviews(Method_RNN_Base):
    embed_dim = 100
    hidden_dim = 128
    num_layers = 1
    bidirectional = True
    dropout = 0.3

    def __init__(self, mName, mDescription):
        super().__init__(mName, mDescription)

    def _build_layers(self):
        train = self.data['train']
        self.embedding = nn.Embedding(train.vocab_size, self.embed_dim,
                                      padding_idx=train.pad_idx)
        self.rnn = nn.GRU(self.embed_dim, self.hidden_dim, num_layers=self.num_layers,
                          batch_first=True, bidirectional=self.bidirectional,
                          dropout=self.dropout if self.num_layers > 1 else 0.0)
        self.drop = nn.Dropout(self.dropout)
        self.fc = nn.Linear(self.hidden_dim * (2 if self.bidirectional else 1), 2)

    def forward(self, x, lengths):
        '''x: [batch, time] ids, lengths: [batch] -> logits [batch, 2]'''
        emb = self.drop(self.embedding(x))
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        _, h = self.rnn(packed)
        if self.bidirectional:
            h = torch.cat((h[-2], h[-1]), dim=1)
        else:
            h = h[-1]
        return self.fc(self.drop(h))

    def _compute_loss(self, batch):
        x, lengths, y = batch
        return self.loss_function(self.forward(x, lengths), y), y.size(0)

    def _evaluate(self, dataset):
        nn.Module.train(self, False)
        total_loss, correct, n = 0.0, 0, 0
        with torch.no_grad():
            for batch in self._loader(dataset, shuffle=False):
                x, lengths, y = self._to_device(batch)
                logits = self.forward(x, lengths)
                total_loss += self.loss_function(logits, y).item() * y.size(0)
                correct += (logits.argmax(dim=1) == y).sum().item()
                n += y.size(0)
        return {'Loss': total_loss / n, 'Accuracy': correct / n}

    def _test(self, dataset):
        nn.Module.train(self, False)
        preds, trues = [], []
        with torch.no_grad():
            for batch in self._loader(dataset, shuffle=False):
                x, lengths, y = self._to_device(batch)
                preds.append(self.forward(x, lengths).argmax(dim=1).cpu())
                trues.append(y.cpu())
        return {'pred_y': torch.cat(preds), 'true_y': torch.cat(trues)}