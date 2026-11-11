'''
Unidirectional GRU/LSTM language model for joke generation (next-token prediction)
'''

import math
import torch
from torch import nn
import re

from local_code.stage_4_code.Method_RNN_Base import Method_RNN_Base


NO_SPACE_BEFORE = {'.', ',', '?', '!', ':', ';', ')', ']', '...', '-'}
NO_SPACE_AFTER = {'(', '[', '-'}


def detokenize(tokens):
    '''rule-based join of cleaned tokens into readable text (untested heuristics)'''
    out, in_quote, attach = [], False, True      # attach: no space before next token
    for t in tokens:
        if t == '"':
            if not in_quote and not attach:
                out.append(' ')
            out.append(t)
            attach = not in_quote                # after an opening quote, attach the next word
            in_quote = not in_quote
            continue
        if not attach and t not in NO_SPACE_BEFORE:
            out.append(' ')
        out.append(t)
        attach = t in NO_SPACE_AFTER
    text = ''.join(out)
    text = re.sub(r'(^|[.?!]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)
    return re.sub(r'\bi\b', 'I', text)

class Method_RNN_Jokes(Method_RNN_Base):
    rnn_type = 'GRU'        # 'GRU' or 'LSTM'
    embed_dim = 128
    hidden_dim = 128
    num_layers = 1
    dropout = 0.3
    max_epoch = 30          # guess; the best-validation epoch is restored anyway

    def __init__(self, mName, mDescription):
        super().__init__(mName, mDescription)

    def _build_layers(self):
        train = self.data['train']
        self.pad_idx = train.pad_idx
        self.vocab_size = train.vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=self.pad_idx)
        rnn_cls = nn.LSTM if self.rnn_type.upper() == 'LSTM' else nn.GRU
        self.rnn = rnn_cls(self.embed_dim, self.hidden_dim, num_layers=self.num_layers,
                           batch_first=True,
                           dropout=self.dropout if self.num_layers > 1 else 0.0)
        self.drop = nn.Dropout(self.dropout)
        self.fc = nn.Linear(self.hidden_dim, self.vocab_size)

    def _build_loss(self):
        # pad targets (padding and masked <unk>) don't count
        return nn.CrossEntropyLoss(ignore_index=self.pad_idx)

    def forward(self, x, h=None):
        '''x: [batch, time] ids, h: optional hidden state (GRU: tensor, LSTM: (h, c))
        -> logits [batch, time, vocab], new hidden state'''
        out, h = self.rnn(self.drop(self.embedding(x)), h)
        return self.fc(self.drop(out)), h

    def _loss_and_count(self, batch):
        x, lengths, y = batch           # lengths unused: right padding + unidirectional
        logits, _ = self.forward(x)
        loss = self.loss_function(logits.reshape(-1, self.vocab_size), y.reshape(-1))
        return loss, int((y != self.pad_idx).sum())

    def _compute_loss(self, batch):
        return self._loss_and_count(batch)

    def _evaluate(self, dataset):
        nn.Module.train(self, False)
        total, n = 0.0, 0
        with torch.no_grad():
            for batch in self._loader(dataset, shuffle=False):
                loss, k = self._loss_and_count(self._to_device(batch))
                total += loss.item() * k
                n += k
        loss = total / n
        return {'Loss': loss, 'Perplexity': math.exp(loss)}

    def _test(self, dataset):
        # placeholder: test loss/perplexity over known-word targets.
        # Generated samples and the evaluator wiring come later.
        return self._evaluate(dataset)

    def generate(self, prompt_words, vocab, inv_vocab, max_len=55, temperature=1.0, top_k=10):
        '''prompt_words: list of strings. Returns the whole joke as a list of word tokens
        (prompt words first). max_len counts all tokens, prompt included.'''
        pad, unk, bos, eos = (vocab[t] for t in ('<pad>', '<unk>', '<bos>', '<eos>'))
        words = [w.lower() for w in prompt_words]
        ids = [bos] + [vocab.get(w, unk) for w in words]   # unknown prompt words -> <unk>
        nn.Module.train(self, False)
        with torch.no_grad():
            x = torch.tensor([ids], device=self.device)
            h = None
            while len(words) < max_len:
                logits, h = self.forward(x, h)
                logits = logits[0, -1] / temperature
                logits[[pad, unk, bos]] = float('-inf')    # never output these
                if top_k:
                    kth = torch.topk(logits, top_k).values[-1]
                    logits[logits < kth] = float('-inf')
                nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1).item()
                if nxt == eos:
                    break
                words.append(inv_vocab[nxt])
                x = torch.tensor([[nxt]], device=self.device)
        return words