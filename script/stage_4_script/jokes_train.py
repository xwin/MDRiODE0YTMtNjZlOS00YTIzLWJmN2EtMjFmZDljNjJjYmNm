import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root; adjust if needed
from local_code.stage_4_code.Dataset_Loader_Jokes import Dataset_Loader_Jokes
from local_code.stage_4_code.Method_RNN_Jokes import Method_RNN_Jokes, detokenize

torch.manual_seed(0)
random.seed(0)

loader = Dataset_Loader_Jokes("jokes", "full run")
loader.dataset_source_folder_path = "../../data/stage_4_data/text_generation"
data = loader.load()

model = Method_RNN_Jokes("rnn", "jokes")
model.rnn_type = "GRU"
model.embed_dim = 128
model.hidden_dim = 128
model.dropout = 0.3
model.max_epoch = 30
model.batch_size = 64
model.data = data

print(model.run())      # test loss/perplexity over known-word targets (best-validation epoch)

# prompts: first 3 words of 5 test jokes (not seen in training), plus made-up ones
inv = loader.inv_vocab
test_prompts = []
for seq in data["test"].ids:
    ws = [inv[i] for i in seq[1:4]]
    if all(w[0].isalnum() for w in ws) and loader.unk_idx not in seq[1:4]:
        test_prompts.append(ws)
prompts = random.sample(test_prompts, 5) + [
    ["why", "can't", "robots"], ["my", "cat", "is"], ["what", "did", "quokka"]]

for p in prompts:
    print("\nPROMPT:", " ".join(p))
    for _ in range(3):
        toks = model.generate(p, loader.vocab, loader.inv_vocab, temperature=1.0, top_k=10)
        print("  RAW :", " ".join(toks))
        print("  TEXT:", detokenize(toks))