import torch
import torch.nn as nn
from torch.nn import functional as F
from mini_llm_model import MiniLlmModel

# Hyperparameters Small Scale
context_length = 8
batch_size = 32
n_heads = 4
n_layers = 4
eval_interval = 1000
eval_iterations = 1000
max_iters = 10000
emb_size = 32
learning_rate = 1e-3
dropout = 0.2

# Hyperparameters Large Scale
# batch_size = 64
# context_length = 1024
# n_heads = 4
# n_layers = 4
# eval_interval = 1000
# eval_iterations = 1000
# max_iters = 5000
# emb_size = 256
# learning_rate = 3e-4
# dropout = 0.2

# Declare seed for reproducibility
torch.manual_seed(1337)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Open the file
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Make the token vocabulary (all chars and punctuations)
vocab = sorted(list(set(text)))
vocab_size = len(vocab)
print(vocab)
print(vocab_size)

# Create char to int and reverse mapping
# Make tokenizer
stoi = { char : i for i, char in enumerate(vocab)}
itos = { i : char for i, char in enumerate(vocab)}

tokenize = lambda s: [ stoi[char] for char in s]
detokenize = lambda l: "".join([ itos[i] for i in l])

# Create model
model = MiniLlmModel(vocab_size, emb_size, context_length, n_heads, n_layers)
model.load_state_dict(torch.load('char_llm.pth'))
model = model.to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"This model has {total_params} parameters")
model.eval()

# Try generating sample text
idx = torch.zeros((1,1), dtype=torch.long, device=device)
print(detokenize(model.generate(idx, max_new_tokens=400)[0].tolist()))

# Try generating text with custom prompt
prompt = "Hello world, this is a large language model. My name is?"
idx = torch.tensor(tokenize(prompt), dtype=torch.long, device=device)
idx = idx.reshape(1, len(prompt))
print("\n\n\n")
print(detokenize(model.generate(idx, max_new_tokens=400)[0].tolist()))
breakpoint()