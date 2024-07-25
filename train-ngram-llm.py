import torch
import torch.nn as nn
from torch.nn import functional as F
from ngram_language_model import NgramLanguageModel

# Declare seed for reproducibility
torch.manual_seed(1337)

# # Hyperparameters Small Scale
# context_length = 8
# batch_size = 32
# n_heads = 4
# n_layers = 4
# eval_interval = 1000
# eval_iterations = 1000
# max_iters = 10000
# emb_size = 32
# learning_rate = 1e-3
# dropout = 0.2

# Hyperparameters Large Scale
batch_size = 64
context_length = 256
n_heads = 4
n_layers = 4
eval_interval = 1000
eval_iterations = 1000
max_iters = 5000
emb_size = 256
learning_rate = 3e-4
dropout = 0.2

# Fetch device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Open the file
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Print first 1000 chars of the file
# print(text[0:1000])

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

print(detokenize(tokenize("happy to build char-llm")))

# Tokenize the entire data and store it in a tensor
data = torch.tensor(tokenize(text))

# Split data into train and val
n = int(len(data) * 0.9)
train_data = data[0:n]
val_data = data[n:]

# Example showing fetching one block of training
# samples and respective targets
x = train_data[:context_length]
y = train_data[1:context_length+1]
# for i in range(context_length):
#     print(f"when training sample is {x[:i+1]} target is {y[i]}")


def get_batch(split, batch_size=4):
    '''
    Method to get minibatch of data
    '''
    data = train_data if split == "train" else val_data
    indices = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack( [data[i:i+context_length] for i in indices] )
    y = torch.stack( [data[i+1:i+context_length+1] for i in indices] )
    x = x.to(device)
    y = y.to(device)
    return x, y 

# Create model
model = NgramLanguageModel(vocab_size, emb_size, context_length, n_heads, n_layers)
model = model.to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"This model has {total_params} parameters")

# Test the forward pass
xb, yb = get_batch('train')
logits, loss = model(xb, yb)

# Make optimizer object
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Method to calculate average loss on train and val batches
@torch.no_grad()
def estimate_loss(eval_iterations):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iterations)
        for k in range(eval_iterations):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    return out


# Training loop
model.train()
for i in range(max_iters):

    # Every once in a while calculate loss on train and val sets
    if i % eval_interval == 0 or i == (max_iters-1):
        losses = estimate_loss(eval_iterations=eval_iterations)
        print(f"iteration {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train', batch_size=batch_size)

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

model.eval()

# Save the trained model
torch.save(model.state_dict(), 'char_llm.pth')
print("Model saved to char_llm.pth")

# Try generating sample text
idx = torch.zeros((1,1), dtype=torch.long, device=device)
print(detokenize(model.generate(idx, max_new_tokens=400)[0].tolist()))