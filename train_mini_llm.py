import torch
import torch.nn as nn
from torch.nn import functional as F
from mini_llm_model import MiniLlmModel
from utils import parse_args, load_config, Tokenizer

if __name__ == "__main__":

    # Get the input arguments
    args = parse_args()

    # Get the config parameters
    config = load_config(args.config)

    # Declare seed for reproducibility
    torch.manual_seed(1337)

    # Hyperparameters Large Scale
    batch_size = config.get("batch_size")
    context_length = config.get("context_length")
    n_heads = config.get("n_heads")
    n_layers = config.get("n_layers")
    eval_interval = config.get("eval_interval")
    eval_iterations = config.get("eval_iterations")
    max_iters = config.get("max_iters")
    emb_size = config.get("emb_size")
    learning_rate = config.get("learning_rate")

    # Fetch device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Open the file
    with open(args.corpus , 'r', encoding='utf-8') as f:
        text = f.read()

    # Print first 1000 chars of the file
    # print(text[0:1000])

    # Make the token vocabulary (all chars and punctuations)
    vocab = sorted(list(set(text)))
    vocab_size = len(vocab)
    print(vocab)
    print(vocab_size)

    # Make tokenizer
    tokenizer = Tokenizer(vocab)
    print(tokenizer.detokenize(tokenizer.tokenize("happy to build character level mini-llm")))

    # Tokenize the entire data and store it in a tensor
    data = torch.tensor(tokenizer.tokenize(text))

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
    model = MiniLlmModel(vocab_size, emb_size, context_length, n_heads, n_layers)
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
    min_val_loss = float('inf')
    for i in range(max_iters):

        # Every once in a while calculate loss on train and val sets
        if i % eval_interval == 0 or i == (max_iters-1):
            losses = estimate_loss(eval_iterations=eval_iterations)
            print(f"iteration {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            # Save the trained model conditionally
            if losses['val'] < min_val_loss:
                torch.save(model.state_dict(), f"./checkpoints/{config.get('name')}.pth")
                # print(f"Model saved to ./checkpoints/{config.get('name')}.pth")
                min_val_loss = losses['val']

        xb, yb = get_batch('train', batch_size=batch_size)

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    model.eval()

    # Try generating sample text
    idx = torch.zeros((1,1), dtype=torch.long, device=device)
    print(tokenizer.detokenize(model.generate(idx, max_new_tokens=400)[0].tolist()))