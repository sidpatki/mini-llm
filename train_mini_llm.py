import torch
import torch.nn as nn
from torch.nn import functional as F
from mini_llm_model import MiniLlmModel
from utils import parse_args, load_config, Tokenizer, Vocab, DataLoader

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

    # Make the token vocabulary (all chars and punctuations)
    vocab = Vocab(text)
    print(vocab)

    # Make tokenizer
    tokenizer = Tokenizer(vocab)
    print(tokenizer.detokenize(tokenizer.tokenize("happy to build character level mini-llm")))

    # Make dataloder
    data = torch.tensor(tokenizer.tokenize(text))
    data = DataLoader(data, train_fraction = 0.9)

    # Create model
    model = MiniLlmModel(vocab.size, emb_size, context_length, n_heads, n_layers)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"This model has {total_params} parameters")

    # # Test the forward pass
    # xb, yb = data.get_mini_batch('train', context_length, batch_size)
    # logits, loss = model(xb, yb)

    # Make optimizer object
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Method to calculate average loss on train and val batches
    # This is used to keep track of training.
    @torch.no_grad()
    def estimate_loss(eval_iterations):
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iterations)
            for k in range(eval_iterations):
                x, y = data.get_mini_batch(split, context_length, batch_size, device)
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

            # Conditionally save the trained model
            if losses['val'] < min_val_loss:
                torch.save(model.state_dict(), f"./checkpoints/{config.get('name')}.pth")
                # print(f"Model saved to ./checkpoints/{config.get('name')}.pth")
                min_val_loss = losses['val']

        # Get mini batch of data
        xb, yb = data.get_mini_batch('train', context_length, batch_size, device)

        # Pass data through model and get logits and loss
        # We ignore logits during training as we are not doing any generation
        # This is strictly supervised fine tuning (SFT)
        logits, loss = model(xb, yb)

        # Backpropogate the loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Testing the model after training
    model.eval()

    # Try generating sample text
    idx = torch.zeros((1,1), dtype=torch.long, device=device)
    print(tokenizer.detokenize(model.generate(idx, max_new_tokens=400)[0].tolist()))