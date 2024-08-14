import torch
import torch.nn as nn
from torch.nn import functional as F
from mini_llm_model import MiniLlmModel
import argparse
import json

def load_config(path):
    """
    Method to load config file
    """
    with open(path, 'r') as file:
        config = json.load(file)
    return config

def parse_args():
    """
    Method to parse input arguments
    """

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Demo to run the mini-llm model')

    # Define expected arguments
    parser.add_argument('--corpus', type=str, required=True, help='The input .txt file for generating vocab')
    parser.add_argument('--model', type=str, required=True, help='The model .pth file to run')
    parser.add_argument('--config', type=str, required=True, help='The config .json file')

    # Parse the arguments
    args = parser.parse_args()

    return args

if __name__ == "__main__":

    # Get the input arguments
    args = parse_args()

    # Get the config parameters
    config = load_config(args.config)

    # Declare seed for reproducibility
    torch.manual_seed(1337)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Open the file
    with open(args.corpus, 'r', encoding='utf-8') as f:
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
    model = MiniLlmModel(vocab_size, config.get("emb_size"), config.get("context_length"), config.get("n_heads"), config.get("n_layers"))
    model.load_state_dict(torch.load(args.model))
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