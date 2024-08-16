import torch
import torch.nn as nn
from torch.nn import functional as F
from mini_llm_model import MiniLlmModel
from utils import parse_args, load_config, Tokenizer, Vocab

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
    vocab = Vocab(text)
    print(vocab)

    # Make tokenizer
    tokenizer = Tokenizer(vocab)

    # Create model
    model = MiniLlmModel(vocab.size, config.get("emb_size"), config.get("context_length"), config.get("n_heads"), config.get("n_layers"))
    model.load_state_dict(torch.load(args.model))
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"This model has {total_params} parameters")
    model.eval()

    # # Try generating sample text
    # idx = torch.zeros((1,1), dtype=torch.long, device=device)
    # print(tokenizer.detokenize(model.generate(idx, max_new_tokens=400)[0].tolist()))

    # Try generating text with custom prompt
    prompt = "Harry p"
    idx = torch.tensor(tokenizer.tokenize(prompt), dtype=torch.long, device=device)
    idx = idx.reshape(1, len(prompt))
    print("\n\n\n")
    print(tokenizer.detokenize(model.generate(idx, max_new_tokens=400)[0].tolist()))