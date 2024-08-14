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
    parser.add_argument('--config', type=str, required=True, help='The config .json file')
    parser.add_argument('--model', type=str, required=False, help='The model .pth file')

    # Parse the arguments
    args = parser.parse_args()

    return args

class Tokenizer():
    """
    This class represents a character level tokenizer.
    """
    def __init__(self, vocab):

        # Store vocab list
        self.vocab = vocab

        # Create char 2 int and int 2 char mapping
        self.stoi = { char : i for i, char in enumerate(vocab)}
        self.itos = { i : char for i, char in enumerate(vocab)}

    def tokenize(self, s):
        return [self.stoi[char] for char in s]
    
    def detokenize(self, l):
        return "".join([self.itos[i] for i in l])

