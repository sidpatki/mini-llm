import argparse
import json
import torch

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

        # Create char 2 int and int 2 char mapping
        self.stoi = { char : i for i, char in enumerate(vocab.tokens)}
        self.itos = { i : char for i, char in enumerate(vocab.tokens)}

    def tokenize(self, s):
        return [self.stoi[char] for char in s]
    
    def detokenize(self, l):
        return "".join([self.itos[i] for i in l])


class Vocab:
    """
    This class represents the token vocabulary scraped from the training data.
    """

    def __init__(self, text):
        # Find unique characters in the text 
        # and convert into list
        self.tokens = sorted(list(set(text)))

        # Save vocab size
        self.size = len(self.tokens)

    def __str__(self):
        return f"\nTokens: {self.tokens}\nVocab Size: {self.size}"

class DataLoader:
    """
    This class represents container for the dataset
    """

    def __init__(self, data, train_fraction=0.9):

        # Split data into train and val
        n = int(len(data) * train_fraction)
        self.train_data = data[0:n]
        self.val_data = data[n:]

        # # Example showing fetching one block of training
        # # samples and respective targets
        # x = self.train_data[:context_length]
        # y = self.train_data[1:context_length+1]
        # # for i in range(context_length):
        # #     print(f"when training sample is {x[:i+1]} target is {y[i]}")


    def get_mini_batch(self, split, context_length, batch_size, device):
        '''
        Method to get minibatch of data from the dataset
        '''
        data = self.train_data if split == "train" else self.val_data
        indices = torch.randint(len(data) - context_length, (batch_size,))
        x = torch.stack([data[i:i+context_length] for i in indices])
        y = torch.stack([data[i+1:i+context_length+1] for i in indices])
        x = x.to(device)
        y = y.to(device)
        return x, y
