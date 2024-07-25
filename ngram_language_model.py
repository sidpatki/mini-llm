import torch
import torch.nn as nn
from torch.nn import functional as F

# Declare seed for reproducibility
torch.manual_seed(1337)

# Fetch device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SingleHeadSelfAttention(nn.Module):

    def __init__(self, ip_emb_size, op_emb_size, context_length, dropout=0.2):
        '''
        Self attention block
        op_emb_size is the dimention of the key, query and value vectors
        '''
        super().__init__()
        
        # Declare linear layers to convert token embeddings into k, q, v
        self.key = nn.Linear(ip_emb_size, op_emb_size, bias = False)
        self.query = nn.Linear(ip_emb_size, op_emb_size, bias = False)
        self.val = nn.Linear(ip_emb_size, op_emb_size, bias = False)

        # Just a declaration of mask. 
        # Its declared as a buffer as its not a network parameter
        self.register_buffer("mask", torch.tril(torch.ones(context_length, context_length)))

        # Dropout some attentions for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Get dimentions
        B, T, C = x.shape

        # Make keys and queries
        k = self.key(x) # B, T, C
        q = self.query(x) # B, T, C

        # Compute attention
        #  Q @ Kt / sqrt(dim_k)
        att = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5 # B, T, T

        # Mask attention to prevent future tokens influencing past tokens
        # This line should be commented if using bidirectional context
        # Note that [:T, :T] slicing is important because during inferance,
        # initially, the T will be less than context window.
        att = att.masked_fill(self.mask[:T, :T] == 0, float("-inf")) # B, T, T

        # Apply softmax along each column of B, T,  T
        att = F.softmax(att, dim = -1)

        # Apply dropout
        att = self.dropout(att)

        # Calculate value
        v = self.val(x) # B, T, C

        # Calculate weighted average
        out = att @ v # B, T, C

        return out

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_heads, ip_emb_size, op_emb_size, context_length, dropout=0.2):
        super().__init__()
        # Define a module list which runs varios modules in parallel
        # Since multi headed self attention concatinates embeddings
        # we make the op_emb_size smaller. 
        # e.g. if we want the final embeddings to have size 32
        # and there are 4 heads then we set op_emb_size to 8 so that,
        # after concatinating all resulting embeddings from all heads
        # the final embedding will be of 32 size        
        self.heads = nn.ModuleList([SingleHeadSelfAttention(ip_emb_size, op_emb_size//n_heads, context_length, dropout) for _ in range(n_heads)])
        
        # Define a projection layer that changes the size of the output of this
        # to match the input size. This is useful because we use skip connections
        # e.g. y = x + F(x). We use projection layer to make sure that dim of F(x) == x
        self.proj_layer = nn.Linear(op_emb_size, ip_emb_size)

        # Add dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim = -1)
        out = self.proj_layer(out)
        out = self.dropout(out)

        return out

class MLP(nn.Module):
    '''
    Simple MLP which operates on each embedding 
    after it passes through attention
    '''
    def __init__(self, in_size, out_size, dropout=0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_size, out_size * 4),
            nn.ReLU(),
            nn.Linear(out_size * 4, out_size), # projection layer 
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # input is B, T, C
        # output is also B, T, C
        x = self.network(x)
        return x


class AttentionPlusMLPBlock(nn.Module):
    def __init__(self, n_heads, emb_size, context_length, dropout=0.2):
        super().__init__()

        # Declare the multi headed self attention block
        self.sa = MultiHeadSelfAttention(n_heads, emb_size, emb_size, context_length, dropout)

        # MLP
        # attention mechanism ingests the context, but MLP further
        # lets them model process the resulting embeddings
        # Notice that MLP has a non-linearity but the linear layers
        # in attention block were strictly linear
        self.mlp = MLP(emb_size, emb_size, dropout)

        # Layer normalizations
        # This normalizes the activations of neurons to have 0 mean and 1 variance
        self.ln1 = nn.LayerNorm(emb_size)
        self.ln2 = nn.LayerNorm(emb_size)

    def forward(self, x):
        '''
        Applies sa and MLP with skip connections
        '''
        # Pass through self attention
        x = x + self.sa(self.ln1(x)) # B, T, C

        # MLP
        x = x + self.mlp(self.ln2(x)) # B, T, C

        return x


class NgramLanguageModel(nn.Module):

    def __init__(self, vocab_size, emb_size, context_length, n_heads, n_layers, dropout=0.2):
        super().__init__()
        
        # Make a vocab size by vocab size tensor of real values
        # This is representing every character with a vocab size vector
        self.token_embedding_table = nn.Embedding(vocab_size, emb_size)
        
        # Make a positional embedding table
        self.positional_embedding_table = nn.Embedding(context_length, emb_size)
        
        # Add SA + MLP blocks
        self.blocks = nn.Sequential(*[AttentionPlusMLPBlock(n_heads, emb_size, context_length, dropout) for _ in range(n_layers)])

        # Layer norm
        self.ln = nn.LayerNorm(emb_size)

        # This converts low dim embeddings into vocab size logits
        # Which are sampled to get next character
        self.lm_head = nn.Linear(emb_size, vocab_size)

        # Store context length
        self.context_length = context_length


    def forward(self, inputs, targets=None):
        '''
        This function performs a forward pass on the input.
        It also optionally calculates the loss.
        '''

        # Fetch dimentions
        # inputs and targets are both B, T dimentional
        B, T = inputs.shape

        # Get token embeddings
        tok_emb = self.token_embedding_table(inputs) # B, T, C

        # Get positional embeddings
        pos_emb = self.positional_embedding_table(torch.arange(T, device=device)) # B, T, C

        # Add pos emb to tok emb 
        emb = tok_emb + pos_emb # B, T, C

        # Pass embeddings through series of SA + MLP blocks
        emb = self.blocks(emb)
        
        # Pass through layer normalization
        emb = self.ln(emb)

        # Converts from embeddind dim to vocab size tensors.
        # This means that for every input token, we now have
        # a logit distribution over vocabulary. This is also 
        # called as upsampling. We can convert the last token's
        # logits into a distribution and sample from it to 
        # predict next token
        logits = self.lm_head(emb)

        # Return logits if the target is none
        if targets is None:
            return logits, None
            
        # Change dimentions to follow pytorch convention used in cross entropy loss
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)

        # Cross entropy loss. This applies softmax to the logits internally :)
        loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idxs, max_new_tokens):

        # Loop for max_new_tokens times
        for _ in range(max_new_tokens):
            # Crop the list of token indices 
            # such that only last n tokens are provided
            # as context. Because the attention head has
            # fixed size.
            in_context_idxs = idxs[:, -self.context_length:]

            # Get logits for tokens that are in context
            # E.g. if the text was  "hello world"
            # This will return logits for last n characters
            logits, _ = self(in_context_idxs)

            # Slice the logits tensor to get logits 
            # of only last character.
            logits = logits[:,-1,:]

            # Convert to distribution
            probs = F.softmax(logits, dim=-1)

            # Sample from a multinomial distribution
            pred_token_idx = torch.multinomial(probs, num_samples = 1)

            # Append the predicted
            idxs = torch.cat((idxs, pred_token_idx), dim=1)

        return idxs


if __name__ == "__main__":
    # Create model
    model = NgramLanguageModel(vocab_size=26, emb_size=32, context_length=8, n_heads=2, n_layers=2, dropout=0.2)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"This model has {total_params} parameters")