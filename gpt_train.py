import torch
import torch.nn as nn
from torch.nn import functional as F

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available.")
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print("Number of GPUs available:", num_gpus)
    # Loop over available GPUs and print some information about each one
    for i in range(num_gpus):
        print(f"GPU {i}:")
        print("  Name:", torch.cuda.get_device_name(i))
        print("  Capability:", torch.cuda.get_device_capability(i))
else:
    print("CUDA is not available.")


# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 50
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 348
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

torch.manual_seed(2049)

with open('all_thoughts_cleaned.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# Encode seed 
def encode_seed(s): # encoder: take a string, output a tensor of integers
    return torch.tensor([stoi[c] for c in s], dtype=torch.long, device=device).unsqueeze(0)

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

    def generate_with_seed(self, seed, max_new_tokens):
        # seed is a string that will be used to start the generation
        idx = encode_seed(seed)
        return self.generate(idx, max_new_tokens)


# TRAIN 

model = GPTLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
# Save the trained model
torch.save(model.state_dict(), "trained_model.pth")


# GENERATE 
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

'''
# model trained, start generation
print(">+=============================+<")
print('Model trained, now start typing.\n')

from pynput import keyboard
import random
import os
from datetime import date




while True: 
    if str(input(">>> Type 'MEMORYHOARDINGOCD' to continue: ")) == 'MEMORYHOARDINGOCD':
        break


# Create file
today = date.today()
today.strftime("%d_%m_%Y")
filename = f"thoughts_{today}.txt"

if os.path.exists(filename):
    mode = "a+"  # append if the file exists
else:
    mode = "w"  # write (create) if the file does not exist

with open(filename, mode) as file:
    print(f"Date: {today}")
    print("Today's note: ")
    print('+=================================+')
    if mode == "w":
        print("Today's note hasn't been created yet, good work")
    else:
        file.seek(0)
        lines = file.readlines()
        lines = [line.rstrip('\n') for line in lines]
        
        if len(lines) > 0:  # check if the file has at least one line
            print(f"{lines[0]}...\n")

            if len(lines) > 1:  # check if the file has more than one line
                print(f"\n...{lines[-1]}")

    print('+=================================+')
    print()
    while True:
        print('+=====================+')
        print('Memory Hoard:')
        print(' [A] Write (Simply start writing)')
        print(' [B] Read')
        print()
        print('Therapy:')
        print(' [C] Exposure and Response Prevention') # You write a line, and have the choice to dump it 
        print(' [D] Mindfulness') #
        print(' [E] Write blank')
        print(' [F] Talk to someone')
        print()
        user_input = str(input('Thoughts? >>> '))
        if user_input.upper() == 'B':
            None # View all files with thoughts_date format, select one, read through it
        elif user_input.upper() == 'C':
            None # Asks for user input as string, then gives them option to delete it or write it to the file
        elif user_input.upper() == 'D':
            None # Auto generates empty thoughts and you have to let them pass by (emoji scenery scroll)
        elif user_input.upper() == 'E':
            None # Autocompletes thoughts
        elif user_input.upper() == 'F':
            None # Uses ChatGPT API who poses as therapist specialising in MHOCD 
        else:
            file.write(user_input + '\n')



class MyException(Exception): 
    def __init__(self, message=None):
        pass

class TypeListener:
    def __init__(self):
        self.count = 0
        self.limit = 20
        self.seed_string = ''

    def on_press(self, key):
        try:
            if key != keyboard.Key.enter:
                # Check if key pressed is alphanumeric
                char = key.char
                self.count += 1
                self.seed_string += char  # This is how you add the character to the string
                if self.count >= self.limit: # interrupt
                    generated_text = decode(m.generate_with_seed(self.seed_string, max_new_tokens=500)[0].tolist())
                    print(generated_text)
                    raise MyException(key)
        except AttributeError:
            pass  # Non-alphanumeric key pressed

def start_program():
    listener = TypeListener()
    with keyboard.Listener(on_press=listener.on_press) as listener:
        try:
            listener.join()
        except MyException as e:
            pass

start_program()
'''
