import torch
import torch.nn as nn
from torch.nn import functional as F

device= "cuda" if torch.cuda.is_available() else "cpu"
print("model running in: ",device)

block_size= 8
batch_size= 32
eval_iters= 200
learning_rate= 1e-2
max_iters= 3000

torch.manual_seed(1377)

#load data
with open("input.txt", "r", encoding="utf-8") as f:
        text= f.read()

#here are all unique characters that can occur in the text
chars= sorted(list(set(text)))
vocab_size= len(chars)

#create a mapping from characters to integers
str_int= {ch:i for i, ch in enumerate(chars)}
int_str= {i:ch for i, ch in enumerate(chars)}

#encoder takes string "s" as input, generates value of each character "c" from dictionary str_int and gives list of integer as output
encode= lambda s: [str_int[c] for c in s]
#decoder takes a list  of integer i as input, generates value of each integer i from dictionary int_str and join character to make string
decode= lambda l: "".join(int_str[i] for i in l)

#train test splits
data= torch.tensor(encode(text), dtype= torch.long)
n= int(len(data) * 0.9)
train_data= data[:n]
val_data= data[n:]

#data loading
def get_batch(split):
    #generates a small batch of data of input x and target y
    data= train_data if split == "train" else val_data
    ix= torch.randint(len(data) - block_size, (batch_size,)) #tensor of random 32 starting point data from range [0,992] 
    x= torch.stack([data[i: block_size + i] for i in ix]) #list of data starting point from ix, and list stacked to form tensor one after another 
    """
    https://pytorch.org/docs/stable/generated/torch.stack.html
    """
    y= torch.stack([data[i+1: i+ block_size +1] for i in ix])
    x,y= x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out= {}
    model.eval()  #this disables the Dropout as we dont need randomness during estimation.(also done during inference and evaluation)
    for split in ['train', 'val']:
        losses= torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y= get_batch(split)
            logits, loss= model(X,Y)
            losses[k]= loss.item()
        out[split]= losses.mean()
    model.train()
    return out


class BigramLanguageModel(nn.Module):
     
    def __init__(self, vocab_size):
          super().__init__() #this calls the constructor of parent class, here parent class nn.Module 's constructor is executed first
          self.token_embedding_table= nn.Embedding(vocab_size, vocab_size) #first argument vocab_size means unique tokens, and second is dimension of embedding
          """
            In this case, the size of the embeddings is set to vocab_size,
            so each token will be represented as a vector whose dimensionality is equal to the total number of tokens in the vocabulary.
          """

    def forward(self, idx, targets=None):
        #idx and targets are both (B, T) tensors of integers
        logits= self.token_embedding_table(idx) # (B, T, C)
        """
        this is a type of lookup table that maps discrete token indices to continuous vectors representations. In NLP many sequence tasks, words
        are represented by indices rather than one hot vectors, as one hot vectors has high dimension and sparse.
        Assume vocab_size = 5. So, we have 5 tokens in our vocabulary: ['A', 'B', 'C', 'D', 'E'].

        Suppose batch_size = 2 and sequence_length = 3. This means we're processing 2 sequences, each containing 3 tokens.

        Let's say our input idx (token indices) looks like this:
        idx = tensor([
                        [0, 1, 2],  # First sequence of token indices (e.g., "A B C")
                        [3, 4, 0]   # Second sequence of token indices (e.g., "D E A")
                    ])

        The embedding layer self.token_embedding_table has a shape (5, 5) (because vocab_size = 5)
        Each token index (0 to 4) maps to a unique 5-dimensional vector.
        For example, token 0 might map to [0.1, -0.2, 0.3, 0.4, -0.5].
        These embeddings are randomly initialized but learned during training.

        tensor([
                [  # First sequence
                    [0.1, -0.2, 0.3, 0.4, -0.5],  # Embedding for token 0 ("A")
                    [0.5, -0.3, 0.2, -0.1, 0.6],  # Embedding for token 1 ("B")
                    [-0.3, 0.7, 0.1, 0.2, -0.8]   # Embedding for token 2 ("C")
                ],
                [  # Second sequence
                    [-0.4, 0.2, 0.9, -0.7, 0.5],  # Embedding for token 3 ("D")
                    [0.3, 0.6, -0.2, 0.1, -0.9],  # Embedding for token 4 ("E")
                    [0.1, -0.2, 0.3, 0.4, -0.5]   # Embedding for token 0 ("A")
                ]
            ])

        """

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape          
            logits = logits.view(B*T, C)        #pytorch expects the logits as Batch, Channel, Time(B,C,T), but our logits had previously (B, T, C)
            """
            logits = tensor([
                            [
                                [2.3, -1.5, 0.8, 0.1, -0.2],  # Predictions for token 1 in sequence 1
                                [1.2,  3.0, 0.1, -0.5, 0.3],  # Predictions for token 2 in sequence 1
                                [0.5,  2.5, 1.5,  0.2, -0.4]  # Predictions for token 3 in sequence 1
                            ],
                            [
                                [1.1,  0.3, 1.7,  0.0,  2.5],  # Predictions for token 1 in sequence 2
                                [0.2, -0.3, 2.1,  1.3, -0.1],  # Predictions for token 2 in sequence 2
                                [1.0,  2.7, 0.4,  1.9,  0.2]   # Predictions for token 3 in sequence 2
                            ]
                        ])  # Shape: (2, 3, 5)

            targets = tensor([
                [0, 1, 2],  # True classes for sequence 1
                [3, 1, 0]   # True classes for sequence 2
            ])  # Shape: (2, 3)


            logits = logits.view(6, 5)
            logits = tensor([
                            [ 2.3, -1.5, 0.8,  0.1, -0.2],
                            [ 1.2,  3.0, 0.1, -0.5,  0.3],
                            [ 0.5,  2.5, 1.5,  0.2, -0.4],
                            [ 1.1,  0.3, 1.7,  0.0,  2.5],
                            [ 0.2, -0.3, 2.1,  1.3, -0.1],
                            [ 1.0,  2.7, 0.4,  1.9,  0.2]
                        ])  # Shape: (6, 5)

            """
            targets = targets.view(B*T)
            """
            targets = tensor([0, 1, 2, 3, 1, 0])  # Shape: (6)
            """
            loss=  F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        #idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            logits, loss= self(idx) 
            """
            returned from function forward, and nn.Module class is setup in such a way that calling instance of class automatically calls function forward
            """
            logits= logits[:,-1,:]  #becomes (B,C), focus on last time stamp
            probs= F.softmax(logits, dim=-1) #(B,C), apply softmax to get probabilities
            idx_next= torch.multinomial(probs, num_samples=1) #(B,1),  sample from distribution and choose one
            idx= torch.cat((idx, idx_next), dim=1) #(B, T+1)  append sampled index to the running sequence
            """
            Logits (B, T, C): tensor([[[ 0.1956,  0.8332, -0.3424, -0.0729,  0.6028],
                    [ 1.4237, -0.4964, -0.0199, -1.0700,  0.7314],
                    [ 1.1331,  0.4562,  0.1021, -1.1100,  0.7361]],

                    [[ 0.7881,  0.1230, -1.3852,  0.0794,  1.0223],
                    [ 0.5627,  0.7889, -0.1097,  1.1121, -1.0153],
                    [ 0.0421,  0.6653,  0.3952, -0.1804, -0.0772]]])

            Logits for last time step (B, C): tensor([[ 1.1331,  0.4562,  0.1021, -1.1100,  0.7361],
                    [ 0.0421,  0.6653,  0.3952, -0.1804, -0.0772]])

            Probabilities (B, C): tensor([[0.3145, 0.2485, 0.2225, 0.0727, 0.1418],
                    [0.2329, 0.2763, 0.2295, 0.1327, 0.1286]])

            Sampled next token indices (B, 1): tensor([[0],
                    [1]])

            Current input sequence (B, T): tensor([[4, 1, 3],
                    [2, 4, 0]])

            Updated input sequence (B, T+1): tensor([[4, 1, 3, 0],
                    [2, 4, 0, 1]])
            """
        return idx
                
model= BigramLanguageModel(vocab_size)
m= model.to(device)

optimizer= torch.optim.AdamW(model.parameters(), lr= learning_rate)
"""
In Adam, the weight decay term interacts with the adaptive learning rate, 
which means that the regularization can inadvertently influence the dynamics of the optimization process.

In AdamW, the weight decay acts independently, maintaining the momentum and adaptive learning rate behavior for the gradient update, 
while still providing the benefits of regularization.
"""

for iter in range(max_iters):
    #every once in a while evaluate the loss on train and val sets
    if iter % eval_iters == 0:
        losses= estimate_loss()
        print(f"step: {iter} \n train loss: {losses["train"]: 0.4f} \n val loss: {losses["val"]: 0.4f}")

    #sample a batch of data
    xb, yb= get_batch('train')

    #evaluate the loss
    logits, loss= model(xb, yb)
    optimizer.zero_grad(set_to_none= True)
    loss.backward()
    optimizer.step()

#generate from the model
context= torch.zeros((1,1), dtype= torch.long, device= device)      #start of sequence in tensor
print(decode(m.generate(context, max_new_tokens= 500)[0].tolist())) #generate 500 new tokens starting from context, convert token id to list and then decode it