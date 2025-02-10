import math
import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class MLP(nn.Module):
    def __init__(self, params):
        super(MLP, self).__init__()
        self.embedding  = nn.Embedding(params.p, params.embed_dim)
        self.linear1r   = nn.Linear(params.embed_dim, params.hidden_size, bias = True)
        self.linear1l   = nn.Linear(params.embed_dim, params.hidden_size, bias = True)
        self.linear2    = nn.Linear(params.hidden_size, params.p, bias = False)
        self.act        = nn.GELU()
        self.vocab_size = params.p

    def forward(self, x):
        x1  = self.embedding(x[..., 0])
        x2  = self.embedding(x[..., 1])
        x0  = [x1, x2]
        x1  = self.linear1r(x1)
        x2  = self.linear1l(x2)
        x   = x1 + x2
        x   = self.act(x)
        x   = self.linear2(x)

        return x0, x

def test(model, dataset, device):
    n_correct   = 0
    total_loss  = 0
    model.eval()
    loss_fn     = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in dataset:
            x, y    = x.to(device), y.to(device)
            _, out     = model(x)
            loss    = loss_fn(out, y)
            total_loss += loss.item()
            pred    = torch.argmax(out)
            if pred == y:
                n_correct += 1
    return n_correct / len(dataset), total_loss / len(dataset)

def train(train_dataset, test_dataset, params, verbose = True):
    all_models  = []
    model       = MLP(params).to(params.device)
    optimizer   = torch.optim.Adam(model.parameters(), weight_decay = params.weight_decay, lr = params.lr)
    loss_fn     = torch.nn.CrossEntropyLoss()
        
    train_loader = DataLoader(train_dataset, batch_size = params.batch_size, shuffle = True)
    
    print_every = params.n_batches // params.print_times
    checkpoint_every = None
    if params.n_save_model_checkpoints > 0:
        checkpoint_every = params.n_batches // params.n_save_model_checkpoints

    loss_data = []
    if verbose:
        pbar = tqdm(total = params.n_batches, desc = "Training")
    for i in range(params.n_batches):
        batch   = next(iter(train_loader))
        X, Y    = batch
        X, Y    = X.to(params.device), Y.to(params.device)
        optimizer.zero_grad()
        _, out     = model(X)
        loss    = loss_fn(out, Y)
        loss.backward()
        optimizer.step()
                
        if checkpoint_every and (i + 1) % checkpoint_every == 0:
            #To implement: save models in the dic
            all_models += [deepcopy(model)]

        if (i + 1) % print_every == 0:
            val_acc, val_loss       = test(model, test_dataset, params.device)
            train_acc, train_loss   = test(model, train_dataset, params.device)
            gs = GradientSymmetry(model, p, False)
            loss_data.append({"batch": i+1, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc, "grad_symm": gs, })
            if verbose:
                pbar.set_postfix({"train_loss": f"{train_loss:.4f}", "train_acc": f"{train_acc:.4f}", "val_loss": f"{val_loss:.4f}", "val_acc": f"{val_acc:.4f}",})
                pbar.update(print_every)

    if verbose:
        pbar.close()
    df = pd.DataFrame(loss_data)
    pca(model, p)
    train_acc, train_loss   = test(model, train_dataset, params.device)
    val_acc, val_loss       = test(model, test_dataset, params.device)
    if verbose:
        print(f"Final Train Acc: {train_acc:.4f} | Final Train Loss: {train_loss:.4f}")
        print(f"Final Val Acc: {val_acc:.4f} | Final Val Loss: {val_loss:.4f}")
    return all_models, df

class ExperimentParams:
    n_batches: int = 20000 
    n_save_model_checkpoints: int = 0
    print_times: int = 50
    lr: float = 0.005
    batch_size: int = 128
    hidden_size: int = 256 
    embed_dim: int = 256 
    device: str = DEVICE
    weight_decay: float = 0.0002   
    random_seed: int = 0 

    def __init__(self, p):
        self.p = p

def GradientSymmetry(model, p, boo):
    data = [(a, b, c) for a in range(p) for b in range(p) for c in range(p)]
    random.Random(42).shuffle(data)
    data = data[:100]
    gs = 0
    for abc in data:
        a, b, c = abc
        x = torch.tensor([a, b], device = "cuda")
        embed, out = model(x) 
        embed[0].retain_grad()
        embed[1].retain_grad()
        out[c].backward(retain_graph = True)
        embed_gl = embed[0].grad.detach().cpu().numpy()
        embed_gr = embed[1].grad.detach().cpu().numpy()
        cos_sim = np.sum(embed_gl * embed_gr) / (np.sqrt(np.sum(embed_gl**2)) * np.sqrt(np.sum(embed_gr**2)))
        gs += cos_sim

    return gs/len(data)

def pca(model, p):
    we = model.embedding.weight

    pca = PCA(n_components = 12)
    pca_we = pca.fit_transform(we.detach().cpu().numpy())
    plt.figure(figsize = (30, 6))
    pi = math.pi

    for idx in range(12):
        comp = pca_we[:, idx]
        for num in range(1, p):
            vv = [comp[num * t % p] for t in range(p)] 
            print(vv)

            return 

curr_dic = os.path.join(os.getcwd(), "Datasets")
###
### Still need to implement that it reads from files the p value
###
files = [f for f in os.listdir(curr_dic) if os.path.isfile(os.path.join(curr_dic, f))]

train_files = []
test_files  =  []
for file in files:
    if file[:4] == "Test":
        test_files.append(file)
    else:
        train_files.append(file)

def get_seed(sub):
    return sub[-4]

train_files.sort(key = get_seed)
test_files.sort(key = get_seed)

p = int(sys.argv[1])
params = ExperimentParams(p)
torch.manual_seed(params.random_seed)

df_dic = {}
seeds = [9]
for seed_num in seeds: 
    for i in range(1):
        train_data  = torch.load(f"{curr_dic}/{train_files[seed_num]}", weights_only = True)
        test_data   = torch.load(f"{curr_dic}/{test_files[seed_num]}", weights_only = True)
        all_checkpointed_models, df = train(train_dataset = train_data, test_dataset = test_data, params = params)
        df_dic[f"Seed:{seed_num}_{i}"] = df
        
fig, (ax1, ax2) = plt.subplots(2, sharex = True)
fig.suptitle("Top Training, bottom Test")
for key in df_dic:
    ax1.plot(df_dic[key]["val_acc"], label = f"Val_acc {key}")
    ax1.plot(df_dic[key]["grad_symm"], label = f"GradSymm {key}")
#    ax1.plot(df_dic[key]["train_acc"], label = f"Train_acc {key}")
    ax1.legend()
    ax2.plot(df_dic[key]["val_loss"], label = f"Val_loss {key}")
    ax2.legend()
plt.ylabel("Correct answer %")
plt.xlabel("Checkpoint")
plt.show()

