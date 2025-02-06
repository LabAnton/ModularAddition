import random
from copy import deepcopy
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from devinterp.optim.sgld import SGLD
from devinterp.slt.sampler import estimate_learning_coeff_with_summary
from devinterp.utils import evaluate_ce

import typing
from typing import Type
import numpy as np

from devinterp.vis_utils import EpsilonBetaAnalyzer
from devinterp.utils import plot_trace

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ExperimentParams:
    p: int = 53
    n_batches: int = 50000
    n_save_model_checkpoints: int = 0 
    print_times: int = 200
    lr: float = 0.005
    batch_size: int = 128
    hidden_size: int = 48
    embed_dim: int = 12
    train_frac: float = 0.4
    non_mod_frac: float = 0.2
    random_seed: int = 0 
    device: str = DEVICE
    weight_decay: float = 0.0002

class MLP(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.embedding  = nn.Embedding(params.p, params.embed_dim)                      #Embedding layer 53 x 12 
        self.linear1r   = nn.Linear(params.embed_dim, params.hidden_size, bias = True)  #First right layer 12 x 48
        self.linear1l   = nn.Linear(params.embed_dim, params.hidden_size, bias = True)  #First left layer 12 x 48
        self.linear2    = nn.Linear(params.hidden_size, params.p, bias = False)         #Second layer 48 x 53
        self.act        = nn.GELU() 
        self.vocab_size = params.p

    #Model basically add up both embedding after one layer, then activation function and one last linear layer
    def forward(self, x):
        x1 = self.embedding(x[..., 0])
        x2 = self.embedding(x[..., 1])
        x1 = self.linear1l(x1)
        x2 = self.linear1r(x2)
        x = x1 + x2
        x = self.act(x)
        x = self.linear2(x)
        return x 

#Returns percentage of correct labeled data and average loss
def test(model, dataset, device):
    n_correct = 0
    total_loss = 0
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in dataset:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            total_loss += loss.item()
            pred = torch.argmax(out)
            if pred == y:
                n_correct += 1
    return n_correct / len(dataset), total_loss / len(dataset)

def train(train_dataset, test_dataset, params, verbose = True):
    #verbose is used to show stuff, if it is off then we will not see a progress bar etc.
    all_models = []
    model = MLP(params).to(params.device)
    optimizer = torch.optim.Adam(model.parameters(), weight_decay = params.weight_decay, lr = params.lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    train_loader = DataLoader(train_dataset, batch_size = params.batch_size, shuffle = True)
    
    print_every = params.n_batches // params.print_times
    checkpoint_every = None
    if params.n_save_model_checkpoints > 0:
        checkpoint_every = params.n_batches // params.n_save_model_checkpoints

    loss_data = []
    if verbose:
        pbar = tqdm(total = params.n_batches, desc = "Training")
    for i in range(params.n_batches):
        #Sample random batch of data
        batch = next(iter(train_loader))
        X, Y = batch
        X, Y = X.to(params.device), Y.to(params.device)
        #Gradient update; one batch is trained 
        optimizer.zero_grad()
        out = model(X)
        loss = loss_fn(out, Y)
        loss.backward()
        optimizer.step()
    
        #If we want to take checkpoints every checkpoint_every step the model is saved into a list.
        if checkpoint_every and (i + 1) % checkpoint_every == 0:
            all_models += [deepcopy(model)]

        #Every print_every time the train and test accuracy and loss is saved
        if (i + 1) % print_every == 0:
            val_acc, val_loss = test(model, test_dataset, params.device)
            train_acc, train_loss = test(model, train_dataset, params.device)
            #Loss data is saved into a dic with the batch.
            loss_data.append({"batch": i+1, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc,})
            if verbose:
                pbar.set_postfix({"train_loss": f"{train_loss:.4f}", "train_acc": f"{train_acc:.4f}", "val_loss": f"{val_loss:.4f}", "val_acc": f"{val_acc:.4f}",})
                pbar.update(print_every)
    if verbose:
        pbar.close()
    #dic with loss data is saved into a dataframe
    df = pd.DataFrame(loss_data)
    train_acc, train_loss = test(model, train_dataset, params.device)
    val_acc, val_loss = test(model, test_dataset, params.device)
    if verbose:
        print(f"Final Train Acc: {train_acc:.4f} | Final Train Loss: {train_loss:.4f}")  
        print(f"Final Val Acc: {val_acc:.4f} | Final Val Loss: {val_loss:.4f}")
    return all_models, df

#shuffles a list
def deterministic_shuffle(lst, seed):
    random.seed(seed)
    random.shuffle(lst)
    return lst

#create all combinations of paris 
def get_all_pairs(p):
    pairs = []
    for i in range(p):
        for j in range(p):
            pairs.append((i, j))
    return set(pairs)

#create dataset where data in modulo and without are split
def make_dataset(p):
    data_mod = []
    data_non_mod = []
    pairs = get_all_pairs(p)
    for a, b in pairs:
        if a + b < p:
            data_non_mod.append((torch.tensor([a, b]), torch.tensor((a + b) % p)))
        else:
            data_mod.append((torch.tensor([a, b]), torch.tensor((a + b) % p)))
    return data_non_mod, data_mod

#split data and introduce mod-nonmod split
def train_test_split(non_mod_dataset, mod_dataset, train_split_proportion, non_mod_proportion, seed):
    #first I get length of both datasets, then I calculate the split ratio for each
    l_non_mod           = len(non_mod_dataset)
    l_mod               = len(mod_dataset)
    train_frac          = int((l_mod+l_non_mod) * train_split_proportion)
    non_mod_frac        = int(train_frac * non_mod_proportion)
    
    idx_non_mod         = list(range(l_non_mod))
    idx_non_mod         = deterministic_shuffle(idx_non_mod, seed)
    idx_mod             = list(range(l_mod))
    idx_mod             = deterministic_shuffle(idx_mod, seed)
    
    train_data_non_mod  = idx_non_mod[:non_mod_frac]
    test_data_non_mod   = idx_non_mod[non_mod_frac:]

    train_data_mod      = idx_mod[:(train_frac - non_mod_frac)]
    test_data_mod       = idx_mod[(train_frac - non_mod_frac):]
    
    train_data  = [non_mod_dataset[i] for i in train_data_non_mod] + [mod_dataset[i] for i in train_data_mod]
    test_data   = [non_mod_dataset[i] for i in test_data_non_mod] + [mod_dataset[i] for i in test_data_mod]

    return deterministic_shuffle(train_data, seed), deterministic_shuffle(test_data, seed)

#create function which tells me more about how data where modulo and not is needed 
def num_modolo(dataset, prime):
    l = len(dataset)
    non_modulo = sum([0 if sum(data[0]) > prime-1 else 1 for data in dataset])/l  
    print("Modolo function:", non_modulo)
    return non_modulo 

params = ExperimentParams()
torch.manual_seed(params.random_seed)

list_non_mod_frac = [0.5, 0.525, 0.55, 0.575, 0.6]
df_dic = {}
for non_mod_frac in list_non_mod_frac:
    non_mod_dataset, mod_dataset = make_dataset(params.p)
    #print("NonMod", len(non_mod_dataset), "Mod", len(mod_dataset))
    print("Train_frac:", params.train_frac, "Mod_frac", params.non_mod_frac)
    train_data, test_data = train_test_split(non_mod_dataset, mod_dataset, params.train_frac, non_mod_frac, params.random_seed)
    num_modolo(train_data, params.p)
    print("Train_data:", len(train_data), "Test_data:", len(test_data))

    all_checkpointed_models, df = train(train_dataset= train_data, test_dataset = test_data, params = params)
    df_dic[f"Split_{non_mod_frac}"] = df
    
for key in df_dic:
    plt.plot(df_dic[key]["val_acc"], label = f"Val_acc {key}")
    plt.plot(df_dic[key]["train_acc"], label = f"Train_acc {key}")
    plt.legend()
plt.ylabel("Correct answer %")
plt.xlabel("Checkpoint")
plt.title(f"Train & test correct answer % for modular addition with p = {params.p}")
plt.show()

for key in df_dic:
    plt.plot(df_dic[key]["val_loss"], label = f"Val_loss {key}")
#    plt.plot(df_dic[key]["train_loss"], label = f"Train_loss {key}")
    plt.legend()
plt.ylabel("Loss")
plt.xlabel("Ch  eckpoint")
plt.title(f"Train & test loss for modular addition with p = {params.p}")
plt.show()

exit()
def estimate_llc_given_model(model: torch.nn.Module, loader: torch.utils.data.DataLoader, evaluate: typing.Callable, epsilon: float, beta: float, sampling_method: Type[torch.optim.Optimizer] = SGLD, localization: float = 5, num_chains: int = 2, num_draws: int = 500, num_burnin_steps: int = 0, num_steps_bw_draws: int = 1, device: torch.device = DEVICE, online: bool = True, verbose: bool = False,):
    sweep_stats = estimate_learning_coeff_with_summary(model, loader = loader, evaluate = evaluate, sampling_method = sampling_method, optimizer_kwargs = dict(lr = epsilon, localization=localization, nbeta = beta), num_chains = num_chains, num_draws = num_draws, num_burnin_steps = num_burnin_steps, num_steps_bw_draws = num_steps_bw_draws, device = device, online = online, verbose = verbose,)
    sweep_stats["llc/trace"] = np.array(sweep_stats["llc/trace"])
    return sweep_stats

loader = DataLoader(train_data, shuffle = True, batch_size = params.batch_size)
analyzer = EpsilonBetaAnalyzer()
analyzer.configure_sweep(llc_estimator = estimate_llc_given_model, llc_estimator_kwargs = dict(model = all_checkpointed_models[-1], evaluate = evaluate_ce, device = DEVICE, loader = loader,), min_epsilon = 3e-5, max_epsilon = 3e-1, epsilon_samples = 5, min_beta = None, max_beta = None, beta_samples = 5, dataloader = loader,)
analyzer.sweep()
analyzer.plot()
analyzer.plot(div_out_beta = True)
analyzer.fig.show()

lr = 3e-3
gamma = 5
nbeta = 2
num_draws = 500
num_chains = 2
learning_coeff_stats = estimate_learning_coeff_with_summary(all_checkpointed_models[-1], loader = DataLoader(train_data, batch_size = params.batch_size, shuffle = True), evaluate = evaluate_ce, sampling_method = SGLD, optimizer_kwargs = dict(lr = 0.03, nbeta = 2.0, localization = 5.0), num_chains = 3, num_draws = 1500, device = DEVICE, online = True) 
trace = learning_coeff_stats["loss/trace"]
plot_trace(trace, "Loss", x_axis = "Step", title = f"Loss Trace, avg LLC = {sum(learning_coeff_stats["llc/means"])/len(learning_coeff_stats["llc/means"]):.2f}", plot_mean = False, plot_std = False, fig_size = (12, 9), true_lc = None,)

llcs = [estimate_learning_coeff_with_summary(model_checkpoint, loader = DataLoader(train_data, batch_size = params.batch_size, shuffle = True), evaluate = evaluate_ce, sampling_method = SGLD, optimizer_kwargs = dict(lr = lr, nbeta = nbeta, localization = gamma), num_chains = 1, num_draws = num_draws, device = DEVICE, online = False,) for model_checkpoint in all_checkpointed_models]

fig, ax1 = plt.subplots()
plt.title(F"Lambdahat vs acc for modular addition p = {params.p}, train_Frac={params.train_frac}, nb = {nbeta:.1f}, e = {lr}, y = {gamma}, num_draws = {num_draws}, num_chains = {num_chains}")

ax2 = ax1.twinx()
ax1.plot(df["val_acc"], label = "test acc")
ax1.plot(df["train_acc"], label = "train acc")
ax2.plot([llc["llc/mean"] for llc in llcs], color = "g", label = "Lambdahat")
ax1.set_xlabel("Checkpoint no.")
fig.legend(loc = "center right")

fig.show()

fig, ax1 = plt.subplots()
plt.title(f"Lambdahat vs loss for modular addition, p={params.p}, train_frac={params.train_frac}, nβ={nbeta:.1f}, ε={lr}, γ={gamma}, num_draws={num_draws}, num_chains={num_chains}")
ax2 = ax1.twinx()
ax1.plot(df["val_loss"], label = "test loss")
ax1.plot(df["train_loss"], label = "train loss")
ax2.plot([llc["llc/mean"] for llc in llcs], color = "g", label = "Lambdahat")
ax1.set_xlabel("Checkpoint no.")
fig.legend(loc = "center right")
plt.show()
