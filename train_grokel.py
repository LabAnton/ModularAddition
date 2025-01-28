import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class MLP(nn.Module):
    def __init__(self, params):
        self.embedding  = nn.Embedding(params.p, params.embed_dim)
        self.linear1r   = nn.Linear(params.embed_dim, params.hidden_size, bias = True)
        self.linear1l   = nn.Linear(params.embed_dim, params.hidden_size, bias = True)
        self.linear2    = nn.Linear(params.hidden_size, params.p, bias = False)
        self.act        = nn.GELU()
        self.vocab_size = params.p

    def forward(self, x):
        x1  = self.embedding(x[..., 0])
        x2  = self.embedding(x[..., 1])
        x1  = self.linear1l(x1)
        x2  = self.linear1r(x2)
        x   = x1 + x2
        x = self.act(x)
        x = self.linear2(x)
        return x

def test(model, dataset, device):
    n_correct   = 0
    total_loss  = 0
    model.eval()
    loss_fn     = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in dataset:
            x, y    = x.to(device), y.to(device)
            out     = model(x)
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
        out     = model(x)
        loss    = loss_fn(out, Y)
        loss.backward()
        optimizer.step()
        
        if checkpoint_every and (i + 1) % checkpoint_every == 0:
            #To implement: save models in the dic
            all_models += [deepcopy(model)]

        if (i + 1) % print_every = 0:
            val_acc, val_loss       = test(model, test_dataset, params.device)
            train_acc, train_loss   = test(model, train_dataset, params.device)
            loss_data.append({"batch": i+1, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc,})
            if verbose:
                pbar.set_postfix({{"train_loss": f"{train_loss:.4f}", "train_acc": f"{train_acc:.4f}", "val_loss": f"{val_loss:.4f}", "val_acc": f"{val_acc:.4f}",})
                pbar.update(print_every)

    if verbose:
        pbar.close()
    df = pd.DataFrame(loss_data)
    train_acc, train_loss   = test(model, train_dataset, params.device)
    val_acc, val_loss       = test(model, test_dataset, params.device)
    if verbose:
        print(f"Final Train Acc: {train_acc:.4f} | Final Train Loss: {train_loss:.4f}")
        print(f"Final Val Acc: {val_acc:.4f} | Final Val Loss: {val_loss:.4f}")
    return all_models, df

    
