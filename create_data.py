import random
import torch
import sys
import os

if len(sys.argv) != 5:
    raise ValueError("Please provide two arguments; p, train_split_proportion, nm_proportion and seed")

p = int(sys.argv[1])
train_split_proportion = float(sys.argv[2])
nm_proportion = float(sys.argv[3])
seed = int(sys.argv[4])

def deterministic_shuffle(lst, seed):
    random.shuffle(lst)
    return lst
    
def get_all_pairs(p):
    pairs = []
    for i in range(p):
        for j in range(p):
            pairs.append((i, j))
    return set(pairs)

#Split the data into two sets, one where we just add up the two values
#and one where we have to take the modulo
def make_dataset(p):
    m_data = []
    nm_data = []
    pairs = get_all_pairs(p)
    for a, b in pairs:
        if a + b < p:
           nm_data.append((torch.tensor([a, b]), torch.tensor((a + b) % p))) 
        else:
            m_data.append((torch.tensor([a, b]), torch.tensor((a + b) % p)))
    return m_data, nm_data

#Create train and test dataset such that we have a predefined mod/non-mod split in the training data.
#Meaning I want for example 70% of the training data to sum to values below p (non-modulo)
def train_test_split(m_data, nm_data, train_split_proportion, nm_proportion, seed):
    nm_length       = len(nm_data)
    m_length        = len(m_data)
    train_frac      = int((m_length + nm_length) * train_split_proportion)
    nm_train_frac   = int(train_frac * nm_proportion)
    
    nm_idx          = list(range(nm_length))
    nm_idx          = deterministic_shuffle(nm_idx, seed)
    m_idx           = list(range(m_length))
    m_idx           = deterministic_shuffle(m_idx, seed)

    nm_train_data   = nm_idx[:nm_train_frac]
    nm_test_data    = nm_idx[nm_train_frac:]
    
    m_train_data    = m_idx[:(train_frac - nm_train_frac)]
    m_test_data     = m_idx[(train_frac - nm_train_frac):]
    
    train_dataset   = [nm_data[i] for i in nm_train_data] + [m_data[i] for i in m_train_data]
    test_dataset    = [nm_data[i] for i in nm_test_data] + [m_data[i] for i in m_test_data]

    return deterministic_shuffle(train_dataset, seed), deterministic_shuffle(test_dataset, seed)

torch.manual_seed(seed)
random.seed(seed)

m_data, nm_data = make_dataset(p)
train_dataset, test_dataset = train_test_split(m_data, nm_data, train_split_proportion, nm_proportion, seed)

current_folder = os.getcwd()
#tps = is how train and test data is split; nmp is the proportion of modulo and non-modulo data in train set
if not os.path.exists(f"{current_folder}/Datasets/Train_p_{p}_tps_{train_split_proportion}_nmp_{nm_proportion}_seed_{seed}.pt"):
    torch.save(train_dataset, f"{current_folder}/Datasets/Train_p_{p}_tps_{train_split_proportion}_nmp_{nm_proportion}_seed_{seed}.pt")
if not os.path.exists(f"{current_folder}/Datasets/Test_p_{p}_tps_{train_split_proportion}_nmp_{nm_proportion}_seed_{seed}.pt"):
    torch.save(test_dataset, f"{current_folder}/Datasets/Test_p_{p}_tps_{train_split_proportion}_nmp_{nm_proportion}_seed_{seed}.pt")
test = torch.load(f"{current_folder}/Datasets/Train_p_{p}_tps_{train_split_proportion}_nmp_{nm_proportion}_seed_{seed}.pt", weights_only = True)
if test != None:
    print("Sucessfull")
