import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import value_iter as vi
from param import params

DTYPE = "float32"
if DTYPE == "float64":
    NP_DTYPE = np.float64
    TORCH_DTYPE = torch.float64  # PyTorchのデータ型を指定
elif DTYPE == "float32":
    NP_DTYPE = np.float32
    TORCH_DTYPE = torch.float32  # PyTorchのデータ型を指定
else:
    raise ValueError("Unknown dtype.")

def price_loss(nn, data):#k_gridに関してxiを求める他は適当でよい。
    price = vi.price_fn(data["grid"], data["dist"], data["ashock"],nn)
    v0_exp, v1_exp = vi.next_value(data, nn,params)#ここ書いてgrid, gm, ashock, ishockの後ろ二つに関する期待値 v0_expなんかおかしい
    k_next = vi.policy_fn(data["ashock"], data["grid"], data["dist"], nn)
    e0 = -params.gamma * k_next * price + params.beta * v0_exp
    e1 = -(1-params.delta) * data["k_grid"]* price + params.beta * v1_exp
    threshold = (e0 - e1) / params.eta
    xi = min(params.B, max(0, threshold))
    alpha = xi / params.B
    
    inow = alpha * (params.gamma * k_next - (1-params.delta) * data["k"])#k_nextをかいて
    ynow = data["ashock"]*data["ishock"] * data["grid"]**params.theta * inow**params.nu
    Iagg = np.dot(data["dist"], inow)
    Yagg = np.dot(data["dist"], ynow)
    Cagg = Yagg - Iagg
    target = 1 / Cagg
    loss = torch.mean((price - target)**2)
    return loss

def price_train(nn, optimizer, num_epochs, n_sample, threshold):
    data = vi.get_dataset(params, 500, nn)
    ashock = vi.generate_ashock_values(10000, params.ashock, params.pi_a)
    k_grid = torch.tensor(data["k_grid"], dtype=TORCH_DTYPE)
    dist = torch.tensor(data["dist"], dtype=TORCH_DTYPE)
    ashock = torch.tensor(ashock, dtype=TORCH_DTYPE)
    n_samples = len(ashock)
    indices = torch.randperm(k_grid.size(0))[:n_sample]
    dataset_tmp = {"grid": k_grid[indices], "dist": dist[indices], "ashock": ashock}
    dataset = Pricedatasets(dataset_tmp)
    valid_size = 192
    train_size = len(dataset_tmp) - valid_size
    train_data, valid_data = random_split(dataset_tmp, [train_size, valid_size])
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=64, shuffle=True)
    
    while avg_val_loss > threshold and epoch < num_epochs:
        for data in train_loader:
            optimizer.zero_grad()
            loss = price_loss(nn, data)
            loss.backward()
            optimizer.step()


def next_gm_train(nn, params, ):
    data = vi.get_dataset(params, 2000, nn)
    ashock = vi.generate_ashock_values(2000, params.ashock, params.pi_a)
    k_grid = torch.tensor(data["k_grid"], dtype=TORCH_DTYPE)
    dist = torch.tensor(data["dist"], dtype=TORCH_DTYPE)
    ashock = torch.tensor(ashock, dtype=TORCH_DTYPE)
    n_samples = len(ashock)
    dataset_tmp = {"grid": k_grid, "dist": dist, "ashock": ashock}
    dataset = Pricedatasets(dataset_tmp)


class GMdatasets(Dataset):#ashockはget_datasetで生成したものを使う。
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data["ashock"])
    def __getitem__(self, idx):
        return {"grid": self.data["grid"][idx], "dist": self.data["dist"][idx], "ashock": self.data["ashock"][idx]}
    
class Pricedatasets(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data["ashock"])
    def __getitem__(self, idx):
        return {"grid": self.data["grid"][idx], "dist": self.data["dist"][idx], "ashock": self.data["ashock"][idx]}
    