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
        epoch += 1
        loss_list = []
        for data in train_loader:
            optimizer.zero_grad()
            loss = price_loss(nn, data)
            loss.backward()
            optimizer.step()
        for valid_data in valid_loader:
            loss = price_loss(nn, valid_data)
            loss_list.append(loss)
        avg_val_loss = sum(loss_list) / len(loss_list)
        print(f"epoch: {epoch}, avg_val_loss: {avg_val_loss}")


def next_gm_train(nn, params, optimizer, T):
    data = vi.get_dataset(params, T, nn)
    ashock = vi.generate_ashock_values(T, params.ashock, params.pi_a)
    k_grid = torch.tensor(data["k_grid"], dtype=TORCH_DTYPE)
    dist = torch.tensor(data["dist"], dtype=TORCH_DTYPE)
    ashock = torch.tensor(ashock, dtype=TORCH_DTYPE)
    next_gm_tmp = vi.next_gm_modeldata(["grid"][1:,:]).squeeze()
    next_gm = torch.sum(next_gm_tmp, dist[1:, :], dim=1)
    n_samples = len(ashock)
    dataset = CustomDataset(k_grid[:-2], dist[:-2], ashock[:-2], next_gm)
    valid_size = 192
    train_size = len(dataset) - valid_size
    train_data, valid_data = random_split(dataset, [train_size, valid_size])
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=64, shuffle=True)
    for data in train_loader:
        optimizer.zero_grad()
        loss = next_gm_loss(nn, data)
        loss.backward()
        optimizer.step()

def next_gm_loss(nn, data):
    loss_func = nn.MSELoss()
    next_gm = vi.dist_gm(data["grid"], data["dist"], data["ashock"], nn)
    loss = loss_func(next_gm, data["next_gm"])
    return loss


class CustomDataset(Dataset):
    def __init__(self, k_grid, dist, ashock, next_gm):
        # Ensure all tensors are the same length
        assert len(k_grid) == len(dist) == len(ashock) == len(next_gm), "All input tensors must have the same length"
        
        self.k_grid = k_grid
        self.dist = dist
        self.ashock = ashock
        self.next_gm = next_gm

    def __len__(self):
        return len(self.k_grid)

    def __getitem__(self, idx):
        sample = {
            'grid': self.k_grid[idx],
            'dist': self.dist[idx],
            'ashock': self.ashock[idx],
            'next_gm': self.next_gm[idx]
        }
        return sample
    
class Pricedatasets(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data["ashock"])
    def __getitem__(self, idx):
        return {"grid": self.data["grid"][idx], "dist": self.data["dist"][idx], "ashock": self.data["ashock"][idx]}
    