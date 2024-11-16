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
from param import KTParam as params

DTYPE = "float32"
if DTYPE == "float64":
    NP_DTYPE = np.float64
    TORCH_DTYPE = torch.float64  # PyTorchのデータ型を指定
elif DTYPE == "float32":
    NP_DTYPE = np.float32
    TORCH_DTYPE = torch.float32  # PyTorchのデータ型を指定
else:
    raise ValueError("Unknown dtype.")

def price_loss(nn, data, params):#k_gridに関してxiを求める他は適当でよい。
    price = vi.price_fn(data["grid"], data["dist"], data["ashock"],nn)
    v0_exp, v1_exp = vi.next_value(data, nn,params, simul=True)#ここ書いてgrid, gm, ashock, ishockの後ろ二つに関する期待値 v0_expなんかおかしい
    k_next = vi.policy_fn(data["ashock"], data["grid"], data["dist"], nn)
    e0 = -params.gamma * k_next * price + params.beta * v0_exp
    e1 = -(1-params.delta) * data["grid"]* price + params.beta * v1_exp
    threshold = (e0 - e1) / params.eta
    xi = torch.min(torch.tensor(params.B, dtype=TORCH_DTYPE), torch.max(torch.tensor(0, dtype=TORCH_DTYPE), threshold))
    alpha = xi / params.B
    
    inow = alpha * (params.gamma * k_next - (1-params.delta) * data["grid"])#k_nextをかいて
    ashock = torch.repeat_interleave(data["ashock"].unsqueeze(-1), data["grid"].size(1), dim=1)
    ishock = torch.repeat_interleave(data["ishock"].unsqueeze(-1), data["grid"].size(1), dim=1)
    ynow = ashock*ishock * data["grid"]**params.theta * inow**params.nu
    Iagg = torch.sum(data["dist"] * inow, dim=1)
    Yagg = torch.sum(data["dist"]* ynow, dim=1)
    Cagg = Yagg - Iagg
    target = 1 / Cagg
    loss = torch.mean((price - target)**2)
    return loss

def price_train(params, nn, optimizer, num_epochs, n_sample, threshold):
    data = vi.get_dataset(params, 500, nn, num_sample=10)#これめっちゃ長いなんで。gridが多いからだ。
    ashock = vi.generate_ashock_values(10, 500, params.ashock, params.pi_a)
    ishock = vi.generate_ashock_values(10, 500, params.ishock, params.pi_i)
    ashock = torch.tensor(ashock, dtype=TORCH_DTYPE).view(-1, 1).squeeze(-1)
    ishock = torch.tensor(ishock, dtype=TORCH_DTYPE).view(-1, 1).squeeze(-1)
    dataset = MyDataset(grid=data["grid"], dist=data["dist"], ashock=ashock, ishock=ishock)
    valid_size = 192
    train_size = len(dataset) - valid_size
    train_data, valid_data = random_split(dataset, [train_size, valid_size])
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=64, shuffle=True)
    avg_val_loss = 100
    epoch = 0
    while avg_val_loss > threshold and epoch < num_epochs:
        epoch += 1
        loss_list = []
        for train_data in train_loader:
            optimizer.zero_grad()
            loss = price_loss(nn, train_data, params)
            loss.backward()
            optimizer.step()
        for valid_data in valid_loader:
            loss = price_loss(nn, valid_data, params)
            loss_list.append(loss)
        avg_val_loss = sum(loss_list) / len(loss_list)
        print(f"epoch: {epoch}, avg_val_loss: {avg_val_loss}")
        
def next_gm(grid, dist, ashock, nn):
    gm_tmp = nn.gm_model(grid.unsqueeze(-1))
    gm = torch.sum(gm_tmp * dist.unsqueeze(-1), dim=-2)
    state = torch.cat([ashock.unsqueeze(-1), gm], dim=2).reshape(-1, 2)
    next_gm = nn.next_gm_model(state)
    return gm.squeeze(-1), next_gm

class Nextgm_train(Dataset):
    def __init__(self, gm, next_gm):
        self.target = gm[:, 1:]
        self.next_gm = next_gm[:, :-2]

def next_gm_train(nn, params, optimizer, T,num_sample):
    data = vi.get_dataset(params, T, nn, num_sample, gm_train=True)
    grid = torch.tensor(data["grid"], dtype=TORCH_DTYPE)
    dist = torch.tensor(data["dist"], dtype=TORCH_DTYPE)
    ashock = tensor(data["ashock"], dtype=TORCH_DTYPE)
    dist = just_padding(data["dist"])
    grid = just_padding(data["grid"])
    next_gm_tmp = vi.next_gm(grid, dist, ashock, nn).reshape(num_sample, T)
    n_samples = len(ashock)
    dataset = MyDataset(k_grid[:-2], dist[:-2], ashock[:-2], next_gm)
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

def just_padding(list_of_arrays):
    list_of_arrays = [torch.tensor(data, dtype=TORCH_DTYPE) for data in list_of_arrays]
    max_cols = max(array.size(1) for array in list_of_arrays)
    padded_arrays = []
    for array in list_of_arrays:
        padded_array = F.pad(array, (0, max_cols - array.size(1)), mode='constant', value=0)
        padded_arrays.append(padded_array)
    data = torch.stack(padded_arrays, dim=0)
    data_reshaped = data.permute(1,0,2).contiguous()#num_sample, T, nの配列
    return data_reshaped

class MyDataset(Dataset):
    def __init__(self, k_cross=None, ashock=None, ishock=None, grid=None, dist=None):
        self.data = {}
        if k_cross is not None:
            if isinstance(k_cross, np.ndarray):
                k_cross = torch.tensor(k_cross, dtype=TORCH_DTYPE)
            k_cross = k_cross.view(-1, 1)
            self.data['k_cross'] = k_cross
        if ashock is not None:
            if isinstance(ashock, np.ndarray):
                ashock = torch.tensor(ashock, dtype=TORCH_DTYPE)
            #ashock = ashock.view(-1, 1)
            self.data['ashock'] = ashock
        if ishock is not None:
            if isinstance(ishock, np.ndarray):
                ishock = torch.tensor(ishock, dtype=TORCH_DTYPE)
            #ishock = ishock.view(-1, 1)
            self.data['ishock'] = ishock
        if grid is not None:
            grid = [torch.tensor(data, dtype=TORCH_DTYPE) for data in grid]
            self.data['grid'] = padding(grid)
        if dist is not None:
            dist = [torch.tensor(data, dtype=TORCH_DTYPE) for data in dist]
            self.data['dist'] = padding(dist)

    def __len__(self):
        # 使用しているデータの最初の項目の長さを返す
        return next(iter(self.data.values())).shape[0]

    def __getitem__(self, idx):
        # データが存在する場合のみ項目を返す
        return {key: value[idx] for key, value in self.data.items()}

def padding(list_of_arrays):
    max_cols = max(array.size(1) for array in list_of_arrays)
    padded_arrays = []
    for array in list_of_arrays:
        padded_array = F.pad(array, (0, max_cols - array.size(1)), mode='constant', value=0)
        padded_arrays.append(padded_array)
    data = torch.cat(padded_arrays, dim=0)
    return data

