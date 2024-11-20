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
    price = vi.price_fn(data["grid"], data["dist"], data["ashock"][:, 0],nn)
    e0, e1 = next_value_gm(data, nn,params, data["grid"].size(1))#batch, max_cols,1
    threshold = (e0 - e1) / params.eta#batch, max_cols,1
    xi = torch.min(torch.tensor(params.B, dtype=TORCH_DTYPE), torch.max(torch.tensor(0, dtype=TORCH_DTYPE), threshold))#batch, max_cols,1
    alpha = (xi / params.B).squeeze(-1)#batch, max_cols,1
    k_next = vi.policy_fn(data["ashock"][:, 0], data["grid"], data["dist"], nn).repeat(1, alpha.size(1))#batch, max_cols
    inow = alpha * (params.gamma * k_next - (1-params.delta) * data["grid"])
    ashock = data["ashock"]
    ishock = data["ishock"]
    ynow = ashock*ishock * data["grid"]**params.theta * inow**params.nu
    Iagg = torch.sum(data["dist"] * inow, dim=1)
    Yagg = torch.sum(data["dist"]* ynow, dim=1)
    Cagg = Yagg - Iagg
    target = 1 / Cagg
    loss = torch.mean((price - target)**2)
    return loss

def price_train(params, nn, optimizer, num_epochs, num_sample, T, threshold):
    data = vi.get_dataset(params, T, nn, num_sample)#これめっちゃ長いなんで。gridが多いからだ。
    max_cols = max(array.numel() for array in data["grid"])
    ashock = vi.generate_ashock(1, T, params.ashock, params.pi_a).transpose(1, 0).repeat(1, max_cols)#T, max_cols
    ishock = vi.generate_ishock(max_cols,T, params.ishock, params.pi_i).transpose(1,0)#T, max_cols
    ashock = torch.tensor(ashock, dtype=TORCH_DTYPE)
    ishock = torch.tensor(ishock, dtype=TORCH_DTYPE)
    dataset = MyDataset(grid=data["grid"], dist=data["dist"], ashock=ashock, ishock=ishock)
    valid_size = 5
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
        
def gm_fn(grid, dist, nn):
    gm_tmp = nn.gm_model(grid.unsqueeze(-1))
    gm = torch.sum(gm_tmp * dist.unsqueeze(-1), dim=-2)
    return gm.squeeze(-1)

def next_gm_fn(gm, ashock, nn):
    state = torch.cat([ashock, gm], dim=1)
    next_gm = nn.next_gm_model(state)
    return next_gm

def next_value_gm(data, nn, params, max_cols):
    price = vi.price_fn(data["grid"], data["dist"], data["ashock"][:, 0], nn)#batch, 1
    next_gm = vi.dist_gm(data["grid"], data["dist"], data["ashock"][:,0],nn)#batch, 1
    ashock_ts = params.ashock
    ishock_ts = params.ishock
    ashock = data["ashock"]
    ashock_idx = torch.tensor([[torch.where(ashock_ts == val)[0].item() for val in row] for row in ashock])
    ashock_exp = params.pi_a[ashock_idx]#batch, max_cols, 5
    ishock = data["ishock"]
    ishock_idx = torch.tensor([[torch.where(ishock_ts == val)[0].item() for val in row] for row in ishock])
    ishock_exp = params.pi_i[ishock_idx]#batch, max_cols, 5
    prob = ashock_exp.unsqueeze(-1) * ishock_exp.unsqueeze(-2)#batch, max_cols, 5, 5
    
    next_k = vi.policy_fn(ashock[:,0], data["grid"], data["dist"], nn).repeat(1, max_cols).unsqueeze(-1)#batch, max_cols,1
    a_mesh, i_mesh = torch.meshgrid(ashock_ts, ishock_ts, indexing='ij')
    a_flat = a_mesh.flatten().unsqueeze(0).repeat_interleave(next_k.size(0), dim=0).unsqueeze(1).repeat(1, max_cols, 1)# batch,max_cols, i*a
    i_flat = i_mesh.flatten().unsqueeze(0).repeat_interleave(next_k.size(0), dim=0).unsqueeze(1).repeat(1, max_cols, 1)# batch,max_cols, i*a
    next_k_flat = next_k.repeat(1,1,a_flat.size(2))#batch, max_cols, i*a
    next_gm_flat = next_gm.repeat(1,max_cols).unsqueeze(-1).repeat(1,1,a_flat.size(2))#batch, max_cols, i*a
    k_cross_flat = data["grid"].unsqueeze(-1).repeat(1,1,a_flat.size(2))#batch, max_cols, i*a
    pre_k_flat = (1-params.delta) * k_cross_flat#batch, max_cols, i*a
    
    data_e0 = torch.stack([next_k_flat, a_flat, i_flat, next_gm_flat], dim=-1)#batch, max_cols, i*a, 4
    data_e1 = torch.stack([pre_k_flat, a_flat, i_flat, next_gm_flat], dim=-1)#batch, max_cols, i*a, 4
    value0 = nn.value0(data_e0).squeeze(-1)
    value1 = nn.value0(data_e1).squeeze(-1)
    value0 = value0.view(-1, max_cols, len(params.ashock), len(params.ishock))#batch, max_cols, a, i
    value1 = value1.view(-1, max_cols, len(params.ashock), len(params.ishock))#batch, max_cols, a, i
    
    expected_v0 = (value0 *  prob).sum(dim=(2,3)).unsqueeze(-1)#batch, max_cols, 1
    expected_v1 = (value1 *  prob).sum(dim=(2,3)).unsqueeze(-1)#batch, max_cols, 1
    
    e0 = -params.gamma * next_k + price.repeat(1, max_cols).unsqueeze(-1) + params.beta * expected_v0
    e1 = -params.gamma * (1-params.delta)*data["grid"].unsqueeze(-1) + price.repeat(1, max_cols).unsqueeze(-1) + params.beta * expected_v1
    
    return e0, e1
    
    
class NextGMDataset(Dataset):
    def __init__(self, gm, ashock):

        num_samples, T = gm.shape
        # 入力: gm[:, :-1], ターゲット: gm[:, 1:]
        inputs = gm[:, :-1].reshape(-1)  # shape: (num_samples * (T-1),)
        targets = gm[:, 1:].reshape(-1)  # shape: (num_samples * (T-1),)
        ashock_reshaped = ashock[:, :-1].reshape(-1)
        
        # 2列のテンソルに結合
        self.data = torch.stack((inputs, ashock_reshaped, targets), dim=1)  # shape: (num_samples * (T-1), 3)
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        """
        Args:
            idx (int): サンプルのインデックス
        Returns:
            tuple: (input, target) のタプル
        """
        input_val = self.data[idx, 0:1]    # shape: ()
        ashock_val = self.data[idx, 1:2]   # shape: ()
        target_val = self.data[idx, 2:3]   # shape: ()
        return input_val, ashock_val, target_val
    

def next_gm_train(nn, params, optimizer, T,num_sample):
    data = vi.get_dataset(params, T, nn, num_sample, gm_train=True)
    grid = [torch.tensor(grid, dtype=TORCH_DTYPE) for grid in data["grid"]]
    dist = [torch.tensor(dist, dtype=TORCH_DTYPE) for dist in data["dist"]]
    ashock = torch.tensor(data["ashock"], dtype=TORCH_DTYPE)
    dist = just_padding(data["dist"])
    grid = just_padding(data["grid"])
    gm = gm_fn(grid, dist, nn)
    dataset = NextGMDataset(gm, ashock)
    valid_size = 192
    train_size = len(dataset) - valid_size
    train_data, valid_data = random_split(dataset, [train_size, valid_size])
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=64, shuffle=True)
    loss_list = []
    for input, ashock_val, target in train_loader:
        optimizer.zero_grad()
        loss = next_gm_loss(nn, input, ashock_val, target)
        loss.backward()
        optimizer.step()
    for input, ashock_val, target in valid_loader:
        loss = next_gm_loss(nn, input, ashock_val, target)
        loss_list.append(loss)
    avg_val_loss = sum(loss_list) / len(loss_list)
    print(f"avg_val_loss: {avg_val_loss}")

def next_gm_loss(nn, input, ashock, target):
    next_gm = next_gm_fn(input, ashock, nn)
    loss = torch.mean((next_gm - target)**2)
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
    def __init__(self,k_cross=None, ashock=None, ishock=None, grid=None, dist=None):
        self.data = {}
        if k_cross is not None:
            if isinstance(k_cross, np.ndarray):
                k_cross = torch.tensor(k_cross, dtype=TORCH_DTYPE)
            k_cross = k_cross.view(-1, 1).squeeze(-1)
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
            padded = padding(grid)
            self.data['grid'] = padded#.repeat(num_sample, 1)
        if dist is not None:
            dist = [torch.tensor(data, dtype=TORCH_DTYPE) for data in dist]
            padded = padding(dist)
            self.data['dist'] = padded#.repeat(num_sample, 1)

    def __len__(self):
        # 使用しているデータの最初の項目の長さを返す
        return next(iter(self.data.values())).shape[0]

    def __getitem__(self, idx):
        # データが存在する場合のみ項目を返す
        return {key: value[idx] for key, value in self.data.items()}

def padding(list_of_arrays):
    max_cols = max(array.numel() for array in list_of_arrays)
    padded_arrays = []
    for array in list_of_arrays:
        padded_array = F.pad(array, (0, max_cols - array.numel()), mode='constant', value=0)
        padded_arrays.append(padded_array)
    data = torch.stack(padded_arrays, dim=0)
    return data
