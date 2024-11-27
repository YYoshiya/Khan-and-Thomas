import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import pred_train as pred

DTYPE = "float32"
if DTYPE == "float64":
    NP_DTYPE = np.float64
    TORCH_DTYPE = torch.float64  # PyTorchのデータ型を指定
elif DTYPE == "float32":
    NP_DTYPE = np.float32
    TORCH_DTYPE = torch.float32  # PyTorchのデータ型を指定
else:
    raise ValueError("Unknown dtype.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyDataset(Dataset):
    def __init__(self, num_sample, k_cross=None, ashock=None, ishock=None, grid=None, dist=None):
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
            self.data['grid'] = padded.repeat(num_sample, 1)
        if dist is not None:
            dist = [torch.tensor(data, dtype=TORCH_DTYPE) for data in dist]
            padded = padding(dist)
            self.data['dist'] = padded.repeat(num_sample, 1)

    def __len__(self):
        # 使用しているデータの最初の項目の長さを返す
        return next(iter(self.data.values())).shape[0]

    def __getitem__(self, idx):
        # データが存在する場合のみ項目を返す
        return {key: value[idx] for key, value in self.data.items()}
class Valueinit(Dataset):
    def __init__(self, k_cross=None, ashock=None, ishock=None, K_cross=None, target_attr='k_cross', input_attrs=None):
        
        if k_cross is not None:
            if not isinstance(k_cross, torch.Tensor):
                k_cross = torch.tensor(k_cross, dtype=TORCH_DTYPE)
            self.k_cross = k_cross.view(-1, 1).squeeze(-1)

        if ashock is not None:
            if not isinstance(ashock, torch.Tensor):
                ashock = torch.tensor(ashock, dtype=TORCH_DTYPE)
            self.ashock = ashock

        if ishock is not None:
            if not isinstance(ishock, torch.Tensor):
                ishock = torch.tensor(ishock, dtype=TORCH_DTYPE)
            self.ishock = ishock

        if K_cross is not None:
            if not isinstance(K_cross, torch.Tensor):
                K_cross = torch.tensor(K_cross, dtype=TORCH_DTYPE)
            self.K_cross = K_cross.view(-1, 1).squeeze(-1)

        # Validate target_attr and set it
        if target_attr not in ['k_cross', 'ashock', 'ishock', 'K_cross']:
            raise ValueError(f"Invalid target_attr: {target_attr}. Must be one of 'k_cross', 'ashock', 'ishock', 'K_cross'.")
        self.target_attr = target_attr

        # Set input attributes
        if input_attrs is None:
            # Default to using all attributes if not specified
            self.input_attrs = ['k_cross', 'ashock', 'ishock', 'K_cross']
        else:
            # Validate input attributes
            for attr in input_attrs:
                if attr not in ['k_cross', 'ashock', 'ishock', 'K_cross']:
                    raise ValueError(f"Invalid input attribute: {attr}. Must be one of 'k_cross', 'ashock', 'ishock', 'K_cross'.")
            self.input_attrs = input_attrs

    def __len__(self):
        # Find the first non-None attribute and return its length
        for attr in ['k_cross', 'ashock', 'ishock', 'K_cross']:
            data = getattr(self, attr, None)
            if data is not None:
                return len(data)
        raise ValueError("No valid data attributes were provided. Dataset length cannot be determined.")
    
    def __getitem__(self, idx):
        # Stack only the attributes specified in input_attrs
        inputs = [getattr(self, attr)[idx] for attr in self.input_attrs]
        X = torch.stack(inputs, dim=-1)
        y = getattr(self, self.target_attr)[idx]  # Use the attribute specified by target_attr
        return {'X': X, 'y': y}

        

def padding(list_of_arrays):
    max_cols = max(array.numel() for array in list_of_arrays)
    padded_arrays = []
    for array in list_of_arrays:
        padded_array = F.pad(array, (0, max_cols - array.numel()), mode='constant', value=0)
        padded_arrays.append(padded_array)
    data = torch.stack(padded_arrays, dim=0)
    return data

def value_fn(train_data, nn, params):
    gm_tmp = nn.gm_model(train_data["grid"].unsqueeze(-1))
    gm = torch.sum(gm_tmp * train_data["dist"].unsqueeze(-1), dim=-2)
    state = torch.cat([train_data["k_cross"].unsqueeze(-1), train_data["ashock"].unsqueeze(-1), train_data["ishock"].unsqueeze(-1), gm], dim=1)
    value = nn.value0(state)
    return value

def policy_fn(ashock, grid, dist, nn):
    gm_tmp = nn.gm_model_policy(grid.unsqueeze(-1))
    gm = torch.sum(gm_tmp * dist.unsqueeze(-1), dim=-2)
    state = torch.cat([ashock.unsqueeze(-1), gm], dim=1)
    next_k = nn.policy(state)
    return next_k

def policy_fn_sc(ashock, grid, dist, nn):
    gm_tmp = nn.gm_model_price(grid.unsqueeze(-1))
    gm = torch.sum(gm_tmp * dist.unsqueeze(-1), dim=-2)
    state = torch.cat([ashock.expand(gm.size(0), 1), gm], dim=1)
    next_k = nn.policy(state)
    return next_k

def price_fn(grid, dist, ashock, nn):
    gm_tmp = nn.gm_model_price(grid.unsqueeze(-1))
    gm_price = torch.sum(gm_tmp * dist.unsqueeze(-1), dim=-2)
    state = torch.cat([ashock.unsqueeze(-1), gm_price], dim=1)
    price = nn.price_model(state)
    return price


def price_fn_sc(grid, dist, ashock, nn):
    gm_tmp = nn.gm_model(grid.unsqueeze(-1))
    gm_price = torch.sum(gm_tmp * dist.unsqueeze(-1), dim=-2)
    state = torch.cat([ashock.expand(gm_price.size(0), 1), gm_price], dim=1)
    price = nn.price_model(state)
    return price
def policy_iter_init2(params, optimizer, nn, T, num_sample):
    ashock = generate_ashock(num_sample, T, params.ashock, params.pi_a).view(-1, 1).squeeze(-1)
    ishock = generate_ishock(num_sample, T, params.ishock, params.pi_i).view(-1, 1).squeeze(-1)
    K_cross = np.random.choice(params.k_grid_np, num_sample* T)
    dataset = Valueinit(ashock=ashock, K_cross=K_cross, target_attr='K_cross', input_attrs=['ashock', 'K_cross'])
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    count = 0
    for epoch in range(20):
        for train_data in dataloader:#policy_fnからnex_kを出してprice, gammaをかけて引く。
            count += 1
            train_data['X'] = train_data['X'].to(device, dtype=TORCH_DTYPE)
            next_k = nn.policy(train_data['X']).squeeze(-1)
            target = torch.full_like(next_k, 2, dtype=TORCH_DTYPE).to(device)
            optimizer.zero_grad()
            loss = F.mse_loss(next_k, target)
            loss.backward()
            optimizer.step()
            if count % 10 == 0:
                print(f"count: {count}, loss: {loss.item()}")


def policy_iter_init(params, optimizer, nn, T, num_sample):
    ashock = generate_ashock(num_sample, T, params.ashock, params.pi_a).view(-1, 1).squeeze(-1)
    ishock = generate_ishock(num_sample, T, params.ishock, params.pi_i).view(-1, 1).squeeze(-1)
    K_cross = np.random.choice(params.K_grid_np, num_sample* T)
    dataset = Valueinit(ashock=ashock, ishock=ishock, K_cross=K_cross, target_attr='K_cross', input_attrs=['ashock', 'ishock', 'K_cross'])
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    count = 0
    for epoch in range(10):
        for train_data in dataloader:#policy_fnからnex_kを出してprice, gammaをかけて引く。
            count += 1
            train_data['X'] = train_data['X'].to(device, dtype=TORCH_DTYPE)
            train_data['y'] = train_data['y'].to(device, dtype=TORCH_DTYPE)
            optimizer.zero_grad()
            e0 = next_value_init_policy(train_data['X'], nn, params, "cuda")
            loss = -torch.mean(e0)
            loss.backward()
            optimizer.step()
            if count % 10 == 0:
                print(f"count: {count}, loss: {-loss.item()}")


def policy_iter(data, params, optimizer, nn, T, num_sample, price=None):
    #with torch.no_grad():
        #data = get_dataset(params, T, nn, num_sample)
    if price is not None:
        price = 3.5
    ashock = generate_ashock(num_sample, T, params.ashock, params.pi_a)
    ashock = torch.tensor(ashock, dtype=TORCH_DTYPE).view(-1, 1).squeeze(-1)
    ishock = generate_ashock(num_sample, T, params.ishock, params.pi_i)
    ishock = torch.tensor(ishock, dtype=TORCH_DTYPE).view(-1, 1).squeeze(-1)
    k_cross = np.random.choice(params.k_grid_np, num_sample* T)
    dataset = MyDataset(num_sample, k_cross=k_cross, ashock=ashock, ishock=ishock, grid=data["grid"], dist=data["dist"])
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    countp = 0
    for epoch in range(10):
        for train_data in dataloader:#policy_fnからnex_kを出してprice, gammaをかけて引く。
            train_data = {key: value.to(device, dtype=TORCH_DTYPE) for key, value in train_data.items()}
            countp += 1
            optimizer.zero_grad()
            next_v, _ = next_value(train_data, nn, params, "cuda", p_init=price)
            loss = -torch.mean(next_v)
            loss.backward()
            optimizer.step()
            if countp % 10 == 0:
                print(f"count: {countp}, loss: {-loss.item()}")

def next_value_2(train_data, nn, params, device):
    next_gm = nn.next_gm_model(torch.cat(train_data[:,1:2], train_data[:,3:4], dim=1))
    price = torch.tensor(3.5, dtype=TORCH_DTYPE).unsqueeze(-1).to(device)
    ashock_ts = torch.tensor(params.ashock, dtype=TORCH_DTYPE).to(device)
    ishock_ts = torch.tensor(params.ishock, dtype=TORCH_DTYPE).to(device)
    ashock = train_data["ashock"]
    ashock_idx = [torch.where(ashock_ts == val)[0].item() for val in ashock]
    ashock_exp = torch.tensor(params.pi_a[ashock_idx], dtype=TORCH_DTYPE).unsqueeze(-1).to(device)
    ishock = train_data["ishock"]
    ishock_idx = [torch.where(ishock_ts == val)[0].item() for val in ishock]
    ishock_exp = torch.tensor(params.pi_i[ishock_idx], dtype=TORCH_DTYPE).unsqueeze(1).to(device)
    probabilities = ashock_exp * ishock_exp

    data = torch.cat([train_data[:, 0:1], train_data[:, 2:3]], dim=1)
    next_k = nn.policy(data)
    a_mesh, i_mesh = torch.meshgrid(ashock_ts, ishock_ts, indexing='ij')
    a_flat = a_mesh.flatten().unsqueeze(0).repeat_interleave(next_k.size(0), dim=0).unsqueeze(-1)# batch, i*a, 1
    i_flat = i_mesh.flatten().unsqueeze(0).repeat_interleave(next_k.size(0), dim=0).unsqueeze(-1)
    next_k_flat = next_k.repeat_interleave(a_flat.size(1), dim=1).unsqueeze(-1)#batch, i*a, 1
    next_gm_flat = next_gm.repeat_interleave(a_flat.size(1), dim=1).unsqueeze(-1)#batch, i*a, 1
    k_cross_flat = train_data[:, 0:1].repeat_interleave(a_flat.size(1), dim=1).unsqueeze(-1)#batch, i*a, 1
    pre_k_flat = (1-params.delta)*k_cross_flat
    
    data_e0 = torch.cat([next_k_flat, a_flat, i_flat, next_gm_flat], dim=2)
    data_e1 = torch.cat([pre_k_flat, a_flat, i_flat, next_gm_flat], dim=2)
    value0 = nn.value0(data_e0).squeeze(-1)
    value1 = nn.value0(data_e1).squeeze(-1)
    value0 = value0.view(-1, len(params.ashock), len(params.ishock))  # (batch_size, a, i) 
    value1 = value1.view(-1, len(params.ashock), len(params.ishock))  # (batch_size, a, i)
    
    # 確率と価値を掛けて期待値を計算
    expected_value0 = (value0 * probabilities).sum(dim=(1, 2)).unsqueeze(-1)  # (batch_size,)
    expected_value1 = (value1 * probabilities).sum(dim=(1, 2)).unsqueeze(-1)  # (batch_size,)
    
    e0 = -params.gamma * next_k * price + params.beta * expected_value0
    e1 = -(1-params.delta) * params.gamma * train_data[:, 0:1] * price + params.beta * expected_value1
    
def value_iter_2(nn, params, optimizer, T, num_sample):
    ashock = generate_ashock(num_sample, T, params.ashock, params.pi_a).view(-1, 1).squeeze(-1)
    ishock = generate_ishock(num_sample, T, params.ishock, params.pi_i).view(-1, 1).squeeze(-1)
    k_cross = np.random.choice(params.k_grid_np, num_sample* T)
    K_cross = np.random.choice(params.K_grid_np, num_sample* T)
    dataset = Valueinit(k_cross, ashock, ishock, K_cross, target_attr="k_cross")
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    for epoch in range(20):
        for train_data in dataloader:
            train_data['X'] = train_data['X'].to(device, dtype=TORCH_DTYPE)
            train_data['y'] = train_data['y'].to(device, dtype=TORCH_DTYPE)
            v = nn.value0(train_data['X'])
            price = torch.tensor(3.2, dtype=TORCH_DTYPE).unsqueeze(-1).to(device)
            wage = params.eta / price
            profit = get_profit(train_data[:, 0:1], train_data[:, 1:2], train_data[:, 2:3], price, params)
    
def value_iter(data, nn, params, optimizer, T, num_sample):
    #data = get_dataset(params, T, nn, num_sample)
    ashock = generate_ashock(num_sample,T, params.ashock, params.pi_a)
    ishock = generate_ishock(num_sample,T, params.ishock, params.pi_i)
    ashock = torch.tensor(ashock, dtype=TORCH_DTYPE).view(-1, 1).squeeze(-1)
    ishock = torch.tensor(ishock, dtype=TORCH_DTYPE).view(-1, 1).squeeze(-1)
    k_cross = np.random.choice(params.k_grid_np, num_sample* T)
    dataset = MyDataset(num_sample, k_cross, ashock, ishock, data["grid"], data["dist"])
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    countv = 0
    for epoch in range(10):
        for train_data in dataloader:
            train_data = {key: value.to(device, dtype=TORCH_DTYPE) for key, value in train_data.items()}
            countv += 1
            v = value_fn(train_data, nn, params)#value_fn書いて
            price = price_fn(train_data["grid"],train_data["dist"], train_data["ashock"], nn)#入力は分布とashockかな。
            wage = params.eta / price
            profit = get_profit(train_data["k_cross"], train_data["ashock"], train_data["ishock"], price, params).unsqueeze(-1)
            e0, e1 = next_value(train_data, nn, params, "cuda")#ここ書いてgrid, gm, ashock, ishockの後ろ二つに関する期待値 v0_expなんかおかしい
            threshold = (e0 - e1) / params.eta
            #ここ見にくすぎる。
            xi = torch.min(torch.tensor(params.B, dtype=TORCH_DTYPE).to(device), torch.max(torch.tensor(0, dtype=TORCH_DTYPE).to(device), threshold))
            vnew = profit - price*wage*xi**2/(2*params.B) + (xi/params.B)*e0 + (1-xi/params.B)*e1
            loss = F.mse_loss(v, vnew.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if countv % 10 == 0:
                print(f"count: {countv}, loss: {loss.item()}")

def value_init(nn, params, optimizer, T, num_sample):   
    ashock = generate_ashock(num_sample, T, params.ashock, params.pi_a).view(-1, 1).squeeze(-1)
    ishock = generate_ishock(num_sample, T, params.ishock, params.pi_i).view(-1, 1).squeeze(-1)
    k_cross = np.random.choice(params.k_grid_np, num_sample* T)
    K_cross = np.random.choice(params.K_grid_np, num_sample* T)
    dataset = Valueinit(k_cross, ashock, ishock, K_cross, target_attr="k_cross")
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    countv = 0
    for epoch in range(20):
        for train_data in dataloader:
            countv += 1
            train_data['X'] = train_data['X'].to(device, dtype=TORCH_DTYPE)
            train_data['y'] = train_data['y'].to(device, dtype=TORCH_DTYPE)
            optimizer.zero_grad()
            v = nn.value0(train_data['X']).squeeze(-1)
            loss = F.mse_loss(v, 4.2*train_data['y'])
            loss.backward()
            optimizer.step()
            if countv % 10 == 0:
                print(f"count: {countv}, loss: {loss.item()}")
    


def get_profit(k_cross, ashock, ishock, price, params):
    wage = params.eta / price.squeeze(-1)
    yterm = ashock * ishock * k_cross**params.theta
    n = (params.nu * yterm / wage)**(1 / (1 - params.nu))
    y = yterm * n**params.nu
    v0temp = y - wage * n + (1 - params.delta) * k_cross
    return v0temp*price.squeeze(-1)

def dist_gm(grid, dist, ashock, nn):
    gm_tmp = nn.gm_model(grid.unsqueeze(-1))
    gm = torch.sum(gm_tmp * dist.unsqueeze(-1), dim=-2)
    state = torch.cat([ashock.unsqueeze(-1), gm], dim=1)
    next_gm = nn.next_gm_model(state)
    return next_gm

def next_value_init_policy(train_data, nn, params, device):
    ashock_ts = torch.tensor(params.ashock, dtype=TORCH_DTYPE).to(device)
    ishock_ts = torch.tensor(params.ishock, dtype=TORCH_DTYPE).to(device)
    ashock = train_data[:,0]
    ashock_idx = [torch.where(ashock_ts == val)[0].item() for val in ashock]
    ashock_exp = torch.tensor(params.pi_a[ashock_idx], dtype=TORCH_DTYPE).unsqueeze(-1).to(device)
    ishock = train_data[:,1]
    ishock_idx = [torch.where(ishock_ts == val)[0].item() for val in ishock]
    ishock_exp = torch.tensor(params.pi_i[ishock_idx], dtype=TORCH_DTYPE).unsqueeze(1).to(device)
    probabilities = ashock_exp * ishock_exp
    next_gm = pred.next_gm_fn(train_data[:,2].unsqueeze(-1), train_data[:,0].unsqueeze(-1), nn)
    
    data = torch.cat([train_data[:, 0:1], train_data[:, 2:3]], dim=1)
    next_k = nn.policy(data)
    a_mesh, i_mesh = torch.meshgrid(ashock_ts, ishock_ts, indexing='ij')
    a_flat = a_mesh.flatten().unsqueeze(0).repeat_interleave(next_k.size(0), dim=0).unsqueeze(-1)# batch, i*a, 1
    i_flat = i_mesh.flatten().unsqueeze(0).repeat_interleave(next_k.size(0), dim=0).unsqueeze(-1)
    next_k_flat = next_k.repeat_interleave(a_flat.size(1), dim=1).unsqueeze(-1)#batch, i*a, 1
    next_gm_flat = next_gm.repeat_interleave(a_flat.size(1), dim=1).unsqueeze(-1)#batch, i*a, 1
    
    data_e0 = torch.cat([next_k_flat, a_flat, i_flat, next_gm_flat], dim=2)
    value0 = nn.value0(data_e0).squeeze(-1)
    value0 = value0.view(-1, len(params.ashock), len(params.ishock))  # (batch_size, a, i)
    expected_value0 = (value0 * probabilities).sum(dim=(1, 2)).unsqueeze(-1)  # (batch_size,)
    e0 = -params.gamma * next_k * 1 + params.beta * expected_value0
    return e0

def next_value(train_data, nn, params, device, grid=None, p_init=None):
    if p_init is not None:
        price = torch.tensor(p_init, dtype=TORCH_DTYPE).unsqueeze(0).unsqueeze(-1).repeat(train_data["ashock"].size(0), 1).to(device)
    else:
        price = price_fn(train_data["grid"], train_data["dist"], train_data["ashock"], nn)
    next_gm = dist_gm(train_data["grid"], train_data["dist"], train_data["ashock"],nn)
    ashock_ts = torch.tensor(params.ashock, dtype=TORCH_DTYPE).to(device)
    ishock_ts = torch.tensor(params.ishock, dtype=TORCH_DTYPE).to(device)
    ashock = train_data["ashock"]
    ashock_idx = [torch.where(ashock_ts == val)[0].item() for val in ashock]
    ashock_exp = torch.tensor(params.pi_a[ashock_idx], dtype=TORCH_DTYPE).unsqueeze(-1).to(device)
    ishock = train_data["ishock"]
    ishock_idx = [torch.where(ishock_ts == val)[0].item() for val in ishock]
    ishock_exp = torch.tensor(params.pi_i[ishock_idx], dtype=TORCH_DTYPE).unsqueeze(1).to(device)
    probabilities = ashock_exp * ishock_exp
    
    next_k = policy_fn(ashock, train_data["grid"], train_data["dist"], nn)#batch, 1
    a_mesh, i_mesh = torch.meshgrid(ashock_ts, ishock_ts, indexing='ij')
    a_flat = a_mesh.flatten().unsqueeze(0).repeat_interleave(next_k.size(0), dim=0).unsqueeze(-1)# batch, i*a, 1
    i_flat = i_mesh.flatten().unsqueeze(0).repeat_interleave(next_k.size(0), dim=0).unsqueeze(-1)
    next_k_flat = next_k.repeat_interleave(a_flat.size(1), dim=1).unsqueeze(-1)#batch, i*a, 1
    next_gm_flat = next_gm.repeat_interleave(a_flat.size(1), dim=1).unsqueeze(-1)#batch, i*a, 1
    k_cross_flat = train_data["k_cross"].unsqueeze(-1).repeat_interleave(a_flat.size(1), dim=1).unsqueeze(-1)#batch, i*a, 1
    pre_k_flat = (1-params.delta)*k_cross_flat
    k_check = train_data["k_cross"]*(1-params.delta)
    
    data_e0 = torch.cat([next_k_flat, a_flat, i_flat, next_gm_flat], dim=2)
    data_e1 = torch.cat([pre_k_flat, a_flat, i_flat, next_gm_flat], dim=2)
    value0 = nn.value0(data_e0).squeeze(-1)
    value1 = nn.value0(data_e1).squeeze(-1)
    value0 = value0.view(-1, len(params.ashock), len(params.ishock))  # (batch_size, a, i)
    value1 = value1.view(-1, len(params.ashock), len(params.ishock))  # (batch_size, a, i)

    # 確率と価値を掛けて期待値を計算
    expected_value0 = (value0 * probabilities).sum(dim=(1, 2)).unsqueeze(-1)  # (batch_size,)
    expected_value1 = (value1 * probabilities).sum(dim=(1, 2)).unsqueeze(-1)  # (batch_size,)
    
    e0 = -params.gamma * next_k * price + params.beta * expected_value0
    e1 = -(1-params.delta) * params.gamma * train_data["k_cross"].unsqueeze(-1) * price + params.beta * expected_value1
    
    return e0, e1
    

def get_dataset(params, T, nn, num_sample):
    device = "cpu"
    nn.price_model.to(device)
    nn.gm_model.to(device)
    nn.value0.to(device)
    nn.gm_model_policy.to(device)
    nn.policy.to(device)
    nn.next_gm_model.to(device)
    nn.gm_model_price.to(device)
    
    dist_now = torch.full((10,), 1.0 / 10, dtype=TORCH_DTYPE)
    k_now = torch.full_like(dist_now, params.kSS, dtype=TORCH_DTYPE)
    a = torch.tensor(np.random.choice(params.ashock), dtype=TORCH_DTYPE)  # Aggregate shock (scalar)
    a = a.repeat(k_now.size(0))
    i = torch.tensor(np.random.choice(params.ishock, size=(k_now.size(0),)), dtype=TORCH_DTYPE)
    dist_history = []
    k_history = []
    ashock_history = []
    ishock_history = []

    for t in range(T):
        k_now_data = k_now.unsqueeze(0).repeat(k_now.size(0), 1)
        dist_now_data = dist_now.unsqueeze(0).repeat(k_now.size(0), 1)
        basic_s = {
            "k_cross": k_now,
            "grid": k_now_data,
            "ashock": a,  # Aggregate shock (scalar)
            "ishock": i,
            "dist": dist_now_data
        }
        
        e0, e1 = next_value(basic_s, nn, params, "cpu")
        xi_tmp = ((e0 - e1) / params.eta).squeeze(-1)
        xi = torch.clamp(xi_tmp, min=0.0, max=params.B)
        alpha = xi / params.B
        index = torch.where(alpha < 1.0)[0]
        J = index.size(0)  # Number of elements satisfying the condition

        # Initialize new tensors
        dist_new = torch.zeros(J + 1, dtype=TORCH_DTYPE)
        k_new = torch.zeros(J + 1, dtype=TORCH_DTYPE)
        i_new = torch.zeros(J + 1, dtype=TORCH_DTYPE)
        a_new = torch.zeros(J + 1, dtype=TORCH_DTYPE)

        # Update the new distribution
        dist_new[0] = torch.sum(alpha * dist_now)

        if J > 0:
            dist_new[1:] = (1 - alpha[index]) * dist_now[index]
            k_new[1:] = ((1 - params.delta) / params.gamma) * k_now[index]
            i_new[1:] = next_ishock(i[index], params.ishock, params.pi_i)
        else:
            # If J = 0, these slices are empty, so we skip them
            pass

        # Update the capital grid
        k_new[0] = policy_fn(a, k_now_data, dist_now_data, nn).squeeze(-1)[0]  # 'a' is scalar

        # Update the individual shocks
        next_a = next_ashock(a[0], params.ashock, params.pi_a)
        i_new[0] = torch.tensor(np.random.choice(params.ashock), dtype=TORCH_DTYPE)
        a_new[:] = next_a

        # Record history
        dist_history.append(dist_now.clone())
        k_history.append(k_now.clone())
        ashock_history.append(a[0].clone())  # Record scalar 'a'
        ishock_history.append(i.clone())

        # Update for the next iteration
        dist_now = dist_new
        k_now = k_new
        i = i_new
        a = a_new

    device = "cuda"
    nn.price_model.to(device)
    nn.gm_model.to(device)
    nn.value0.to(device)
    nn.gm_model_policy.to(device)
    nn.policy.to(device)
    nn.next_gm_model.to(device)
    nn.gm_model_price.to(device)
    
    return {
            "grid": k_history,
            "dist": dist_history,
            "ashock": ashock_history,
            "ishock": ishock_history
        }
    
def generate_ishock(num_sample, T, shock, Pi):
    """
    PyTorch を使用して T 個の ishock 値を生成します。
    
    Parameters:
    - num_sample (int): サンプルの数
    - T (int): 各サンプルで生成する時点の数
    - shock (torch.Tensor): 状態に対応する ishock の値 (形状: (nz,))
    - Pi (torch.Tensor): 遷移確率行列 (形状: (nz, nz))
    
    Returns:
    - torch.Tensor: 生成された ishock の値 (形状: (num_sample, T))
    """
    # Pi がゼロ行を含まないようにする
    row_sums = Pi.sum(dim=1, keepdim=True)
    if torch.any(row_sums == 0):
        raise ValueError("Pi 行列の各行の合計がゼロになっている行があります。")
    
    # Pi を正規化
    Pi_normalized = Pi / row_sums
    
    # 浮動小数点誤差を修正して各行の合計が1になるように再正規化
    Pi_normalized = Pi_normalized / Pi_normalized.sum(dim=1, keepdim=True)
    
    # 状態の数
    nz = shock.size(0)
    
    # デバイスの取得
    device = Pi.device
    
    # 初期状態をランダムに選択（均等分布）
    initial_states = torch.randint(low=0, high=nz, size=(num_sample,), device=device)
    
    # 状態を格納するテンソルを初期化
    states = torch.zeros(num_sample, T, dtype=torch.long, device=device)
    states[:, 0] = initial_states
    
    # 各時点で状態をサンプリング
    for t in range(1, T):
        # 前時点の状態
        prev_states = states[:, t - 1]  # 形状: (num_sample,)
        
        # 前時点の状態に対応する確率分布を取得
        probs = Pi_normalized[prev_states]  # 形状: (num_sample, nz)
        
        # 各サンプルごとに次の状態をサンプリング
        next_states = torch.multinomial(probs, num_samples=1).squeeze(1)  # 形状: (num_sample,)
        
        # 現在の時点に次の状態を設定
        states[:, t] = next_states
    
    # 状態インデックスを対応する ishock 値にマッピング
    ishock_values = shock[states]  # 形状: (num_sample, T)
    
    return ishock_values


def generate_ashock(num_sample, T, shock, Pi):
    """
    PyTorch を使用して T 個の ashock 値を生成します。
    
    Parameters:
    - num_sample (int): サンプルの数
    - T (int): 各サンプルで生成する値の個数
    - shock (torch.Tensor): 状態に対応する ashock の値 (形状: (nz,))
    - Pi (torch.Tensor): 遷移確率行列 (形状: (nz, nz))
    
    Returns:
    - torch.Tensor: 生成された ashock の値 (形状: (num_sample, T))
    """
    # Pi がゼロ行を含まないようにする
    row_sums = Pi.sum(dim=1, keepdim=True)
    if torch.any(row_sums == 0):
        raise ValueError("Pi 行列の各行の合計がゼロになっている行があります。")

    # Pi を正規化
    Pi_normalized = Pi / row_sums

    # 確率の合計が厳密に 1 になるように再正規化（数値誤差の修正）
    Pi_normalized = Pi_normalized / Pi_normalized.sum(dim=1, keepdim=True)

    # 状態の数
    nz = shock.size(0)

    # 初期状態をランダムに選択（均等分布）
    states = torch.randint(low=0, high=nz, size=(num_sample, T), device=Pi.device)

    # 各サンプルの初期状態をランダムに設定
    states[:, 0] = torch.randint(low=0, high=nz, size=(num_sample,), device=Pi.device)

    for t in range(1, T):
        # 前時点の状態
        prev_states = states[:, t - 1]  # 形状: (num_sample,)

        # 前時点の状態に対応する確率分布を取得
        probs = Pi_normalized[prev_states]  # 形状: (num_sample, nz)

        # 各サンプルごとに次の状態をサンプリング
        next_states = torch.multinomial(probs, num_samples=1).squeeze(1)  # 形状: (num_sample,)

        # 現在の時点に次の状態を設定
        states[:, t] = next_states

    # 状態インデックスを対応する ashock 値にマッピング
    ashock_values = shock[states]  # 形状: (num_sample, T)

    return ashock_values

def next_ashock(current, shock, Pi):
    # currentがスカラーの場合もベクトルの場合も対応
    if current.dim() == 0:  # currentがスカラーの場合
        current = current.unsqueeze(0)

    next_shocks = []
    for cur in current:
        index = (shock == cur).nonzero(as_tuple=True)[0].item()
        row = Pi[index]
        next_index = torch.multinomial(torch.tensor(row, dtype=TORCH_DTYPE), 1).item()
        next_shocks.append(shock[next_index])
    
    return torch.tensor(next_shocks, dtype=current.dtype)



def next_ishock(current, shock, Pi):
    indices = torch.tensor([torch.where(shock == c)[0].item() for c in current])
    row_sums = Pi.sum(axis=1)
    Pi_normalized = Pi / row_sums[:, np.newaxis]
    probs = Pi_normalized[indices]
    probs_ts = torch.tensor(probs, dtype=TORCH_DTYPE)
    next_indices = torch.multinomial(probs_ts, 1).squeeze()
    next_shocks = shock[next_indices]
    return next_shocks