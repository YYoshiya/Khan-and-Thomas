import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset, DataLoader, random_split

DTYPE = "float32"
if DTYPE == "float64":
    NP_DTYPE = np.float64
    TORCH_DTYPE = torch.float64  # PyTorchのデータ型を指定
elif DTYPE == "float32":
    NP_DTYPE = np.float32
    TORCH_DTYPE = torch.float32  # PyTorchのデータ型を指定
else:
    raise ValueError("Unknown dtype.")

class MyDictDataset(Dataset):
    def __init__(self, next_k, ashock, ishock, grid, dist):
        self.next_k = next_k
        self.ashock = ashock
        self.ishock = ishock
        self.grid = grid
        self.dist = dist
    
    def __len__(self):
        return self.next_k.size(0)

    def __getitem__(self, idx):
        return {
            'next_k': self.next_k[idx],          # 例: スカラーや1次元テンソル
            'ashock': self.ashock[idx],          # 例: スカラーや1次元テンソル
            'ishock': self.ishock[idx],          # 例: スカラーや1次元テンソル
            'grid': self.grid[idx],              # 例: 1次元テンソル（サイズ50）
            'dist': self.dist[idx]         # 例: 1次元テンソル（サイズ50）
        }


def value_fn(train_data, value0, policy, gm_model, params):
    gm_tmp = gm_model(train_data["grid"])
    gm = torch.dot(gm_tmp, train_data["dist"])
    state = torch.cat([train_data["next_k"], train_data["ashock"], train_data["ishock"], gm], dim=1)
    value = value0(state)
    return value

def policy_fn(ashock, grid, dist, policy, gm_model):
    gm_tmp = gm_model(grid)
    gm = torch.dot(gm_tmp, dist)
    state = torch.cat([ashock, gm], dim=1)
    next_k = policy(state)
    return next_k

def price_fn(grid, dist, ashock, gm_model_price, price_model):
    gm_price_tmp = gm_model_price(grid)
    gm_price = torch.dot(gm_price_tmp, dist)
    state = torch.cat([ashock, gm_price], dim=1)
    price = price_model(state)
    return price


def policy_iter(value0, policy, gm_model, params, optimizer):
    with torch.no_grad():
        data = get_dataset(policy, gm)
    dataset = MyDataset(data)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    for train_data in dataloader:
        next_v, _ = next_value(train_data, value0, policy, params)
        loss.zero_grad()
        loss = -torch.mean(next_v)
        loss.backward()
        optimizer.step()

def value_iter(value0, policy, gm_model, gm_model_price, price_model, params, optimizer, epochs):
    data = get_dataset(policy, gm)
    dataset = MyDataset(data)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    for train_data in dataloader:
        v = value_fn(train_data, value0, policy, gm_model, params)#value_fn書いて
        price = price_fn(train_data["grid"],train_data["dist"], train_data["ashock"], gm_model_price, price_model)#入力は分布とashockかな。
        #wage = params.eta / price
        profit = get_profit(train_data["k_grid"], train_data["ashock"], price, params)
        v0_exp, v1_exp = next_value(train_data, policy, params)#ここ書いてgrid, gm, ashock, ishockの後ろ二つに関する期待値 v0_expなんかおかしい
        e0 = -params.gamma * policy_fn(train_data["ashock"], train_data["grid"], train_data["dist"], policy, gm_model) * price + params.beta * v0_exp
        e1 = -(1-params.delta) * train_data["k_cross"]* price + params.beta * v1_exp
        threshold = (e0 - e1) / marams.eta
        xi = min(params.B, max(0, threshold))
        vnew = profit - p*w*xi**2/(2*params.B) + xi/params.B*e0 + (1-xi/params.B)*e1
        loss = torch.sum((vnew - v)**2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return value0


def get_profit(k_cross, ashock, ishock, price, params):
    wage = params.eta / price
    yterm = ashock * ishock * k_cross**params.theta
    n = (params.nu * yterm / wage)**(1 / (1 - params.nu))
    y = yterm * n**params.nu
    v0temp = y - wage * n + (1 - params.delta) * k_cross
    return v0temp*price

def dist_gm(grid, dist, ashock, gm_model, k_pred_model):
    gm_tmp = gm_model(grid)
    gm = torch.dot(gm_tmp, dist)
    state = torch.cat([ashock, gm], dim=1)
    k_pred = k_pred_model(state)
    return k_pred


def next_value(train_data, value0, policy, gm_model, k_pred_model, params):
    next_gm = dist_gm(train_data["grid"], train_data["dist"], train_data["ashock"], gm_model, k_pred_model)
    ashock = train_data["ashock"]
    ashock_idx = [torch.where(params.ashock == val)[0].item() for val in ashock]
    ashock_exp = params.pi_a[ashock_idx]
    ishock = train_data["ishock"]
    ishock_idx = [torch.where(params.ishock == val)[0].item() for val in ishock]
    ishock_exp = params.pi_i[ishock_idx]
    k_mesh, a_mesh, i_mesh = torch.meshgrid(train_data["k_grid"], params.ashock, params.ishock, indexing='ij')
    k_flat = k_mesh.flatten()
    a_flat = a_mesh.flatten()
    i_flat = i_mesh.flatten()
    
    pre_k_flat = (1-params.delta)*k_flat
    data_policy = torch.cat([ashock, next_gm], dim=1)
    next_k = policy_fn(ashock, train_data["dist"], policy, gm_model).squeeze()
    next_k_flat = torch.full_like(k_flat, next_k)
    next_gm_flat = torch.full_like(a_flat, next_gm)
    
    data_e0 = torch.cat([next_k_flat, a_flat, i_flat, next_gm_flat], dim=1)
    data_e1 = torch.cat([pre_k_flat/params.gamma, a_flat, i_flat, next_gm_flat], dim=1)
    value_e0 = value0(data_e0).squeeze(-1).reshape(train_data["k_grid"].size(), ashock.size(), ishock.size())
    value_e1 = value0(data_e1).squeeze(-1).reshape(train_data["k_grid"].size(), ashock.size(), ishock.size())
    
    value_exp_e0_tmp = value_e0 * ashock_exp.unsqueeze(1) * ishock_exp.unsqueeze(0)
    value_exp_e1_tmp = value_e1 * ashock_exp.unsqueeze(1) * ishock_exp.unsqueeze(0)
    value_exp_e0 = torch.sum(value_exp_tmp, dim=(1,2))
    value_exp_e1 = torch.sum(value_exp_tmp, dim=(1,2))
    
    return value_exp_e0, value_exp_e1

def get_dataset(policy, value0, gm_model, gm_model_price, price_model, params, T):
    dist_now = torch.full((params.k_grid.size,), 1.0 / params.k_grid.size, dtype=torch.float32)
    if isinstance(params.k_grid, torch.Tensor):
        k_now = params.k_grid.clone()
    else:
        k_now = torch.tensor(params.k_grid, dtype=torch.float32)
    ashock = generate_ashock_values(500, params.ashock, params.pi_a)  # Should return a torch tensor
    ishock = generate_ashock_values(500, params.ishock, params.pi_i)  # Should return a torch tensor

    ashock = torch.tensor(ashock, dtype=TORCH_DTYPE)
    ishock = torch.tensor(ishock, dtype=TORCH_DTYPE)
    dist_history = []
    k_history = []
    # Define T based on the length of shock sequences
    for t in range(T):
        # Initialize new distribution and capital grid with zeros
        dist_new = torch.zeros_like(dist_now)
        k_new = torch.zeros_like(k_now)
        # Current shocks
        a = ashock[t]
        i = ishock[t]
        # Prepare the state dictionary
        basic_s = {
            "k_grid": params.k_grid,  # Ensure params.k_grid is a torch tensor
            "ashock": a,
            "ishock": i,
            "dist": dist_now
        }
        # Compute next values using the provided next_value function
        next_value_e0, next_value_e1 = next_value(basic_s, value0, policy, gm_model, params)
        
        # Compute xi and clamp its values between 0 and params.B
        xi = (next_value_e0 - next_value_e1) / params.eta
        xi = torch.clamp(xi, min=0.0, max=params.B)
        # Compute alpha
        alpha = xi / params.B
        # Update the new distribution
        dist_new[0] = torch.dot(alpha, dist_now)
        dist_new[1:] = (1 - alpha[:-1]) * dist_now[:-1]
        # Update the new capital grid
        k_new[0] = policy_fn(a, dist_now, policy, gm_model).squeeze()
        k_new[1:] = ((1 - params.delta) / params.gamma) * k_now[:-1]
        # Record the history by cloning to prevent in-place modifications
        dist_history.append(dist_now.clone())
        k_history.append(k_now.clone())
        # Update current distributions and capital grids for the next iteration
        dist_now = dist_new
        k_now = k_new
    # Stack the history lists into tensors
    dist_history_tensor = torch.stack(dist_history)
    k_history_tensor = torch.stack(k_history)
    
    return {
        "k_grid": k_history_tensor,
        "dist": dist_history_tensor
    }
        

def generate_ashock_values(T, ashock, Pi):
    """
    指定された遷移確率行列Piに基づいてT個のashock値を生成します。
    
    Parameters:
    - T (int): 生成する値の個数
    - ashock (np.array): 状態に対応するashockの値
    - Pi (np.array): 遷移確率行列
    
    Returns:
    - np.array: 生成されたashockの値（長さTの配列）
    """
    
    row_sums = Pi.sum(axis=1)
    Pi_normalized = Pi / row_sums[:, np.newaxis]
    states = np.zeros(T, dtype=int)
    nz = len(ashock)
    # 初期状態をランダムに選択（均等分布）
    states[0] = np.random.choice(nz)
    
    # 各時点での状態を遷移確率に従って選択
    for t in range(1, T):
        current_state = states[t-1]
        states[t] = np.random.choice(nz, p=Pi_normalized[current_state])
    
    # 状態に対応するashockの値を取得（長さTの配列）
    return ashock[states]