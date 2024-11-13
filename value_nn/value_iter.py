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

class MyDataset(Dataset):
    def __init__(self, k_cross=None, ashock=None, ishock=None, grid=None, dist=None):
        self.k_cross = k_cross
        self.ashock = ashock
        self.ishock = ishock
        self.grid = grid
        self.dist = dist
    
    def __len__(self):
        return self.next_k.size(0)

    def __getitem__(self, idx):
        return {
            'k_cross': self.k_cross[idx] if self.k_cross is not None else None,
            'ashock': self.ashock[idx] if self.ashock is not None else None,
            'ishock': self.ishock[idx] if self.ishock is not None else None,
            'grid': self.grid[idx],
            'dist': self.dist[idx]
        }


def value_fn(train_data, nn, params):
    gm_tmp = nn.gm_model(train_data["grid"].unsqueeze(-1))
    gm = torch.matmal(gm_tmp, train_data["dist"].unsqueeze(1)).squeeze(-1).squeeze(-1)
    state = torch.cat([train_data["next_k"], train_data["ashock"], train_data["ishock"], gm], dim=1)
    value = nn.value0(state)
    return value

def policy_fn(ashock, grid, dist, nn):
    gm_tmp = nn.gm_model(grid.unsqueeze(-1))
    gm = torch.matmal(gm_tmp, dist.unsqueeze(1)).squeeze(-1).squeeze(-1)
    state = torch.cat([ashock, gm], dim=1)
    next_k = nn.policy(state)
    return next_k

def price_fn(grid, dist, ashock, nn):
    gm_price_tmp = nn.gm_model_price(grid.unsqueeze(-1))
    gm_price = torch.matmal(gm_price_tmp, dist.unsqueeze(1)).squeeze(-1).squeeze(-1)
    state = torch.cat([ashock, gm_price], dim=1)
    price = nn.price_model(state)
    return price


def policy_iter(params, optimizer, nn, T):
    with torch.no_grad():
        data = get_dataset(params, T, nn)
    ashock = generate_ashock_values(T, params.ashock, params.pi_a)
    dataset = MyDataset(ashock=ashock, grid=data["grid"], dist=data["dist"])
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    for train_data in dataloader:
        next_v, _ = next_value(train_data, params, nn)
        loss.zero_grad()
        loss = -torch.mean(next_v)
        loss.backward()
        optimizer.step()

def value_iter(nn, params, optimizer, T, epochs):
    data = get_dataset(nn)
    ashock = generate_ashock_values(T, params.ashock, params.pi_a)
    ishock = generate_ashock_values(T, params.ishock, params.pi_i)
    k_cross = np.random.choice(params.k_grid, T)
    dataset = MyDataset(k_cross, ashock, ishock, data["grid"], data["dist"])
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    for train_data in dataloader:
        v = value_fn(train_data, nn, params)#value_fn書いて
        price = price_fn(train_data["grid"],train_data["dist"], train_data["ashock"], nn)#入力は分布とashockかな。
        wage = params.eta / price
        profit = get_profit(train_data["k_cross"], train_data["ashock"], price, params)
        v0_exp, v1_exp = next_value(train_data, params, nn)#ここ書いてgrid, gm, ashock, ishockの後ろ二つに関する期待値 v0_expなんかおかしい
        e0 = -params.gamma * policy_fn(train_data["ashock"], train_data["grid"], train_data["dist"], nn) * price + params.beta * v0_exp
        e1 = -(1-params.delta) * train_data["k_cross"]* price + params.beta * v1_exp
        threshold = (e0 - e1) / params.eta
        xi = min(params.B, max(0, threshold))
        vnew = profit - price*wage*xi**2/(2*params.B) + xi/params.B*e0 + (1-xi/params.B)*e1
        loss = torch.mean((vnew - v)**2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def get_profit(k_cross, ashock, ishock, price, params):
    wage = params.eta / price
    yterm = ashock * ishock * k_cross**params.theta
    n = (params.nu * yterm / wage)**(1 / (1 - params.nu))
    y = yterm * n**params.nu
    v0temp = y - wage * n + (1 - params.delta) * k_cross
    return v0temp*price

def dist_gm(grid, dist, ashock, nn):
    gm_tmp = nn.gm_model(grid)
    gm = torch.dot(gm_tmp, dist)
    state = torch.cat([ashock, gm], dim=1)
    next_gm = nn.next_gm_model(state)
    return next_gm


def next_value(train_data, nn, params, simul=False):
    next_gm = dist_gm(train_data["grid"], train_data["dist"], train_data["ashock"],nn)
    ashock = train_data["ashock"]
    ashock_idx = [torch.where(params.ashock == val)[0].item() for val in ashock]
    ashock_exp = params.pi_a[ashock_idx]
    ishock = train_data["ishock"]
    ishock_idx = [torch.where(params.ishock == val)[0].item() for val in ishock]
    ishock_exp = params.pi_i[ishock_idx]
    if simul:
        k_mesh, a_mesh, i_mesh = torch.meshgrid(train_data["grid"], params.ashock, params.ishock, indexing='ij')
        size = train_data["grid"].size()
    else:
        k_mesh, a_mesh, i_mesh = torch.meshgrid(train_data["k_cross"], ashock, ishock, indexing='ij')
        size = train_data["k_cross"].size()
    k_flat = k_mesh.flatten()
    a_flat = a_mesh.flatten()
    i_flat = i_mesh.flatten()
    
    pre_k_flat = (1-params.delta)*k_flat
    data_policy = torch.cat([ashock, next_gm], dim=1)
    next_k = policy_fn(ashock, train_data["dist"],nn).squeeze()
    next_k_flat = torch.full_like(k_flat, next_k)
    next_gm_flat = torch.full_like(a_flat, next_gm)
    
    data_e0 = torch.cat([next_k_flat, a_flat, i_flat, next_gm_flat], dim=1)
    data_e1 = torch.cat([pre_k_flat/params.gamma, a_flat, i_flat, next_gm_flat], dim=1)
    value_e0 = nn.value0(data_e0).squeeze(-1).reshape(size, ashock.size(), ishock.size())
    value_e1 = nn.value0(data_e1).squeeze(-1).reshape(size, ashock.size(), ishock.size())
    
    value_exp_e0_tmp = value_e0 * ashock_exp.unsqueeze(1) * ishock_exp.unsqueeze(0)
    value_exp_e1_tmp = value_e1 * ashock_exp.unsqueeze(1) * ishock_exp.unsqueeze(0)
    value_exp_e0 = torch.sum(value_exp_e0_tmp, dim=(1,2))
    value_exp_e1 = torch.sum(value_exp_e1_tmp, dim=(1,2))
    
    return value_exp_e0, value_exp_e1

def get_dataset(params, T, nn):
    dist_now = torch.full((params.k_grid.size,), 1.0 / params.k_grid.size, dtype=torch.float32)#砂川さんのやつだと5個でスタート
    k_now = torch.full_like(dist_now, params.kSS, dtype=TORCH_DTYPE)
    ashock = generate_ashock_values(T, params.ashock, params.pi_a)  # Should return a torch tensor
    ishock = generate_ashock_values(T, params.ishock, params.pi_i)  # Should return a torch tensor

    ashock = torch.tensor(ashock, dtype=TORCH_DTYPE)
    ishock = torch.tensor(ishock, dtype=TORCH_DTYPE)
    dist_history = []
    k_history = []
    # Define T based on the length of shock sequences
    for t in range(T):
        # Current shocks
        a = ashock[t]
        i = ishock[t]
        # Prepare the state dictionary
        basic_s = {
            "grid": k_now,  # Ensure params.k_grid is a torch tensor
            "ashock": a,
            "ishock": i,
            "dist": dist_now
        }
        # Compute next values using the provided next_value function
        next_value_e0, next_value_e1 = next_value(basic_s, nn, params, simul=True)
        
        # Compute xi and clamp its values between 0 and params.B
        xi = (next_value_e0 - next_value_e1) / params.eta
        xi = torch.clamp(xi, min=0.0, max=params.B)
        # Compute alpha
        alpha = xi / params.B
        
        indices_alpha_lt_1 = np.where(alpha < 1)[0]
        if len(indices_alpha_lt_1) == 0:
            J = -1
        else:
            J = indices_alpha_lt_1.max()
        dist_new = torch.zeros(J + 2)
        k_new = torch.zeros(J + 2)
        # Update the new distribution
        dist_new[0] = torch.dot(alpha, dist_now)
        dist_new[1:J+2] = (1 - alpha[:J+1]) * dist_now[:J+1]
        # Update the new capital grid
        k_new[0] = policy_fn(a, dist_now, nn).squeeze()
        k_new[1:J+2] = ((1 - params.delta) / params.gamma) * k_now[:J+1]
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
        "grid": k_history_tensor,
        "dist": dist_history_tensor
    }
        

def generate_ashock_values(T, shock, Pi):
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
    nz = len(shock)
    # 初期状態をランダムに選択（均等分布）
    states[0] = np.random.choice(nz)
    
    # 各時点での状態を遷移確率に従って選択
    for t in range(1, T):
        current_state = states[t-1]
        states[t] = np.random.choice(nz, p=Pi_normalized[current_state])
    
    # 状態に対応するashockの値を取得（長さTの配列）
    return shock[states]