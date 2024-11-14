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

def compute_inner_product(grid, dist):
    # 次元数を確認して内積を取る
    if grid.dim() == 1 and dist.dim() == 1:
        # 両方が1次元の場合、通常の内積
        return torch.dot(grid, dist)
    elif grid.dim() == 2 and dist.dim() == 2:
        # 両方が2次元の場合、バッチごとに内積
        return torch.bmm(grid.unsqueeze(1), dist.unsqueeze(2)).squeeze()
    else:
        raise ValueError("grid と dist の次元が一致している必要があります")

def value_fn(train_data, nn, params):
    gm_tmp = nn.gm_model(train_data["grid"].unsqueeze(-1))
    gm = torch.sum(gm_tmp * train_data["dist"].unsqueeze(-1), dim=-2)
    state = torch.cat([train_data["k_cross"], train_data["ashock"].unsqueeze(-1), train_data["ishock"].unsqueeze(-1), gm], dim=1)
    value = nn.value0(state)
    return value

def policy_fn(ashock, grid, dist, nn):
    gm_tmp = nn.gm_model(grid.unsqueeze(-1))
    gm = torch.sum(gm_tmp * dist.unsqueeze(-1), dim=-2)
    state = torch.cat([ashock.unsqueeze(-1), gm], dim=1)
    next_k = nn.policy(state)
    return next_k

def price_fn(grid, dist, ashock, nn):
    gm_tmp = nn.gm_model(grid.unsqueeze(-1))
    gm_price = torch.sum(gm_tmp * dist.unsqueeze(-1), dim=-2)
    state = torch.cat([ashock.unsqueeze(-1), gm_price], dim=1)
    price = nn.price_model(state)
    return price


def policy_iter(params, optimizer, nn, T, num_sample):
    with torch.no_grad():
        data = get_dataset(params, T, nn, num_sample)
    ashock = generate_ashock_values(num_sample, T, params.ashock, params.pi_a)
    ishock = generate_ashock_values(num_sample, T, params.ishock, params.pi_i)
    dataset = MyDataset(ashock=ashock, ishock=ishock, grid=data["grid"], dist=data["dist"])
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    for train_data in dataloader:
        next_v, _ = next_value(train_data, params, nn)
        loss.zero_grad()
        loss = -torch.mean(next_v)
        loss.backward()
        optimizer.step()

def value_iter(nn, params, optimizer, T, num_sample):
    data = get_dataset(params, T, nn, num_sample)
    ashock = generate_ashock_values(num_sample,T, params.ashock, params.pi_a)
    ishock = generate_ashock_values(num_sample,T, params.ishock, params.pi_i)
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
    gm_tmp = nn.gm_model(grid.unsqueeze(-1))
    gm = torch.sum(gm_tmp * dist.unsqueeze(-1), dim=-2)
    state = torch.cat([ashock.unsqueeze(-1), gm], dim=1)
    next_gm = nn.next_gm_model(state)
    return next_gm


def next_value(train_data, nn, params, simul=False):
    next_gm = dist_gm(train_data["grid"], train_data["dist"], train_data["ashock"],nn)
    ashock_ts = torch.tensor(params.ashock, dtype=TORCH_DTYPE)
    ishock_ts = torch.tensor(params.ishock, dtype=TORCH_DTYPE)
    ashock = train_data["ashock"]
    ashock_idx = [torch.where(ashock_ts == val)[0].item() for val in ashock]
    ashock_exp = torch.tensor(params.pi_a[ashock_idx], dtype=TORCH_DTYPE)
    ishock = train_data["ishock"]
    ishock_idx = [torch.where(ishock_ts == val)[0].item() for val in ishock]
    ishock_exp = torch.tensor(params.pi_i[ishock_idx], dtype=TORCH_DTYPE)

    if simul:
        k_mesh, a_mesh, i_mesh = torch.meshgrid(train_data["grid"][0, :], ashock_ts, ishock_ts, indexing='ij')
        size = train_data["grid"].size(1)
        next_k = policy_fn(ashock, train_data["grid"], train_data["dist"],nn)
    else:
        k_mesh, a_mesh, i_mesh = torch.meshgrid(train_data["k_cross"][0,:], ashock, ishock, indexing='ij')
        size = train_data["k_cross"].size(1)
        next_k = policy_fn(ashock, train_data["k_cross"], train_data["dist"], nn)
    
    batch_size = ashock.size(0)
    len_ashock = len(ashock_ts)
    len_ishock = len(ishock_ts)
    k_flat = k_mesh.flatten().unsqueeze(0).repeat_interleave(next_k.size(0), dim=0)
    a_flat = a_mesh.flatten().unsqueeze(0).repeat_interleave(next_k.size(0), dim=0).unsqueeze(-1)
    i_flat = i_mesh.flatten().unsqueeze(0).repeat_interleave(next_k.size(0), dim=0).unsqueeze(-1)
    
    pre_k_flat = ((1-params.delta)*k_flat).unsqueeze(-1)
    next_k_flat = next_k.repeat_interleave(k_flat.size(1), dim=1).unsqueeze(-1)
    next_gm_flat = next_gm.repeat_interleave(k_flat.size(1), dim=1).unsqueeze(-1)
    
    data_e0 = torch.cat([next_k_flat, a_flat, i_flat, next_gm_flat], dim=2)
    data_e1 = torch.cat([pre_k_flat/params.gamma, a_flat, i_flat, next_gm_flat], dim=2)
    value_e0 = nn.value0(data_e0).squeeze(-1)
    value_e1 = nn.value0(data_e1).squeeze(-1)

    value_e0 = value_e0.view(batch_size, size, len_ashock, len_ishock)
    value_e1 = value_e1.view(batch_size, size, len_ashock, len_ishock)

    ashock_exp_unsq = ashock_exp.unsqueeze(1).unsqueeze(3)  # (batch_size, 1, len_ashock, 1)
    ishock_exp_unsq = ishock_exp.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, len_ishock)
    
    # 同時確率を計算し、形状を (batch_size, 1, len_ashock, len_ishock) に
    joint_probs = ashock_exp_unsq * ishock_exp_unsq  # (batch_size, 1, len_ashock, len_ishock)
    
    # joint_probs を (batch_size, size, len_ashock, len_ishock) に拡張
    joint_probs_expanded = joint_probs.expand(-1, size, -1, -1)
    
    # 重み付けと合計を実行し、期待値を計算
    expected_value_e0 = torch.sum(value_e0 * joint_probs_expanded, dim=(2, 3))  # (batch_size, size)
    expected_value_e1 = torch.sum(value_e1 * joint_probs_expanded, dim=(2, 3))  # (batch_size, size)
    
    return expected_value_e0, expected_value_e1

def get_dataset(params, T, nn, num_sample):
    dist_now = torch.full((num_sample, params.k_grid.size,), 1.0 / params.k_grid.size, dtype=torch.float32)#砂川さんのやつだと5個でスタート
    k_now = torch.full_like(dist_now, params.kSS, dtype=TORCH_DTYPE)
    ashock = generate_ashock_values(num_sample, T, params.ashock, params.pi_a)  # Should return a torch tensor
    ishock = generate_ashock_values(num_sample, T, params.ishock, params.pi_i)  # Should return a torch tensor

    ashock = torch.tensor(ashock, dtype=TORCH_DTYPE)
    ishock = torch.tensor(ishock, dtype=TORCH_DTYPE)
    dist_history = []
    k_history = []
    # Define T based on the length of shock sequences
    for t in range(T):
        # Current shocks
        a = ashock[:,t]
        i = ishock[:,t]
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
        dist_new = torch.zeros(num_sample, J + 2)
        k_new = torch.zeros(num_sample, J + 2)
        # Update the new distribution
        dist_new[:, 0] = torch.dot(alpha.flatten(), dist_now.flatten())
        dist_new[:, 1:J+2] = (1 - alpha[:,:J+1]) * dist_now[:, :J+1]
        # Update the new capital grid
        k_new[:, 0] = policy_fn(a, k_now, dist_now, nn).squeeze(-1)
        k_new[:, 1:J+2] = ((1 - params.delta) / params.gamma) * k_now[:, :J+1]
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
        

def generate_ashock_values(num_sample, T, shock, Pi):
    """
    指定された遷移確率行列 Pi に基づいて T 個の ashock 値を生成します。
    
    Parameters:
    - num_sample (int): サンプルの数
    - T (int): 各サンプルで生成する値の個数
    - shock (np.array): 状態に対応する ashock の値
    - Pi (np.array): 遷移確率行列
    
    Returns:
    - np.array: 生成された ashock の値（形状 (num_sample, T) の配列）
    """
    row_sums = Pi.sum(axis=1)
    Pi_normalized = Pi / row_sums[:, np.newaxis]
    states = np.zeros((num_sample, T), dtype=int)
    nz = len(shock)
    
    # 初期状態をランダムに選択（均等分布）
    states[:, 0] = np.random.choice(nz, size=num_sample)
    
    # 各時点での状態を遷移確率に従って選択
    for t in range(1, T):
        for i in range(num_sample):
            current_state = states[i, t - 1]
            states[i, t] = np.random.choice(nz, p=Pi_normalized[current_state])
    
    # 状態に対応する ashock の値を取得（形状 (num_sample, T) の配列）
    return shock[states]