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

def policy_fn_vec(ashock, grid, dist, nn):
    gm_tmp = nn.gm_model(grid.unsqueeze(-1))
    gm = torch.sum(gm_tmp * dist.unsqueeze(-1), dim=-2)
    state = torch.stack([ashock, gm], dim=1)
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
    ashock = torch.tensor(ashock, dtype=TORCH_DTYPE).view(-1, 1).squeeze(-1)
    ishock = generate_ashock_values(num_sample, T, params.ishock, params.pi_i)
    ishock = torch.tensor(ishock, dtype=TORCH_DTYPE).view(-1, 1).squeeze(-1)
    k_cross = np.random.choice(params.k_grid, num_sample* T)
    dataset = MyDataset(k_cross=k_cross, ashock=ashock, ishock=ishock, grid=data["grid"], dist=data["dist"])
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    for train_data in dataloader:#policy_fnからnex_kを出してprice, gammaをかけて引く。
        next_v, _ = next_value(train_data, nn, params)
        optimizer.zero_grad()
        loss = -torch.mean(next_v)
        loss.backward()
        optimizer.step()

def value_iter(nn, params, optimizer, T, num_sample):
    data = get_dataset(params, T, nn, num_sample)
    ashock = generate_ashock_values(num_sample,T, params.ashock, params.pi_a)
    ishock = generate_ashock_values(num_sample,T, params.ishock, params.pi_i)
    ashock = torch.tensor(ashock, dtype=TORCH_DTYPE).view(-1, 1).squeeze(-1)
    ishock = torch.tensor(ishock, dtype=TORCH_DTYPE).view(-1, 1).squeeze(-1)
    k_cross = np.random.choice(params.k_grid, num_sample* T)
    dataset = MyDataset(k_cross, ashock, ishock, data["grid"], data["dist"])
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    for train_data in dataloader:
        v = value_fn(train_data, nn, params)#value_fn書いて
        price = price_fn(train_data["grid"],train_data["dist"], train_data["ashock"], nn)#入力は分布とashockかな。
        wage = params.eta / price
        profit = get_profit(train_data["k_cross"], train_data["ashock"], train_data["ishock"], price, params)
        e0, e1 = next_value(train_data, nn, params)#ここ書いてgrid, gm, ashock, ishockの後ろ二つに関する期待値 v0_expなんかおかしい
        threshold = (e0 - e1) / params.eta
        #ここ見にくすぎる。
        xi = torch.min(torch.tensor(params.B, dtype=TORCH_DTYPE), torch.max(torch.tensor(0, dtype=TORCH_DTYPE), threshold))
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

def next_value(train_data, nn, params):
    price = price_fn(train_data["grid"], train_data["dist"], train_data["ashock"], nn)
    next_gm = dist_gm(train_data["grid"], train_data["dist"], train_data["ashock"],nn)
    ashock_ts = torch.tensor(params.ashock, dtype=TORCH_DTYPE)
    ishock_ts = torch.tensor(params.ishock, dtype=TORCH_DTYPE)
    ashock = train_data["ashock"]
    ashock_idx = [torch.where(ashock_ts == val)[0].item() for val in ashock]
    ashock_exp = torch.tensor(params.pi_a[ashock_idx], dtype=TORCH_DTYPE).unsqueeze(-1)
    ishock = train_data["ishock"]
    ishock_idx = [torch.where(ishock_ts == val)[0].item() for val in ishock]
    ishock_exp = torch.tensor(params.pi_i[ishock_idx], dtype=TORCH_DTYPE).unsqueeze(-1)
    probabilities = ashock_exp * ishock_exp
    
    next_k = policy_fn(ashock, train_data["grid"], train_data["dist"], nn)#batch, 1
    a_mesh, i_mesh = torch.meshgrid(ashock_ts, ishock_ts, indexing='ij')
    a_flat = a_mesh.flatten().unsqueeze(0).repeat_interleave(next_k.size(0), dim=0).unsqueeze(-1)# batch, i*a, 1
    i_flat = i_mesh.flatten().unsqueeze(0).repeat_interleave(next_k.size(0), dim=0).unsqueeze(-1)
    next_k_flat = next_k.repeat_interleave(a_flat.size(1), dim=1).unsqueeze(-1)#batch, i*a, 1
    next_gm_flat = next_gm.repeat_interleave(a_flat.size(1), dim=1).unsqueeze(-1)#batch, i*a, 1
    k_cross_flat = train_data["k_cross"].unsqueeze(-1).repeat_interleave(a_flat.size(1), dim=1).unsqueeze(-1)#batch, i*a, 1
    pre_k_flat = (1-params.delta)*k_cross_flat
    
    data_e0 = torch.cat([next_k_flat, a_flat, i_flat, next_gm_flat], dim=2)
    data_e1 = torch.cat([pre_k_flat/params.gamma, a_flat, i_flat, next_gm_flat], dim=2)
    value0 = nn.value0(data_e0).squeeze(-1)
    value1 = nn.value0(data_e1).squeeze(-1)
    value0 = value0.view(-1, len(params.ashock), len(params.ishock))  # (batch_size, a, i)
    value1 = value1.view(-1, len(params.ashock), len(params.ishock))  # (batch_size, a, i)

    # 確率と価値を掛けて期待値を計算
    expected_value0 = (value0 * probabilities).sum(dim=(1, 2)).unsqueeze(-1)  # (batch_size,)
    expected_value1 = (value1 * probabilities).sum(dim=(1, 2)).unsqueeze(-1)  # (batch_size,)
    
    e0 = -params.gamma * next_k * price + params.beta * expected_value0
    e1 = -train_data["k_cross"].unsqueeze(-1) * price + params.beta * expected_value1
    
    return e0, e1
    

def get_dataset(params, T, nn, num_sample, gm_train=False):
    dist_now = torch.full((params.k_grid.size(0),), 1.0 / params.k_grid.size(0), dtype=TORCH_DTYPE)
    k_now = torch.full_like(dist_now, params.kSS, dtype=TORCH_DTYPE)
    a = torch.tensor(np.random.choice(params.ashock), dtype=TORCH_DTYPE)  # 集計ショック（スカラー）
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
            "ashock": a,  # 集計ショック（スカラー）
            "ishock": i,
            "dist": dist_now_data
        }
        e0, e1 = next_value(basic_s, nn, params)
        xi = ((e0 - e1) / params.eta).squeeze(-1)
        xi = torch.clamp(xi, min=0.0, max=params.B)
        alpha = xi / params.B

        indices_alpha_lt_1 = torch.where(alpha < 1)
        if indices_alpha_lt_1[0].numel() == 0:
            J = -1
        else:
            J = indices_alpha_lt_1[0].max().item()

        dist_new = torch.zeros(J + 2, dtype=TORCH_DTYPE)
        k_new = torch.zeros(J + 2, dtype=TORCH_DTYPE)
        i_new = torch.zeros(J + 2, dtype=TORCH_DTYPE)

        # 新しい分布を更新
        dist_new[0] = torch.sum(alpha * dist_now)
        dist_new[1:J+2] = (1 - alpha[:J+1]) * dist_now[:J+1]

        # 新しい資本グリッドを更新
        k_new[0] = policy_fn_vec(a, k_now, dist_now, nn).squeeze(-1)  # 'a'はスカラー
        k_new[1:J+2] = ((1 - params.delta) / params.gamma) * k_now[:J+1]

        # 個人ショックを更新
        # 政策関数に従って移動するエージェント
        i_new[0] = next_ishock(i[0:1], params.ishock, params.pi_i)

        # 調整しないエージェント
        if J + 1 > 0:
            i_new[1:J+2] = next_ishock(i[:J+1], params.ishock, params.pi_i)

        # 履歴を記録
        dist_history.append(dist_now.clone())
        k_history.append(k_now.clone())
        ashock_history.append(a.clone())  # スカラーの'a'を記録
        ishock_history.append(i.clone())

        # 次のイテレーションのために現在の分布と資本グリッドを更新
        dist_now = dist_new
        k_now = k_new
        i = i_new

        # 次期の集計ショック'a'を更新
        a = next_ashock(a, params.ashock, params.pi_a)

    if gm_train:
        return {
            "grid": k_history,
            "dist": dist_history,
            "ashock": ashock_history,
            "ishock": ishock_history
        }
    else:
        return {
            "grid": k_history,
            "dist": dist_history
        }

        

def generate_ishock(num_sample, k_size, T, shock, Pi):
    """
    指定された遷移確率行列 Pi に基づいて T 個の ishock 値を生成します。
    
    Parameters:
    - num_sample (int): サンプルの数
    - k_size (int): 各サンプル内の状態の数
    - T (int): 各サンプルで生成する値の個数
    - shock (np.array): 状態に対応する ishock の値
    - Pi (np.array): 遷移確率行列
    
    Returns:
    - np.array: 生成された ishock の値（形状 (num_sample, k_size, T) の配列）
    """
    nz = len(shock)
    # 行方向に正規化
    Pi_normalized = Pi / Pi.sum(axis=1, keepdims=True)
    # 累積確率を計算
    Pi_cum = np.cumsum(Pi_normalized, axis=1)
    
    # 初期状態を均等分布からランダムに選択
    states = np.random.randint(low=0, high=nz, size=(num_sample, k_size))
    
    # ishock の配列を事前に用意
    ishock_values = np.empty((num_sample, k_size, T), dtype=shock.dtype)
    ishock_values[:, :, 0] = shock[states]
    
    for t in range(1, T):
        # 各現在の状態に対応する累積確率を取得
        current_cum_prob = Pi_cum[states]
        # 一様乱数を生成
        random_vals = np.random.rand(num_sample, k_size)
        # 次の状態を決定
        next_states = (random_vals[..., np.newaxis] < current_cum_prob).argmax(axis=2)
        states = next_states
        # ishock の値を割り当て
        ishock_values[:, :, t] = shock[states]
    
    return ishock_values

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

def next_ashock(current, shock, Pi):
    index = (shock == current).nonzero(as_tuple=True)[0].item()
    row = Pi[index]
    next_index = torch.multinomial(torch.tensor(row, dtype=TORCH_DTYPE), 1).item()
    return shock[next_index]


def next_ishock(current, shock, Pi):
    indices = torch.tensor([torch.where(shock == c)[0].item() for c in current])
    row_sums = Pi.sum(axis=1)
    Pi_normalized = Pi / row_sums[:, np.newaxis]
    probs = Pi_normalized[indices]
    probs_ts = torch.tensor(probs, dtype=TORCH_DTYPE)
    next_indices = torch.multinomial(probs_ts, 1).squeeze()
    next_shocks = shock[next_indices]
    return next_shocks