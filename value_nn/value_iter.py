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
    dist_now = torch.full((params.k_grid.size(0),), 1.0 / params.k_grid.size(0), dtype=TORCH_DTYPE)#砂川さんのやつだと5個でスタート
    k_now = torch.full_like(dist_now, params.kSS, dtype=TORCH_DTYPE)
    ashock = torch.zeros((params.k_grid.size(0), T))
    ashock[:, 0] = torch.tensor(np.random.choice(params.ashock), dtype=TORCH_DTYPE)
    a = ashock[:, 0]
    i = np.random.choice(params.ishock, size=(params.k_grid.size(0)))
    i = torch.tensor(i, dtype=TORCH_DTYPE)
    dist_history = []
    k_history = []
    # Define T based on the length of shock sequences
    for t in range(T):
        k_now_data = k_now.unsqueeze(0).repeat(k_now.size(0), 1)
        dist_now_data = dist_now.unsqueeze(0).repeat(k_now.size(0), 1)
        # Prepare the state dictionary
        basic_s = {
            "k_cross": k_now,#k_now
            "grid": k_now_data,  # k_now, k_now
            "ashock": a,#k_now
            "ishock": i,#k_now
            "dist": dist_now_data#k_now, k_now
        }
        # Compute next values using the provided next_value function
        e0, e1 = next_value(basic_s, nn, params)
        
        # Compute xi and clamp its values between 0 and params.B
        xi = ((e0 - e1) / params.eta).squeeze(-1)
        xi = torch.clamp(xi, min=0.0, max=params.B)
        # Compute alpha
        alpha = xi / params.B#k_now
        
        indices_alpha_lt_1 = torch.where(alpha < 1)
        if indices_alpha_lt_1[0].numel() == 0:  # 条件を満たす要素がない場合
            J = -1
        else:
            J = indices_alpha_lt_1[0].max().item()  # 最大値を取得（テンソルから整数に変換）

        dist_new = torch.zeros(J + 2)  # J + 2 行のゼロテンソルを作成

        k_new = torch.zeros(J + 2)
        # Update the new distribution
        dist_new[0] = torch.sum(alpha * dist_now)
        dist_new[1:J+2] = (1 - alpha[:J+1]) * dist_now[:J+1]
        # Update the new capital grid
        k_new[0] = policy_fn_vec(a[0:1], k_now, dist_now, nn).squeeze(-1)
        k_new[1:J+2] = ((1 - params.delta) / params.gamma) * k_now[:J+1]
        # Record the history by cloning to prevent in-place modifications
        dist_history.append(dist_now.clone())
        k_history.append(k_now.clone())
        # Update current distributions and capital grids for the next iteration
        dist_now = dist_new
        k_now = k_new
        ashock[:, t] = a
        
        a = next_ashock(a[0:1], params.ashock, params.pi_a).repeat(k_now.size(0))
        i = next_ishock(i, params.ishock, params.pi_i)
    if gm_train:
        return {
            "grid": k_history,
            "dist": dist_history,
            "ishock": ishock}
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

def next_ashock(current,shock, Pi):
    indices = torch.where(shock == current)[0]
    index = indices[0].item()
    row_sums = Pi.sum(axis=1)
    Pi_normalized = Pi / row_sums[:, np.newaxis]
    prob = Pi_normalized[index]
    state_index = torch.tensor([torch.multinomial(torch.tensor(prob), 1).item()])  # `multinomial` で選択
    return shock[state_index]

def next_ishock(current, shock, Pi):
    """
    個人の現在のショックに基づいて、次のショックを決定する関数。
    各行（個人）ごとに現在のショックから次のショックをサンプリング。

    Parameters:
        current (torch.Tensor): 各個人の現在のショック (1次元テンソル)
        shock (torch.Tensor): ショックの可能な値のテンソル (1次元テンソル)
        Pi (torch.Tensor): ショック遷移確率行列 (2次元テンソル)

    Returns:
        torch.Tensor: 各個人の次のショック (1次元テンソル)
    """
    # 次のショックを格納するテンソル
    next_shocks = torch.zeros_like(current)

    # 各個人の次のショックを決定
    for idx, cur in enumerate(current):
        # 現在のショックのインデックスを取得
        current_index = (shock == cur).nonzero(as_tuple=True)[0].item()
        
        # 現在のショックに対応する遷移確率を取得
        transition_prob = Pi[current_index]
        
        # 次のショックを遷移確率に基づいてサンプリング
        next_shocks[idx] = shock[torch.multinomial(transition_prob, num_samples=1).item()]

    return next_shocks