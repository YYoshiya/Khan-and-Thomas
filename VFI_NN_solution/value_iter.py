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
from param import params
import matplotlib.pyplot as plt
from datetime import datetime
import json

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
def move_models_to_device(nn, device):
    """
    指定されたニューラルネットワークの各モデルを指定されたデバイスに移動します。
    
    Parameters:
    - nn: ニューラルネットワークオブジェクト。各モデルは属性として持っている必要があります。
    - device: 移動先のデバイス。例："cpu" または "cuda"
    """
    nn.price_model.to(device)
    nn.gm_model.to(device)
    nn.value0.to(device)
    nn.gm_model_policy.to(device)
    nn.policy.to(device)
    nn.next_gm_model.to(device)
    nn.gm_model_price.to(device)
    nn.target_value.to(device)
    nn.target_gm_model.to(device)

class MyDataset(Dataset):
    def __init__(self, num_sample, k_cross=None, ashock=None, ishock=None, grid=None, dist=None, grid_k=None, dist_k=None):
        self.data = {}
        if k_cross is not None:
            if isinstance(k_cross, np.ndarray):
                k_cross = torch.tensor(k_cross, dtype=TORCH_DTYPE)
            k_cross = k_cross.view(-1, 1).squeeze(-1)
            self.data['k_cross'] = k_cross
        if ashock is not None:
            if isinstance(ashock, np.ndarray):
                ashock = torch.tensor(ashock, dtype=TORCH_DTYPE)
            self.data['ashock'] = ashock
        if ishock is not None:
            if isinstance(ishock, np.ndarray):
                ishock = torch.tensor(ishock, dtype=TORCH_DTYPE)
            self.data['ishock'] = ishock
        if grid is not None:
            grid = [
                torch.tensor(data, dtype=TORCH_DTYPE) if isinstance(data, np.ndarray) else data.clone().detach()
                for data in grid
            ]
            self.data['grid'] = torch.stack(grid, dim=0).repeat(num_sample, 1, 1)
            
        if dist is not None:
            dist = [
                torch.tensor(data, dtype=TORCH_DTYPE) if isinstance(data, np.ndarray) else data.clone().detach()
                for data in dist
            ]
            self.data['dist'] = torch.stack(dist, dim=0).repeat(num_sample, 1, 1)
        
        if grid_k is not None:
            grid_k = [
                torch.tensor(data, dtype=TORCH_DTYPE) if isinstance(data, np.ndarray) else data.clone().detach()
                for data in grid_k
            ]
            self.data['grid_k'] = torch.stack(grid_k, dim=0).repeat(num_sample, 1)
        
        if dist_k is not None:
            dist_k = [
                torch.tensor(data, dtype=TORCH_DTYPE) if isinstance(data, np.ndarray) else data.clone().detach()
                for data in dist_k
            ]
            self.data['dist_k'] = torch.stack(dist_k, dim=0).repeat(num_sample, 1)

    def __len__(self):
        # 使用しているデータの最初の項目の長さを返す
        return next(iter(self.data.values())).shape[0]

    def __getitem__(self, idx):
        # データが存在する場合のみ項目を返す
        return {key: value[idx] for key, value in self.data.items()}
class Valueinit(Dataset):
    def __init__(self, k_cross=None, ashock=None, ishock=None, K_cross=None, price=None, target_attr='k_cross', input_attrs=None):
        
        if k_cross is not None:
            if not isinstance(k_cross, torch.Tensor):
                k_cross = torch.tensor(k_cross, dtype=TORCH_DTYPE)
            self.k_cross = k_cross.view(-1, 1).squeeze(-1)

        if ashock is not None:
            if not isinstance(ashock, torch.Tensor):
                ashock = torch.tensor(ashock, dtype=TORCH_DTYPE)
            ashock_norm = (ashock - params.ashock_min) / (params.ashock_max - params.ashock_min)
            self.ashock = ashock_norm

        if ishock is not None:
            if not isinstance(ishock, torch.Tensor):
                ishock = torch.tensor(ishock, dtype=TORCH_DTYPE)
            ishock_norm = (ishock - params.ishock_min) / (params.ishock_max - params.ishock_min)
            self.ishock = ishock_norm

        if K_cross is not None:
            if not isinstance(K_cross, torch.Tensor):
                K_cross = torch.tensor(K_cross, dtype=TORCH_DTYPE)
            self.K_cross = K_cross.view(-1, 1).squeeze(-1)
        
        if price is not None:
            if not isinstance(price, torch.Tensor):
                price = torch.tensor(price, dtype=TORCH_DTYPE)
            price_norm = (price - params.price_min) / (params.price_max - params.price_min)
            self.price = price_norm.view(-1, 1).expand(self.K_cross.size(0), 1).squeeze(-1)

        # Validate target_attr and set it
        if target_attr not in ['k_cross', 'ashock', 'ishock', 'K_cross', 'price']:
            raise ValueError(f"Invalid target_attr: {target_attr}. Must be one of 'k_cross', 'ashock', 'ishock', 'K_cross', 'price'.")
        self.target_attr = target_attr

        # Set input attributes
        if input_attrs is None:
            # Default to using all attributes if not specified
            self.input_attrs = ['k_cross', 'ashock', 'ishock', 'K_cross', 'price']
        else:
            # Validate input attributes
            for attr in input_attrs:
                if attr not in ['k_cross', 'ashock', 'ishock', 'K_cross', 'price']:
                    raise ValueError(f"Invalid input attribute: {attr}. Must be one of 'k_cross', 'ashock', 'ishock', 'K_cross', 'price'.")
            self.input_attrs = input_attrs

    def __len__(self):
        # Find the first non-None attribute and return its length
        for attr in ['k_cross', 'ashock', 'ishock', 'K_cross', 'price']:
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


def soft_update(target, source, tau):
    """
    ターゲットネットワークのパラメータをメインネットワークのパラメータでソフトに更新します。
    
    Parameters:
        target (nn.Module): ターゲットネットワーク
        source (nn.Module): メインネットワーク
        tau (float): 更新割合
    """
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


def value_fn(train_data, nn, params):
    grid_norm = (train_data["grid_k"] - params.k_grid_min) / (params.k_grid_max - params.k_grid_min)
    ashock_norm = (train_data["ashock"] - params.ashock_min) / (params.ashock_max - params.ashock_min)
    ishock_norm = (train_data["ishock"] - params.ishock_min) / (params.ishock_max - params.ishock_min)
    gm_tmp = nn.gm_model(grid_norm.unsqueeze(-1))
    gm = torch.sum(gm_tmp * train_data["dist_k"].unsqueeze(-1), dim=-2)
    state = torch.cat([train_data["k_cross"].unsqueeze(-1), ashock_norm.unsqueeze(-1), ishock_norm.unsqueeze(-1), gm], dim=1)
    value = nn.value0(state)
    return value

def policy_fn(ashock, ishock,  grid, dist, price, nn):
    ashock_norm = (ashock - params.ashock_min) / (params.ashock_max - params.ashock_min)
    ishock_norm = (ishock - params.ishock_min) / (params.ishock_max - params.ishock_min)
    grid_norm = (grid - params.k_grid_min) / (params.k_grid_max - params.k_grid_min)
    gm_tmp = nn.gm_model_policy(grid_norm.unsqueeze(-1))
    gm = torch.sum(gm_tmp * dist.unsqueeze(-1), dim=-2)
    price_norm = (price - params.price_min) / (params.price_max - params.price_min)
    state = torch.cat([ashock_norm.unsqueeze(-1), ishock_norm.unsqueeze(-1), gm, price_norm], dim=1)#エラー出ると思う。
    output = nn.policy(state) * 8
    next_k = output
    return next_k

def policy_fn_sim(ashock, ishock, grid_k, dist_k, price, nn):
    ashock_norm = (ashock - params.ashock_min) / (params.ashock_max - params.ashock_min)
    ishock_norm = (ishock - params.ishock_min) / (params.ishock_max - params.ishock_min)
    grid_norm = (grid_k - params.k_grid_min) / (params.k_grid_max - params.k_grid_min)
    gm_tmp = nn.gm_model_policy(grid_norm.unsqueeze(-1))
    gm = torch.sum(gm_tmp * dist_k.unsqueeze(-1), dim=-2).expand(-1, ishock.size(1)).unsqueeze(-1)#batch, i, 1
    price_norm = (price - params.price_min) / (params.price_max - params.price_min)
    state = torch.cat([ashock_norm.unsqueeze(-1), ishock_norm.unsqueeze(-1), gm, price_norm.unsqueeze(-1)], dim=-1)
    output = nn.policy(state) * 8
    next_k = output
    return next_k

def price_fn(grid, dist, ashock, nn, mean=None):
    if mean is not None:
        mean = torch.sum(grid * dist, dim=-1)
        state = torch.stack([ashock, mean], dim=1)
    else:
        grid_norm = (grid - params.k_grid_min) / (params.k_grid_max - params.k_grid_min)
        ashock_norm = (ashock - params.ashock_min) / (params.ashock_max - params.ashock_min)
        gm_tmp = nn.gm_model_price(grid_norm)#batch, grid_size, i_size
        gm_price = torch.sum(gm_tmp * dist, dim=-2)#batch, i_size
        state = torch.cat([ashock_norm.unsqueeze(-1), gm_price], dim=1)#batch, i_size+1
    price = nn.price_model(state)#batch, 1
    return price

def golden_section_search_batch(
    train_data,
    price,
    nn,
    params,
    device,
    left_bound: float,
    right_bound: float,
    batch_size: int,
    max_iter: int = 20, 
    tol: float = 1e-5,
):
    """
    golden_section_search_batchの高速化版。
    next_e0のうち、k以外は一度だけ計算し、
    ループ内では next_k 依存の部分だけ計算するようにする。
    """

    # next_e0 のうち k 以外をキャッシュし、k だけ差し替える関数を作る
    next_e0_partial = init_next_e0(train_data, price, nn, params, device)

    # 黄金比
    phi = 0.618033988749895
    
    # 初期区間 [a, b]
    a = torch.full((batch_size,), left_bound, device=device, dtype=TORCH_DTYPE)
    b = torch.full((batch_size,), right_bound, device=device, dtype=TORCH_DTYPE)
    
    for _ in range(max_iter):
        dist_ = b - a
        c = b - phi * dist_
        d = a + phi * dist_

        fc = next_e0_partial(c)
        fd = next_e0_partial(d)

        mask = (fc > fd)
        b[mask] = d[mask]
        a[~mask] = c[~mask]

    x_star = 0.5 * (a + b)
    f_star = next_e0_partial(x_star)

    return x_star, f_star


def policy_iter_init2(params, optimizer, nn, T, num_sample, init_price):
    ashock_idx = torch.randint(0, len(params.ashock), (num_sample*T,))
    ishock_idx = torch.randint(0, len(params.ishock), (num_sample*T,))
    ashock = params.ashock[ashock_idx]
    ishock = params.ishock[ishock_idx]
    K_cross = np.random.choice(params.K_grid_np, num_sample* T)
    dataset = Valueinit(ashock=ashock,ishock=ishock, K_cross=K_cross, price=init_price ,target_attr='K_cross', input_attrs=['ashock', 'ishock', 'K_cross', 'price'])
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    count = 0
    for epoch in range(10):
        for train_data in dataloader:#policy_fnからnex_kを出してprice, gammaをかけて引く。
            count += 1
            train_data['X'] = train_data['X'].to(device, dtype=TORCH_DTYPE)
            next_k = nn.policy(train_data['X']).squeeze(-1) * 8
            target = torch.full_like(next_k, 2.5, dtype=TORCH_DTYPE).to(device)
            optimizer.zero_grad()
            loss = F.mse_loss(next_k, target)
            loss.backward()
            optimizer.step()
            if count % 100 == 0:
                print(f"count: {count}, loss: {loss.item()}")


# By implementing hard targetting, we might be able to accelerate the training process. But, not yet.
def value_iter(data, nn, params, optimizer, T, num_sample, p_init=None, mean=None):
    ashock_idx = torch.randint(0, len(params.ashock), (num_sample*T,))
    ishock_idx = torch.randint(0, len(params.ishock), (num_sample*T,))
    ashock = params.ashock[ashock_idx]
    ishock = params.ishock[ishock_idx]
    k_cross = np.random.choice(params.k_grid_tmp_lin, num_sample* T)
    dataset = MyDataset(num_sample, k_cross, ashock, ishock, data["grid"], data["dist"] ,data["grid_k"], data["dist_k"])
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    test_data = MyDataset(num_sample, k_cross, ashock, ishock, data["grid"], data["dist"] ,data["grid_k"], data["dist_k"])
    test_dataloader = DataLoader(test_data, batch_size=250, shuffle=True)
    countv = 0
    tau = 0.05
    for epoch in range(10):
        for train_data in dataloader:
            train_data = {key: value.to(device, dtype=TORCH_DTYPE) for key, value in train_data.items()}
            countv += 1
            with torch.no_grad():
                price = price_fn(train_data["grid"],train_data["dist"], train_data["ashock"], nn, mean=mean)
                if p_init is not None:
                    price = torch.full_like(price, p_init, dtype=TORCH_DTYPE).to(device)
                #入力は分布とashockかな。
                wage = params.eta / price
                profit = get_profit(train_data["k_cross"], train_data["ashock"], train_data["ishock"], price, params)
                _, e0 = golden_section_search_batch(train_data, price, nn, params, "cuda", params.k_grid_min, params.k_grid_max, batch_size=price.size(0))
                e1 = next_e1(train_data, price, nn, params, device)
                threshold = (e0 - e1) / params.eta
                #ここ見にくすぎる。
                xi = torch.min(torch.tensor(params.B, dtype=TORCH_DTYPE).to(device), torch.max(torch.tensor(0, dtype=TORCH_DTYPE).to(device), threshold))
                vnew = profit - (params.eta*xi**2)/(2*params.B) + (xi/params.B)*e0 + (1-(xi/params.B))*e1
            v = value_fn(train_data, nn, params).squeeze(-1)
            loss = F.mse_loss(v, vnew)
            optimizer.zero_grad()
            loss.backward()
            for param in nn.gm_model.parameters():
                if torch.isnan(param.grad).any():
                    print("NaN detected in gradients")
                    import sys
                    sys.exit("Training stopped due to NaN in gradients")

            optimizer.step()
            soft_update(nn.target_value, nn.value0, tau)
            soft_update(nn.target_gm_model, nn.gm_model, tau)
            if countv % 100 == 0:
                print(f"count: {countv}, loss: {loss.item()}")
    nn.target_value.load_state_dict(nn.value0.state_dict())
    nn.target_gm_model.load_state_dict(nn.gm_model.state_dict())
    with torch.no_grad():
        test_count = 0
        total_loss = 0.0
        min_loss = float('inf')  # 初期化: 最小値を非常に大きな値に設定
        max_loss = float('-inf') # 初期化: 最大値を非常に小さな値に設定
        for test_data in test_dataloader:
            test_count += 1
            test_data = {key: value.to(device, dtype=TORCH_DTYPE) for key, value in test_data.items()}
            price = price_fn(test_data["grid"], test_data["dist"], test_data["ashock"], nn, mean=mean)
            if p_init is not None:
                price = torch.full_like(price, p_init, dtype=TORCH_DTYPE).to(device)
            wage = params.eta / price
            profit = get_profit(test_data["k_cross"], test_data["ashock"], test_data["ishock"], price, params)
            _, e0 = golden_section_search_batch(test_data, price, nn, params, "cuda", params.k_grid_min, params.k_grid_max, batch_size=price.size(0))
            e1 = next_e1(test_data, price, nn, params, device)
            threshold = (e0 - e1) / params.eta
            xi = torch.min(torch.tensor(params.B, dtype=TORCH_DTYPE).to(device), torch.max(torch.tensor(0, dtype=TORCH_DTYPE).to(device), threshold))
            vnew = profit - (params.eta*xi**2)/(2*params.B) + (xi/params.B)*e0 + (1-(xi/params.B))*e1
            v = value_fn(test_data, nn, params).squeeze(-1)
            log_v = torch.log(v)
            log_vnew = torch.log(vnew)
            loss_test = torch.abs(log_v - log_vnew).max()
            loss_value = loss_test.item()
            total_loss += loss_value
            if loss_value < min_loss:
                min_loss = loss_value
            if loss_value > max_loss:
                max_loss = loss_value
        average_loss = total_loss / test_count if test_count > 0 else float('nan')

        print(f'Average Test Loss: {average_loss}, Min Loss: {min_loss}, Max Loss: {max_loss}')
    return average_loss, min_loss, max_loss

def value_init(nn, params, optimizer, T, num_sample):   
    ashock_idx = torch.randint(0, len(params.ashock), (num_sample*T,))
    ishock_idx = torch.randint(0, len(params.ishock), (num_sample*T,))
    ashock = params.ashock[ashock_idx]
    ishock = params.ishock[ishock_idx]
    k_cross = np.random.choice(params.k_grid_tmp, num_sample* T)
    K_cross = np.random.choice(params.K_grid_np, num_sample* T)
    dataset = Valueinit(k_cross, ashock, ishock, K_cross, target_attr="k_cross", input_attrs=["k_cross", "ashock", "ishock", "K_cross"])
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    countv = 0
    for epoch in range(10):
        for train_data in dataloader:
            countv += 1
            train_data['X'] = train_data['X'].to(device, dtype=TORCH_DTYPE)
            train_data['y'] = train_data['y'].to(device, dtype=TORCH_DTYPE)
            optimizer.zero_grad()
            v = nn.value0(train_data['X']).squeeze(-1)
            loss = F.mse_loss(v, 4*(train_data['y']**0.7))
            loss.backward()
            optimizer.step()
            if countv % 100 == 0:
                print(f"count: {countv}, loss: {loss.item()}")
    


def get_profit(k_cross, ashock, ishock, price, params):
    wage = params.eta / price.squeeze(-1)
    yterm = ashock * ishock * k_cross**params.theta
    n = (params.nu * yterm / wage)**(1 / (1 - params.nu))
    y = yterm * n**params.nu
    v0temp = y - wage * n + (1 - params.delta) * k_cross
    return v0temp*price.squeeze(-1)

def dist_gm(grid, dist, ashock, nn):
    grid_norm = (grid - params.k_grid_min) / (params.k_grid_max - params.k_grid_min)
    ashock_norm = (ashock - params.ashock_min) / (params.ashock_max - params.ashock_min)
    gm_tmp = nn.target_gm_model(grid_norm.unsqueeze(-1))
    gm = torch.sum(gm_tmp * dist.unsqueeze(-1), dim=-2)
    state = torch.cat([ashock_norm.unsqueeze(-1), gm], dim=1)
    next_gm = nn.next_gm_model(state)
    return next_gm

def generate_price(params, nn, price):
    # Clamp params.price_size to a maximum of 3
    clamped_price_size = min(params.price_size, 3)
    
    # Generate uniform random noise in the range [-clamped_price_size*0.1, clamped_price_size*0.1]
    noise = torch.empty_like(price, device=price.device).uniform_(
        -0.5, 
        clamped_price_size * 0.1
    )
    
    # Add the noise to the original price and return
    return price + noise
    

def init_next_e0(train_data, price, nn, params, device):
    """
    Precompute invariant parts of next_e0 that do not change on each call,
    and return a partial function that computes only the k-dependent part.

    Parameters
    ----------
    train_data : dict
        A dictionary of data (e.g., k_cross, ashock, ishock, dist_k, etc.)
    price : torch.Tensor
        Price tensor (shape: (batch_size,) or (batch_size, 1))
    nn : network object
    params : parameters object
    device : torch.device

    Returns
    -------
    next_e0_partial : function
        A function that takes only k as input and returns the value of next_e0
    """

    with torch.no_grad():
        # First, compute parts that do not depend on k only once, such as dist_gm.
        next_gm = dist_gm(train_data["grid_k"], train_data["dist_k"], train_data["ashock"], nn)
        
        # Compute necessary indices and probability distributions beforehand.
        ashock = train_data["ashock"]
        ishock = train_data["ishock"]

        # Obtain indices for each ashock and ishock (add .to(device) if needed for GPU).
        ashock_idx = [torch.where(params.ashock_gpu == val)[0].item() for val in ashock]
        ishock_idx = [torch.where(params.ishock_gpu == val)[0].item() for val in ishock]
        ashock_exp = params.pi_a_gpu[ashock_idx].unsqueeze(-1)  # For expectation calculation
        ishock_exp = params.pi_i_gpu[ishock_idx].unsqueeze(1)   # For expectation calculation
        probabilities = ashock_exp * ishock_exp  # (batch_size, a, i)

        # Create meshgrid covering all possible future (a, i) pairs and flatten them.
        a_mesh, i_mesh = torch.meshgrid(params.ashock_gpu, params.ishock_gpu, indexing='ij')
        a_mesh_norm = (a_mesh - params.ashock_min) / (params.ashock_max - params.ashock_min)
        i_mesh_norm = (i_mesh - params.ishock_min) / (params.ishock_max - params.ishock_min)
        a_flat = a_mesh_norm.flatten().view(1, -1, 1).expand(price.size(0), -1, -1)
        i_flat = i_mesh_norm.flatten().view(1, -1, 1).expand(price.size(0), -1, -1)

        # Expand next_gm to match the shape needed for computation.
        next_gm_flat = next_gm.expand(-1, a_flat.size(1)).unsqueeze(-1)  # (batch, a*i, 1)

        # Up to here, computations are independent of k.
        # Therefore, cache these computed values and define a partial function
        # that only substitutes k to compute e0.

    def next_e0_partial(k):
        """
        Function to be called each time k changes.
        It uses cached values computed in init_next_e0 and computes the k-dependent part of E0.
        """
        # Reshape k and then concatenate.
        k_expanded = k.view(-1, 1, 1).expand(-1, a_flat.size(1), -1)  # (batch, a*i, 1)
        
        # Create input for target_value.
        data_e0 = torch.cat([k_expanded, a_flat, i_flat, next_gm_flat], dim=2)  # (batch, a*i, 4)
        
        # Call the precomputed target_value (e.g., nn.target_value).
        value0 = nn.target_value(data_e0).squeeze(-1)
        # Reshape to (batch, nA, nI) to compute expectation.
        value0 = value0.view(price.size(0), len(params.ashock), len(params.ishock))
        
        # Compute expected value.
        expected_value0 = (value0 * probabilities).sum(dim=(1, 2))
        
        # Compute e0 = -k * price + β * E[V0]
        # (reshape price if necessary to match dimensions)
        e0 = -k * price.squeeze(-1) + params.beta * expected_value0
        return e0 # (batch,)
    
    return next_e0_partial


def next_e1(train_data, price, nn, params, device):
    with torch.no_grad():
        next_gm = dist_gm(train_data["grid_k"], train_data["dist_k"], train_data["ashock"], nn)
        ashock = train_data["ashock"]
        ashock_idx = [torch.where(params.ashock_gpu == val)[0].item() for val in ashock]
        ashock_exp = params.pi_a_gpu[ashock_idx].unsqueeze(-1)
        ishock = train_data["ishock"]
        ishock_idx = [torch.where(params.ishock_gpu == val)[0].item() for val in ishock]
        ishock_exp = params.pi_i_gpu[ishock_idx].unsqueeze(1)
        probabilities = ashock_exp * ishock_exp

    a_mesh, i_mesh = torch.meshgrid(params.ashock_gpu, params.ishock_gpu, indexing='ij')
    a_mesh_norm = (a_mesh - params.ashock_min) / (params.ashock_max - params.ashock_min)
    i_mesh_norm = (i_mesh - params.ishock_min) / (params.ishock_max - params.ishock_min)
    a_flat = a_mesh_norm.flatten().view(1, -1, 1).expand(train_data["k_cross"].size(0), -1, -1)# batch, i*a, 1
    i_flat = i_mesh_norm.flatten().view(1, -1, 1).expand(train_data["k_cross"].size(0), -1, -1)
    next_gm_flat = next_gm.expand(-1, a_flat.size(1)).unsqueeze(-1)#batch, i*a, 1
    k_cross_flat = train_data["k_cross"].view(-1,1,1).expand(-1, a_flat.size(1), -1)#batch, i*a, 1
    pre_k_flat = (1-params.delta)*k_cross_flat

    data_e1 = torch.cat([pre_k_flat, a_flat, i_flat, next_gm_flat], dim=2)
    value1 = nn.target_value(data_e1).squeeze(-1)
    value1 = value1.view(-1, len(params.ashock), len(params.ishock))  # (batch_size, a, i)
    expected_value1 = (value1 * probabilities).sum(dim=(1, 2)).unsqueeze(-1)  # (batch_size,)
    e1 = -(1-params.delta) * train_data["k_cross"].unsqueeze(-1) * price + params.beta * expected_value1
    return e1.squeeze(-1)
    
def next_value_sim(train_data, nn, params, p_init=None, mean=None):
    G = train_data["grid_k"].size(0)  # grid のサイズ
    i_size = params.ishock.size(0)  # i のサイズ
    price = price_fn(train_data["grid"], train_data["dist"], train_data["ashock"][:,0], nn, mean=mean)#G,1
    if p_init is not None:
        price = torch.full_like(price, p_init, dtype=TORCH_DTYPE)
    next_gm = dist_gm(train_data["grid_k"], train_data["dist_k"], train_data["ashock"][:,0],nn)#G,1
    ashock_idx = torch.where(params.ashock == train_data["ashock"][0, 0])[0].item()
    ashock_exp = params.pi_a[ashock_idx]
    prob = torch.einsum('ik,j->ijk', params.pi_i, ashock_exp).unsqueeze(0).expand(train_data["k_cross"].size(0), -1, -1, -1)
    

    next_k = policy_fn_sim(train_data["ashock"], train_data["ishock"], train_data["grid_k"], train_data["dist_k"], price.expand(-1, i_size), nn)#G, i_size, 1
    a_mesh, i_mesh = torch.meshgrid(params.ashock, params.ishock, indexing='ij')  # indexing='ij' を明示的に指定
    a_mesh_norm = (a_mesh - params.ashock_min) / (params.ashock_max - params.ashock_min)
    i_mesh_norm = (i_mesh - params.ishock_min) / (params.ishock_max - params.ishock_min)
    a_flat = a_mesh_norm.flatten()  # shape: [I*A]
    i_flat = i_mesh_norm.flatten()  # shape: [I*A]
    
    # a_flat と i_flat を [G, 5, I*A, 1] の形状に拡張
    a_4d = a_flat.view(1, 1, -1, 1).expand(G, 5, -1, 1)  # [G, 5, I*A, 1]
    i_4d = i_flat.view(1, 1, -1, 1).expand(G, 5, -1, 1)  # [G, 5, I*A, 1]
    
    # next_k を [G, 5, I*A, 1] の形状に効率的に変換
    # next_k の形状: [5, 1]
    # 1. 次元を追加して [1, 5, 1, 1] に変換
    # 2. expand で [G, 5, 25, 1] に拡張
    next_k_flat = next_k.expand(-1, -1, a_flat.size(0)).unsqueeze(-1)  # [G, 5, I*A, 1]
    next_gm_flat = next_gm.view(-1, 1, 1, 1).expand(G, i_size, a_flat.size(0), 1)  # [G, 5, I*A, 1]
    k_cross_flat = train_data["k_cross"].view(G, 1, 1, 1).expand(G, 5, a_flat.size(0), 1)  # [G, 5, I*A, 1]
    pre_k_flat = (1-params.delta) * k_cross_flat
    
    data_v0 = torch.cat([next_k_flat, a_4d, i_4d, next_gm_flat], dim=3)  # [G, 5, I*A, 4]
    data_v1 = torch.cat([pre_k_flat, a_4d, i_4d, next_gm_flat], dim=3)  # [G, 5, I*A, 4]
    value0 = nn.value0(data_v0).view(G, 5, params.ashock.size(0), params.ishock.size(0))  # [G, 5, A, I]
    value1 = nn.value0(data_v1).view(G, 5, params.ashock.size(0), params.ishock.size(0))  # [G, 5, A, I]
    
    expected_value0 = (value0 * prob).sum(dim=(2, 3))  # [G, 5]
    expected_value1 = (value1 * prob).sum(dim=(2, 3))  # [G, 5]
    
    
    e0 = -next_k.squeeze() * price.expand(G, i_size) + params.beta * expected_value0#G, i_size
    e1 = -(1-params.delta) * train_data["k_cross"].unsqueeze(1).expand(-1, i_size) * price.expand(G, i_size) + params.beta * expected_value1
    
    return e0, e1

def get_dataset(params, T, nn, p_init=None, mean=None, init_dist=None, last_dist=True):
    move_models_to_device(nn, "cpu")
    i_size = params.ishock.size(0)
    grid_size = params.grid_size

    # Initialize distribution over capital and idiosyncratic shocks
    if init_dist is not None:
        dist_now = nn.init_dist
        dist_now_k = nn.init_dist_k
    else:
        dist_now = torch.full((grid_size, i_size), 1.0 / (i_size * grid_size), dtype=params.pi_i.dtype)
        dist_now_k = torch.sum(dist_now, dim=1)  # Aggregate over idiosyncratic shocks
    k_now = params.k_grid  # (grid_size, nz)
    k_now_k = k_now[:, 0]  # Assuming aggregate shock is scalar for now

    # Initialize aggregate shock 'a'
    a_value = torch.randint(0, len(params.ashock), (1,))
    a = torch.full((grid_size, i_size), params.ashock[a_value].item(), dtype=params.pi_i.dtype)

    # Initialize histories
    dist_history = []
    k_history = []
    dist_k_history = []
    grid_k_history = []
    ashock_history = []
    mean_k_history = []
    price_diff_history = []

    # Initialize lists to store statistics each period
    i_over_k_level_history = []
    i_over_k_std_history = []
    inaction_history = []
    positive_spike_history = []
    negative_spike_history = []
    positive_inv_history = []
    negative_inv_history = []

    for t in range(T):
        grid_size = dist_now.size(0)
        dist_now_sum = dist_now.sum()
        # Prepare data for the policy functions
        basic_s = {
            "k_cross": k_now_k,  # Current capital grid (G,)
            "ashock": a,         # Current aggregate shock (G, I)
            "ishock": params.ishock.unsqueeze(0).expand(grid_size, -1),  # Idiosyncratic shocks (G, I)
            "grid": k_now.unsqueeze(0).repeat(grid_size, 1, 1),  
            "dist": dist_now.unsqueeze(0).repeat(grid_size, 1, 1),  
            "grid_k": k_now_k.unsqueeze(0).repeat(grid_size, 1),         # (G, G)
            "dist_k": dist_now_k.unsqueeze(0).repeat(grid_size, 1),      # (G, G)
        }

        # Compute expected values for adjustment decision
        e0, e1 = next_value_sim(basic_s, nn, params, p_init, mean)  # Returns (G, I) tensors
        xi_tmp = ((e0 - e1) / params.eta)  # Adjustment condition
        xi = torch.clamp(xi_tmp, min=0.0, max=params.B)
        alpha = xi / params.B  # Probability of adjustment (G, I)

        price = price_fn(basic_s["grid"], basic_s["dist"], basic_s["ashock"][:,0], nn, mean=mean)  # (G,1)
        if p_init is not None:
            price = torch.full_like(price, p_init, dtype=params.pi_i.dtype)
        # Policy function for adjusted capital
        k_prime_adj = policy_fn_sim(
            basic_s["ashock"], 
            basic_s["ishock"], 
            basic_s["grid_k"], 
            basic_s["dist_k"], 
            price.expand(-1, i_size), 
            nn
        )  # (G, I, 1)
        k_prime_adj = k_prime_adj.squeeze(-1)  # (G, I)

        # Capital for non-adjusting agents
        k_prime_non_adj = (1 - params.delta) * basic_s["k_cross"].unsqueeze(1).expand(-1, i_size)  # (G, I)

        new_price, diff = price_diff(basic_s, params, price, alpha, k_prime_adj)
        price_diff_history.append(diff)
        # Map k_prime to the capital grid using the refactored function
        idx_adj_lower, idx_adj_upper, weight_adj = map_to_grid(k_prime_adj, params.k_grid)
        idx_non_adj_lower, idx_non_adj_upper, weight_non_adj = map_to_grid(k_prime_non_adj, params.k_grid)

        # Initialize new distribution
        dist_new = torch.zeros_like(dist_now)

        update_distribution(
            dist_new, 
            dist_now, 
            alpha, 
            idx_adj_lower, 
            idx_adj_upper, 
            weight_adj, 
            params.pi_i, 
            adjusting=True
        )
        
        update_distribution(
            dist_new, 
            dist_now, 
            alpha, 
            idx_non_adj_lower, 
            idx_non_adj_upper, 
            weight_non_adj, 
            params.pi_i, 
            adjusting=False
        )
        
        ##### Obtain statistics #####
        i_over_k = (k_prime_adj - k_prime_non_adj) / basic_s["k_cross"].unsqueeze(1).expand(-1, i_size)
        i_over_k_alpha = i_over_k * dist_now * alpha
        i_over_k_level = torch.sum(i_over_k_alpha, dim=(0, 1)).item()
        i_over_k_std = torch.sqrt(torch.sum(i_over_k**2 * dist_now * alpha, dim=(0, 1))).item()
        inaction = torch.sum(dist_now * (1 - alpha)).item()
        positive_spike = torch.where(i_over_k > 0.2, dist_now*alpha, torch.zeros_like(dist_now)).sum().item()
        negative_spike = torch.where(i_over_k < -0.2, dist_now*alpha, torch.zeros_like(dist_now)).sum().item()
        positive_inv = torch.where(i_over_k > 0, dist_now*alpha, torch.zeros_like(dist_now)).sum().item()
        negative_inv = torch.where(i_over_k < 0, dist_now*alpha, torch.zeros_like(dist_now)).sum().item()
        ##### Obtain statistics #####

        # Append statistics to their respective lists
        i_over_k_level_history.append(i_over_k_level)
        i_over_k_std_history.append(i_over_k_std)
        inaction_history.append(inaction)
        positive_spike_history.append(positive_spike)
        negative_spike_history.append(negative_spike)
        positive_inv_history.append(positive_inv)
        negative_inv_history.append(negative_inv)

        dist_sum = dist_new.sum()
        # Normalize distribution to prevent numerical errors
        dist_new /= dist_sum

        # Update aggregate capital distribution
        dist_new_k = dist_new.sum(dim=1)  # Sum over idiosyncratic shocks
        k_new_k = params.k_grid[:, 0]

        next_a = next_ashock(a[0,0], params.ashock, params.pi_a)
        a_new = torch.full((grid_size, i_size), next_a.item(), dtype=TORCH_DTYPE)
        mean_k = torch.sum(k_now * dist_now, dim=(-1,-2))

        # Record history
        dist_history.append(dist_now.clone())
        k_history.append(k_now.clone())
        dist_k_history.append(dist_now_k.clone())
        grid_k_history.append(k_now_k.clone())
        ashock_history.append(a[0, 0].item())  # Record scalar 'a'
        mean_k_history.append(mean_k)

        # Update for the next iteration
        dist_now = dist_new
        k_now = k_now  # Capital grid remains the same
        dist_now_k = dist_new_k
        k_now_k = k_new_k
        a = a_new  # Update aggregate shock if necessary

    move_models_to_device(nn, device)
    if last_dist:
        nn.init_dist = dist_now
        nn.init_dist_k = dist_now_k

    ##### Calculate average statistics after period 500 #####
    start_period = 500
    if T > start_period:
        i_over_k_level_mean = sum(i_over_k_level_history[start_period:]) / (T - start_period)
        i_over_k_std_mean = sum(i_over_k_std_history[start_period:]) / (T - start_period)
        inaction_mean = sum(inaction_history[start_period:]) / (T - start_period)
        positive_spike_mean = sum(positive_spike_history[start_period:]) / (T - start_period)
        negative_spike_mean = sum(negative_spike_history[start_period:]) / (T - start_period)
        positive_inv_mean = sum(positive_inv_history[start_period:]) / (T - start_period)
        negative_inv_mean = sum(negative_inv_history[start_period:]) / (T - start_period)
    else:
        raise ValueError("T must be greater than 500.")

    # Compile average statistics into a dictionary
    mean_statistics = {
        "i_over_k_level_mean": i_over_k_level_mean,
        "i_over_k_std_mean": i_over_k_std_mean,
        "inaction_mean": inaction_mean,
        "positive_spike_mean": positive_spike_mean,
        "negative_spike_mean": negative_spike_mean,
        "positive_inv_mean": positive_inv_mean,
        "negative_inv_mean": negative_inv_mean,
    }

    # Directory structure setup
    results_dir = "results/simstats"
    current_datetime = datetime.now()
    date_str = current_datetime.strftime("%Y-%m-%d")
    time_str = current_datetime.strftime("%H_%M")

    # Path for the date-specific folder
    date_folder = os.path.join(results_dir, date_str)
    # Create the date folder if it does not exist
    os.makedirs(date_folder, exist_ok=True)

    # Generate the filename with current time
    filename = f"stats{time_str}.json"
    file_path = os.path.join(date_folder, filename)

    # Write the average statistics to the JSON file
    try:
        with open(file_path, "w") as f:
            json.dump(mean_statistics, f, indent=4)
        print(f"Average statistics have been saved to {file_path}.")
    except Exception as e:
        print(f"An error occurred while writing the file: {e}")

    ##### Exclude average statistics from the return value #####
    return {
        "grid": k_history[100:],         # From the 100th period onwards
        "dist": dist_history[100:],      # From the 100th period onwards
        "dist_k": dist_k_history[100:],  # From the 100th period onwards
        "grid_k": grid_k_history[100:],  # From the 100th period onwards
        "ashock": ashock_history[100:],  # From the 100th period onwards
        "mean_k": mean_k_history[100:],  # From the 100th period onwards
        # Average statistics are excluded from the return value
    }




def map_to_grid(k_prime, k_grid):
    """
    Map k_prime to the capital grid using linear interpolation.
    Returns lower indices, upper indices, and interpolation weights.

    Parameters:
    - k_prime: Tensor of new capital values (G, I)
    - k_grid: Capital grid (G, 1)

    Returns:
    - idx_lower: Lower indices in the grid (G, I)
    - idx_upper: Upper indices in the grid (G, I)
    - weight: Interpolation weights (G, I)
    """
    grid_size = k_grid.size(0)
    k_min = k_grid[0, 0]
    k_max = k_grid[-1, 0]

    # Flatten k_prime for searchsorted and then reshape back
    k_prime_flat = k_prime.reshape(-1)
    idx = torch.searchsorted(k_grid[:, 0], k_prime_flat).view(k_prime.shape)

    # Clamp indices to valid range
    idx = torch.clamp(idx, 0, grid_size - 1)

    # Adjust idx_lower and idx_upper
    idx_lower = torch.clamp(idx - 1, 0, grid_size - 1)
    idx_upper = idx

    k_lower = k_grid[idx_lower, 0]
    k_upper = k_grid[idx_upper, 0]

    # Compute weights, avoiding division by zero
    denom = k_upper - k_lower
    zero_denom_mask = denom.abs() < 1e-8
    denom = denom + zero_denom_mask * 1e-8  # Avoid division by zero

    weight = (k_prime - k_lower) / denom

    # Handle cases where k_prime is outside the grid
    weight = torch.where(k_prime <= k_min, torch.zeros_like(weight), weight)
    weight = torch.where(k_prime >= k_max, torch.ones_like(weight), weight)

    idx_lower = torch.where(k_prime <= k_min, torch.zeros_like(idx_lower), idx_lower)
    idx_upper = torch.where(k_prime <= k_min, torch.zeros_like(idx_upper), idx_upper)

    idx_lower = torch.where(k_prime >= k_max, (grid_size - 1) * torch.ones_like(idx_lower, dtype=torch.long), idx_lower)
    idx_upper = torch.where(k_prime >= k_max, (grid_size - 1) * torch.ones_like(idx_upper, dtype=torch.long), idx_upper)

    # Ensure indices are of integer type
    idx_lower = idx_lower.long()
    idx_upper = idx_upper.long()

    # Clamp weights to [0, 1]
    weight = torch.clamp(weight, 0.0, 1.0)

    return idx_lower, idx_upper, weight


def update_distribution(dist_new, dist_now, alpha, idx_lower, idx_upper, weight, pi_i, adjusting):
    G, I = dist_now.shape

    if adjusting:
        dist_adjust = dist_now * alpha  # (G, I)
    else:
        dist_adjust = dist_now * (1 - alpha)  # (G, I)

    for i in range(I):
        for i_prime in range(I):
            # Transition probability from state i to i_prime
            pi_ii = pi_i[i, i_prime]
            # Mass calculation
            dist_contrib = dist_adjust[:, i] * pi_ii  # (G,)
            # Allocation to lower grid point
            dist_new[:, i_prime].index_add_(0, idx_lower[:, i], dist_contrib * (1 - weight[:, i]))
            # Allocation to upper grid point
            dist_new[:, i_prime].index_add_(0, idx_upper[:, i], dist_contrib * weight[:, i])




    


def price_diff(data, params, price, alpha, next_k):
    wage = params.eta / price
    yterm = data["ashock"]* data["ishock"] * data["grid"][0,:,:]**params.theta
    n = (params.nu * yterm / wage)**(1 / (1 - params.nu))
    ynow = yterm * n**params.nu
    inow = alpha * (next_k - (1 - params.delta) * data["grid"][0,:,:])
    Iagg = torch.sum(inow * data["dist"][0,:,:])
    Yagg = torch.sum(ynow * data["dist"][0,:,:])
    Cagg = Yagg - Iagg
    price_new = 1/Cagg
    diff = torch.abs(price_new - price)
    return price_new, diff

def gm_diff(new_dist, new_a, data, params, nn):
    gm_pred = dist_gm(data["grid_k"], data["dist_k"], data["ashock"][:,0],nn)[0,0]
    grid_norm = (params.k_grid_tmp - params.k_grid_min) / (params.k_grid_max - params.k_grid_min)
    ashock_norm = (new_a - params.ashock_min) / (params.ashock_max - params.ashock_min)
    gm_tmp = nn.target_gm_model(grid_norm.unsqueeze(-1))
    diff = torch.abs(gm_tmp - gm_pred)
    return diff
    
    


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


def plot_mean_k(dataset, start_iteration, end_iteration, save_plot_dir='results/mean_k_sim'):
    """
    mean_k_history の指定範囲をプロットして保存する関数

    Parameters:
    - dataset (dict): get_dataset 関数から返される辞書
    - start_iteration (int): プロット開始の反復番号（例: 500）
    - end_iteration (int): プロット終了の反復番号（例: 600）
    - save_plot_dir (str): プロットを保存するディレクトリのパス（デフォルト: 'results/mean_k_sim'）
    """
    # プロット用ディレクトリの作成
    os.makedirs(save_plot_dir, exist_ok=True)
    
    mean_k_history = dataset.get("mean_k", None)
    
    if mean_k_history is None:
        raise ValueError("Dataset does not contain 'mean_k'. Ensure that get_dataset returns 'mean_k_history'.")
    
    total_iterations = len(mean_k_history) + 100  # get_datasetで100番目から返されているため
    if end_iteration > total_iterations:
        raise ValueError(f"end_iteration ({end_iteration}) exceeds the total available iterations ({total_iterations}).")
    
    
    adjusted_start = start_iteration - 100
    adjusted_end = end_iteration - 100
    
    if adjusted_start < 0 or adjusted_end > len(mean_k_history):
        raise ValueError("指定された範囲がデータセットの範囲を超えています。")
    
    mean_k_slice = mean_k_history[adjusted_start:adjusted_end]
    
    # テンソルをスカラー値に変換
    mean_k_values = [mk.detach().cpu().item() for mk in mean_k_slice]
    
    iterations = range(start_iteration, end_iteration)
    
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, mean_k_values, label='Mean Capital', color='blue')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Capital (mean_k)')
    plt.title(f'Mean Capital from Iteration {start_iteration} to {end_iteration}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
   
    plot_filename = f'mean_k_{start_iteration}_to_{end_iteration}.png'
    plot_path = os.path.join(save_plot_dir, plot_filename)
    plt.savefig(plot_path)
    plt.close()  
    
    print(f"Plot saved to {plot_path}")