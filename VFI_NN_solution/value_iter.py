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
            #ashock = ashock.view(-1, 1)
            self.data['ashock'] = ashock
        if ishock is not None:
            if isinstance(ishock, np.ndarray):
                ishock = torch.tensor(ishock, dtype=TORCH_DTYPE)
            #ishock = ishock.view(-1, 1)
            self.data['ishock'] = ishock
        if grid is not None:
            grid = [torch.tensor(data, dtype=TORCH_DTYPE) for data in grid]
            self.data['grid'] = torch.stack(grid, dim=0).repeat(num_sample, 1, 1)
            
        if dist is not None:
            dist = [torch.tensor(data, dtype=TORCH_DTYPE) for data in dist]
            self.data['dist'] = torch.stack(dist, dim=0).repeat(num_sample, 1, 1)
        
        if grid_k is not None:
            grid_k = [torch.tensor(data, dtype=TORCH_DTYPE) for data in grid_k]
            self.data['grid_k'] = torch.stack(grid_k, dim=0).repeat(num_sample, 1)
        
        if dist_k is not None:
            dist_k = [torch.tensor(data, dtype=TORCH_DTYPE) for data in dist_k]
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
            
            self.price = price.view(-1, 1).expand(self.K_cross.size(0), 1).squeeze(-1)

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
    state = torch.cat([ashock_norm.unsqueeze(-1), ishock_norm.unsqueeze(-1), gm, price], dim=1)#エラー出ると思う。
    output = nn.policy(state)
    next_k = 0.1 + 7.9 * output
    return next_k

def policy_fn_sim(ashock, ishock, grid_k, dist_k, price, nn):
    ashock_norm = (ashock - params.ashock_min) / (params.ashock_max - params.ashock_min)
    ishock_norm = (ishock - params.ishock_min) / (params.ishock_max - params.ishock_min)
    grid_norm = (grid_k - params.k_grid_min) / (params.k_grid_max - params.k_grid_min)
    gm_tmp = nn.gm_model_policy(grid_norm.unsqueeze(-1))
    gm = torch.sum(gm_tmp * dist_k.unsqueeze(-1), dim=-2).expand(-1, ishock.size(1)).unsqueeze(-1)#batch, i, 1
    state = torch.cat([ashock_norm.unsqueeze(-1), ishock_norm.unsqueeze(-1), gm, price.unsqueeze(-1)], dim=-1)
    output = nn.policy(state)
    next_k = 0.1 + 7.9 * output
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
            next_k = 0.1 + 7.9 * nn.policy(train_data['X']).squeeze(-1)
            target = torch.full_like(next_k, 2.5, dtype=TORCH_DTYPE).to(device)
            optimizer.zero_grad()
            loss = F.mse_loss(next_k, target)
            loss.backward()
            optimizer.step()
            if count % 100 == 0:
                print(f"count: {count}, loss: {loss.item()}")


def policy_iter(data, params, optimizer, nn, T, num_sample, p_init=None, mean=None):
    for param in nn.target_value.parameters():
        param.requires_grad = False
    for param in nn.target_gm_model.parameters():
        param.requires_grad = False
    ashock_idx = torch.randint(0, len(params.ashock), (num_sample*T,))
    ishock_idx = torch.randint(0, len(params.ishock), (num_sample*T,))
    ashock = params.ashock[ashock_idx]
    ishock = params.ishock[ishock_idx]
    k_cross = np.random.choice(params.k_grid_tmp_lin, num_sample* T)
    dataset = MyDataset(num_sample, k_cross=k_cross, ashock=ashock, ishock=ishock, grid=data["grid"], dist=data["dist"],grid_k=data["grid_k"], dist_k=data["dist_k"])
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    countp = 0
    for epoch in range(5):
        for train_data in dataloader:#policy_fnからnex_kを出してprice, gammaをかけて引く。
            train_data = {key: value.to(device, dtype=TORCH_DTYPE) for key, value in train_data.items()}
            countp += 1
            next_v, _, next_k = next_value(train_data, nn, params, device, p_init=p_init, mean=mean, policy_train=True)
            #loss_1 = torch.mean(F.relu((0.1 - next_k)*100))
            #loss_2 = torch.mean(F.relu((next_k - 8)*100))
            loss_p = torch.mean(-next_v)
            loss = loss_p# + loss_1 + loss_2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if countp % 100 == 0:
                print(f"count: {countp}, loss: {-loss.item()}, next_k_max: {next_k.max().item()}, next_k_min: {next_k.min().item()}")
    
    for param in nn.target_value.parameters():
        param.requires_grad = True
    for param in nn.target_gm_model.parameters():
        param.requires_grad = True
    return loss.item()

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
    test_dataloader = DataLoader(test_data, batch_size=32, shuffle=True)
    countv = 0
    tau = 0.05
    for epoch in range(20):
        for train_data in dataloader:
            train_data = {key: value.to(device, dtype=TORCH_DTYPE) for key, value in train_data.items()}
            countv += 1
            with torch.no_grad():
                price = price_fn(train_data["grid"],train_data["dist"], train_data["ashock"], nn, mean=mean)
                if p_init is not None:
                    price = torch.full_like(price, p_init, dtype=TORCH_DTYPE).to(device)
                #入力は分布とashockかな。
                wage = params.eta / price
                profit = get_profit(train_data["k_cross"], train_data["ashock"], train_data["ishock"], price, params).unsqueeze(-1)
                e0, e1, next_k = next_value(train_data, nn, params, device, p_init=p_init, mean=mean)#ここ書いてgrid, gm, ashock, ishockの後ろ二つに関する期待値 v0_expなんかおかしい
                threshold = (e0 - e1) / params.eta
                #ここ見にくすぎる。
                xi = torch.min(torch.tensor(params.B, dtype=TORCH_DTYPE).to(device), torch.max(torch.tensor(0, dtype=TORCH_DTYPE).to(device), threshold))
                vnew = profit - (params.eta*xi**2)/(2*params.B) + (xi/params.B)*e0 + (1-(xi/params.B))*e1
            v = value_fn(train_data, nn, params)
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
            profit = get_profit(test_data["k_cross"], test_data["ashock"], test_data["ishock"], price, params).unsqueeze(-1)
            e0, e1, next_k = next_value(test_data, nn, params, device, p_init=p_init, mean=mean)
            threshold = (e0 - e1) / params.eta
            xi = torch.min(torch.tensor(params.B, dtype=TORCH_DTYPE).to(device), torch.max(torch.tensor(0, dtype=TORCH_DTYPE).to(device), threshold))
            vnew = profit - (params.eta*xi**2)/(2*params.B) + (xi/params.B)*e0 + (1-(xi/params.B))*e1
            v = value_fn(test_data, nn, params)
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
    return average_loss

def value_init(nn, params, optimizer, T, num_sample):   
    ashock_idx = torch.randint(0, len(params.ashock), (num_sample*T,))
    ishock_idx = torch.randint(0, len(params.ishock), (num_sample*T,))
    ashock = params.ashock[ashock_idx]
    ishock = params.ishock[ishock_idx]
    k_cross = np.random.choice(params.k_grid_tmp_lin, num_sample* T)
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
            loss = F.mse_loss(v, 4*(train_data['y']**0.8))
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
    # priceと同じ形状・デバイス上で[-0.2, 0.2]の一様乱数を生成
    noise = torch.empty_like(price, device=price.device).uniform_(-0.4, 0.4)

    # 元のpriceに加算して返す (最後にunsqueeze(-1)で次元を増やす)
    return price + noise

    
    
    


def next_value(train_data, nn, params, device, grid=None, p_init=None, mean=None, policy_train=None):
    if p_init is not None:
        price = torch.tensor(p_init, dtype=TORCH_DTYPE).unsqueeze(0).unsqueeze(-1).repeat(train_data["ashock"].size(0), 1).to(device)
    else:
        price = price_fn(train_data["grid"], train_data["dist"], train_data["ashock"], nn, mean=mean)

    if policy_train is not None:
        price = generate_price(params, nn, price)
        
    with torch.no_grad():
        next_gm = dist_gm(train_data["grid_k"], train_data["dist_k"], train_data["ashock"], nn)
        ashock = train_data["ashock"]
        ashock_idx = [torch.where(params.ashock_gpu == val)[0].item() for val in ashock]
        ashock_exp = params.pi_a_gpu[ashock_idx].unsqueeze(-1)
        ishock = train_data["ishock"]
        ishock_idx = [torch.where(params.ishock_gpu == val)[0].item() for val in ishock]
        ishock_exp = params.pi_i_gpu[ishock_idx].unsqueeze(1)
        probabilities = ashock_exp * ishock_exp
    
    next_k = policy_fn(ashock, ishock, train_data["grid_k"], train_data["dist_k"], price, nn)#batch, 
    a_mesh, i_mesh = torch.meshgrid(params.ashock_gpu, params.ishock_gpu, indexing='ij')
    a_mesh_norm = (a_mesh - params.ashock_min) / (params.ashock_max - params.ashock_min)
    i_mesh_norm = (i_mesh - params.ishock_min) / (params.ishock_max - params.ishock_min)
    a_flat = a_mesh_norm.flatten().unsqueeze(0).repeat_interleave(next_k.size(0), dim=0).unsqueeze(-1)# batch, i*a, 1
    i_flat = i_mesh_norm.flatten().unsqueeze(0).repeat_interleave(next_k.size(0), dim=0).unsqueeze(-1)
    next_k_flat = next_k.repeat_interleave(a_flat.size(1), dim=1).unsqueeze(-1)#batch, i*a, 1
    next_gm_flat = next_gm.repeat_interleave(a_flat.size(1), dim=1).unsqueeze(-1)#batch, i*a, 1
    k_cross_flat = train_data["k_cross"].unsqueeze(-1).repeat_interleave(a_flat.size(1), dim=1).unsqueeze(-1)#batch, i*a, 1
    pre_k_flat = (1-params.delta)*k_cross_flat
    
    data_e0 = torch.cat([next_k_flat, a_flat, i_flat, next_gm_flat], dim=2)
    data_e1 = torch.cat([pre_k_flat, a_flat, i_flat, next_gm_flat], dim=2)
    value0 = nn.target_value(data_e0).squeeze(-1)
    value1 = nn.target_value(data_e1).squeeze(-1)
    value0 = value0.view(-1, len(params.ashock), len(params.ishock))  # (batch_size, a, i)
    value1 = value1.view(-1, len(params.ashock), len(params.ishock))  # (batch_size, a, i)
    checkv0 = value0[:, 0, 0]
    checkv1 = value1[:, 0, 0]

    # 確率と価値を掛けて期待値を計算
    expected_value0 = (value0 * probabilities).sum(dim=(1, 2)).unsqueeze(-1)  # (batch_size,)
    expected_value1 = (value1 * probabilities).sum(dim=(1, 2)).unsqueeze(-1)  # (batch_size,)
    
    e0 = -next_k * price + params.beta * expected_value0
    e1 = -(1-params.delta) * train_data["k_cross"].unsqueeze(-1) * price + params.beta * expected_value1
    
    return e0, e1, next_k


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
    k_now_k = k_now[:, 0]  # Assuming ashock is scalar for now

    # Initialize aggregate shock 'a'
    a_value = torch.randint(0, len(params.ashock), (1,))
    a = torch.full((grid_size, i_size), params.ashock[a_value].item(), dtype=params.pi_i.dtype)

    dist_history = []
    k_history = []
    dist_k_history = []
    grid_k_history = []
    ashock_history = []
    mean_k_history = []

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

        price = price_fn(basic_s["grid"], basic_s["dist"], basic_s["ashock"][:,0], nn, mean=mean)#G,1
        if p_init is not None:
            price = torch.full_like(price, p_init, dtype=TORCH_DTYPE)
        # Policy function for adjusted capital
        k_prime_adj = policy_fn_sim(basic_s["ashock"], basic_s["ishock"], basic_s["grid_k"], basic_s["dist_k"], price.expand(-1, i_size), nn)  # (G, I, 1)
        k_prime_adj = k_prime_adj.squeeze(-1)  # (G, I)

        # Capital for non-adjusting agents
        k_prime_non_adj = (1 - params.delta) * basic_s["k_cross"].unsqueeze(1).expand(-1, i_size)  # (G, I)

        # Map k_prime to the capital grid using the refactored function
        idx_adj_lower, idx_adj_upper, weight_adj = map_to_grid(k_prime_adj, params.k_grid)
        idx_non_adj_lower, idx_non_adj_upper, weight_non_adj = map_to_grid(k_prime_non_adj, params.k_grid)

        # Initialize new distribution
        dist_new = torch.zeros_like(dist_now)

        
        update_distribution(dist_new, dist_now, alpha, idx_adj_lower, idx_adj_upper, weight_adj, params.pi_i, adjusting=True)
        
        update_distribution(dist_new, dist_now, alpha, idx_non_adj_lower, idx_non_adj_upper, weight_non_adj, params.pi_i, adjusting=False)



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
    if last_dist is True:
        nn.init_dist = dist_now
        nn.init_dist_k = dist_now_k

    return {
        "grid": k_history[100:],         # 100番目から最後まで
        "dist": dist_history[100:],      # 100番目から最後まで
        "dist_k": dist_k_history[100:],  # 100番目から最後まで
        "grid_k": grid_k_history[100:],  # 100番目から最後まで
        "ashock": ashock_history[100:],  # 100番目から最後まで
        "mean_k": mean_k_history[100:],  # 100番目から最後まで
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
    k_prime_flat = k_prime.view(-1)
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