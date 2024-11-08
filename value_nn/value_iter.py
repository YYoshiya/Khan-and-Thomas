import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset, DataLoader, random_split

class MyDictDataset(Dataset):
    def __init__(self, next_k, ashock, ishock, dist):
        self.next_k = next_k
        self.ashock = ashock
        self.ishock = ishock
        self.dist = dist
    
    def __len__(self):
        return self.next_k.size(0)

    def __getitem__(self, idx):
        return {
            'next_k': self.next_k[idx],          # 例: スカラーや1次元テンソル
            'ashock': self.ashock[idx],          # 例: スカラーや1次元テンソル
            'ishock': self.ishock[idx],          # 例: スカラーや1次元テンソル
            'dist': self.dist[idx]         # 例: 1次元テンソル（サイズ50）
        }


def value_fn(train_data, value0, policy, gm_model, params):
    gm = gm_model(train_data["dist"])
    state = torch.cat([train_data["next_k"], train_data["ashock"], train_data["ishock"], gm], dim=1)
    value = value0(state)
    return value

def policy_fn(ashock, dist, policy, gm_model):
    gm = gm_model(dist)
    state = torch.cat([ashock, gm], dim=1)
    next_k = policy(state)
    return next_k

def price_fn(train_data, gm_model_price, price_model):
    gm_price = gm_model_price(train_data["dist"])
    state = torch.cat([train_data["ashock"], gm_price], dim=1)
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

def value_iter(value0, policy, gm_model, gm_model_price, price_model, params, optimizer):
    data = get_dataset(policy, gm)
    dataset = MyDataset(data)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    for train_data in dataloader:
        v = value_fn(train_data, value0, policy, gm_model, params)#value_fn書いて
        price = price_fn(train_data, gm_model_price, price_model)#入力は分布とashockかな。
        #wage = params.eta / price
        profit = get_profit(train_data["k_grid"], train_data["ashock"], price, params)
        v0_exp, v1_exp = next_value(train_data, policy, params)#ここ書いてgrid, gm, ashock, ishockの後ろ二つに関する期待値 v0_expなんかおかしい
        e0 = -params.gamma * policy_fn(train_data["ashock"], train_data["dist"], policy, gm_model) * price + params.beta * v0_exp
        e1 = -(1-params.delta) * train_data["k_cross"]* price + params.beta * v1_exp
        threshold = (e0 - e1) / marams.eta
        xi = min(params.B, max(0, threshold))
        vnew = profit - p*w*xi**2/(2*params.B) + xi/params.B*e0 + (1-xi/params.B)*e1
        loss = torch.sum((vnew - v)**2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return value0


def get_profit(k_cross, ashock, price, params):
    wage = params.eta / price
    yterm = ashock * k_cross**params.theta
    n = (params.nu * yterm / wage)**(1 / (1 - params.nu))
    y = yterm * n**params.nu
    v0temp = y - wage * n + (1 - params.delta) * k_cross
    return v0temp*price

def next_value(train_data, value0, policy, params):
    next_gm = dist_gm(train_data["dist"])
    ashock = train_data["ashock"]
    ashock_idx = [torch.where(params.ashock == val)[0].item() for val in ashock]
    ashock_exp = params.pi_a[ashock_idx]
    ishock = train_data["ishock"]
    ishock_idx = [torch.where(params.ishock == val)[0].item() for val in ishock]
    ishock_exp = params.pi_i[ishock_idx]
    
    k_cross = train_data["k_cross"]
    pre_k_cross = (1-params.delta)*k_cross
    data_policy = torch.cat([ashock, next_gm], dim=1)
    next_k = policy_fn(ashock, train_data["dist"], policy, gm_model)
    data_e0 = torch.cat([next_k, ashock, ishock, next_gm], dim=1)
    data_e1 = torch.cat([pre_k_cross/params.gamma, ashock, ishock, next_gm], dim=1)
    value_e0 = value0(data_e0)
    value_e1 = value0(data_e1)
    value_exp_e0_tmp = value_e0 * ashock_exp * ishock_exp
    value_exp_e1_tmp = value_e1 * ashock_exp * ishock_exp
    value_exp_e0 = torch.sum(value_exp_tmp, dim=1)
    value_exp_e1 = torch.sum(value_exp_tmp, dim=1)
    return value_exp_e0, value_exp_e1