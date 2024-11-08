import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset, DataLoader, random_split


def value_iter(value0, policy, gm, params):
    data = get_dataset(value0, gm)
    dataset = MyDataset(data)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    for train_data in dataloader:
        price = price_fn()#入力は分布とashockかな。
        #wage = params.eta / price
        profit = get_profit(train_data["k_grid"], train_data["ashock"], price, params)
        v0_exp, v1_exp = next_value(train_data, policy, params)#ここ書いてgrid, gm, ashock, ishockの後ろ二つに関する期待値 v0_expなんかおかしい
        e0 = -params.gamma * polich_fn * price + params.beta * v0_exp
        e1 = -(1-params.delta) * train_data["k_cross"]* price + params.beta * v1_exp
        threshold = (e0 - e1) / marams.eta
        xi = min(params.B, max(0, threshold))
        vnew = profit - p*w*xi**2/(2*params.B) + xi/params.B*e0 + (1-xi/params.B)*e1


def get_profit(k_cross, ashock, price, params):
    wage = params.eta / price
    yterm = ashock * k_cross**params.theta
    n = (params.nu * yterm / wage)**(1 / (1 - params.nu))
    y = yterm * n**params.nu
    v0temp = y - wage * n + (1 - params.delta) * k_cross
    return v0temp*price

def next_value(train_data, policy, params):
    next_gm = dist_gm(train_data["dist"])
    ashock = train_data["ashock"]
    ashock_idx = [torch.where(params.ashock == val)[0].item() for val in ashock]
    ashock_exp = params.pi_a[ashock_idx]
    ishock = train_data["ishock"]
    ishock_idx = [torch.where(params.ishock == val)[0].item() for val in ishock]
    ishock_exp = params.pi_i[ishock_idx]
    
    a = ashock.unsqueeze(1)
    i = ishock.unsqueeze(1)
    k_cross = train_data["k_cross"]
    pre_k_cross = (1-params.delta)*k_cross
    data_policy = torch.cat([a, next_gm], dim=1)
    next_k = policy(data_policy)
    data_e0 = torch.cat([next_k, a, i, next_gm], dim=1)
    data_e1 = torch.cat([(pre_k_cross/params.gamma).unsqueeze(1), a, i, next_gm], dim=1)
    value_e0 = model(data_e0)
    value_e1 = model(data_e1)
    value_exp_e0_tmp = value * ashock_exp * ishock_exp
    value_exp_e1_tmp = value * ashock_exp * ishock_exp
    value_exp_e0 = torch.sum(value_exp_tmp, dim=1)
    value_exp_e1 = torch.sum(value_exp_tmp, dim=1)
    return value_exp_e0, value_exp_e1