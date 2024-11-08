import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset, DataLoader, random_split


def value_iter(value0, gm, params):
    data = get_dataset(value0, gm)
    dataset = MyDataset(data)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    for train_data in dataloader:
        price = price_fn()#入力は分布とashockかな。
        #wage = params.eta / price
        profit = get_profit(train_data["k_grid"], train_data["ashock"], price, params)
        v0_exp = a#ここ書いてgrid, gm, ashock, ishockの後ろ二つに関する期待値 v0_expなんかおかしい
        opk = torch.argmax(-params.gamma * k_grid * price + params.beta * v0_exp)
        e0 = -params.gamma * k_grid[opk] * price + params.beta * v0_exp[opk]
        e1 = -params.gamma * (1-params.delta)*train_data[k_grid] * price + params.beta * v0_exp
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

def next_value(train_data, params):
    next_gm = dist_gm(train_data["dist"])
    for a in a_grid:#ここ確率遷移を知らないとだめ。
        for i in i_grid:
            k_cross = k_grid.unsqueeze(0)
            a = a.unsqueeze(0)
            i = i.unsqueeze(0)
            data = torch.cat([k_cross, a, i, next_gm], dim=1)
            value = model(data)
            value_exp += value *