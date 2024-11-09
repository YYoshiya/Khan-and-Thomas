import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import value_iter as vi

def price_train(nn_class, data):#k_gridに関してxiを求める他は適当でよい。
    price = vi.price_fn(data["grid"], data["dist"], data["ashock"])
    v0_exp, v1_exp = next_value(data, params)#ここ書いてgrid, gm, ashock, ishockの後ろ二つに関する期待値 v0_expなんかおかしい
    e0 = -params.gamma * policy_fn(train_data["ashock"], train_data["grid"], train_data["dist"]) * price + params.beta * v0_exp
    e1 = -(1-params.delta) * train_data["k_cross"]* price + params.beta * v1_exp
    threshold = (e0 - e1) / marams.eta
    xi = min(params.B, max(0, threshold))
    alpha = xi / params.B
    inow = alpha * (params.gamma * k_next - (1-params.delta) * train_data["k"])#k_nextをかいて
    ynow = train_data["ashock"]*train_data["ishock"] * train_data["k"]**params.theta * inow**params.nu
    