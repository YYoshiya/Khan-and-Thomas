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
import pred_train as pred
from param import KTParam as params


class ValueNN(nn.Module):
    def __init__(self, d_in):
        super(ValueNN, self).__init__()
        self.fc1 = nn.Linear(d_in, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, 24)
        self.fc4 = nn.Linear(24, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class GeneralizedMomModel(nn.Module):
    def __init__(self, d_in):
        super(GeneralizedMomModel, self).__init__()
        self.fc1 = nn.Linear(d_in, 12)
        self.fc2 = nn.Linear(12, 12)
        self.fc3 = nn.Linear(12, 12)
        self.fc4 = nn.Linear(12, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x #このあとこれと分布の内積をとる。

class NextkNN(nn.Module):
    def __init__(self, d_in):
        super(PriceNN, self).__init__()
        self.fc1 = nn.Linear(d_in, 12)
        self.fc2 = nn.Linear(12, 12)
        self.fc3 = nn.Linear(12, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return x
    
        

class PriceNN(nn.Module):
    def __init__(self, d_in):
        super(PriceNN, self).__init__()
        self.fc1 = nn.Linear(d_in, 12)
        self.fc2 = nn.Linear(12, 12)
        self.fc3 = nn.Linear(12, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return x


class nn_class:
    def __init__(self):
        self.value0 = ValueNN(4)
        self.policy = NextkNN(2)
        self.gm_model = GeneralizedMomModel(1)
        self.gm_model_policy = GeneralizedMomModel(1)
        self.next_gm_model = PriceNN(1)
        self.gm_model_price = GeneralizedMomModel(1)
        self.price_model = PriceNN(2)
        params_value = list(self.value0.parameters()) + list(self.gm_model.parameters())
        params_policy = list(self.policy.parameters()) + list(self.gm_model_policy.parameters())
        params_price = list(self.gm_model_price.parameters()) + list(self.price_model.parameters())
        params_next_gm = list(self.next_gm_model.parameters())
        self.optimizer_val = optim.Adam(params_value, lr=0.001)
        self.optimizer_pol = optim.Adam(params_policy, lr=0.001)
        self.optimizer_pri = optim.Adam(params_price, lr=0.001)
        self.optimizer_next_gm = optim.Adam(params_next_gm, lr=0.001)

        
nn = nn_class()



while diff > critout:
    vi.policy_iter(params, nn.optimizer_pol, nn)
    vi.value_iter(nn, params, nn.optimizer_val, 200)
    pred.price_train(nn, nn.optimizer_pri, 10, 500, 1e-4)
    pred.next_gm_train(nn, params, nn.optimizer_next_gm, T)