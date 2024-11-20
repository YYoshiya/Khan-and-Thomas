import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import value_iter as vi
import pred_train as pred
from param import KTParam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        super(NextkNN, self).__init__()
        self.fc1 = nn.Linear(d_in, 12)
        self.fc2 = nn.Linear(12, 12)
        self.fc3 = nn.Linear(12, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class PriceNN(nn.Module):
    def __init__(self, d_in):
        super(PriceNN, self).__init__()
        self.fc1 = nn.Linear(d_in, 12)
        self.fc2 = nn.Linear(12, 12)
        self.fc3 = nn.Linear(12, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softplus(self.fc3(x))
        return x


class nn_class:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.value0 = ValueNN(4).to(self.device)
        self.policy = NextkNN(2).to(self.device)
        self.gm_model = GeneralizedMomModel(1).to(self.device)
        self.gm_model_policy = GeneralizedMomModel(1).to(self.device)
        self.next_gm_model = PriceNN(2).to(self.device)
        self.gm_model_price = GeneralizedMomModel(1).to(self.device)
        self.price_model = PriceNN(2).to(self.device)
        params_value = list(self.value0.parameters()) + list(self.gm_model.parameters())
        params_policy = list(self.policy.parameters()) + list(self.gm_model_policy.parameters())
        params_price = list(self.gm_model_price.parameters()) + list(self.price_model.parameters())
        params_next_gm = list(self.next_gm_model.parameters())
        self.optimizer_val = optim.Adam(params_value, lr=0.001)
        self.optimizer_pol = optim.Adam(params_policy, lr=0.001)
        self.optimizer_pri = optim.Adam(params_price, lr=0.001)
        self.optimizer_next_gm = optim.Adam(params_next_gm, lr=0.001)

def initialize_weights(model):
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            init.kaiming_normal_(layer.weight, nonlinearity='relu')  # He Initialization
            if layer.bias is not None:
                init.constant_(layer.bias, 0)  # バイアスをゼロに初期化

        
n_model = nn_class()
params = KTParam()

n_model.value0.apply(initialize_weights)
n_model.policy.apply(initialize_weights)
n_model.gm_model.apply(initialize_weights)
n_model.gm_model_policy.apply(initialize_weights)
n_model.next_gm_model.apply(initialize_weights)
n_model.gm_model_price.apply(initialize_weights)

vi.policy_iter_init(params,n_model.optimizer_pol, n_model, 500, 10)
count = 0
for _ in range(50):
    count += 1
    vi.value_iter(n_model, params, n_model.optimizer_val, 500, 10)
    vi.policy_iter(params, n_model.optimizer_pol, n_model, 500, 10)
    #if count % 7 == 0:
    pred.price_train(params, n_model, n_model.optimizer_pri, 10, 10, 500, 1e-4)#Tを変えてる。
    #pred.next_gm_train(n_model, params, n_model.optimizer_next_gm, 500, 10)