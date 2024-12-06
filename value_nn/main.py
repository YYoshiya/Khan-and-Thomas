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
import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)
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
        self.fc1 = nn.Linear(d_in, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, 24)
        self.fc4 = nn.Linear(24, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.softplus(self.fc4(x))
        return x
    
class PriceNN(nn.Module):
    def __init__(self, d_in):
        super(PriceNN, self).__init__()
        self.fc1 = nn.Linear(d_in, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.softplus(self.output(x))
        return x


class Next_gmNN(nn.Module):
    def __init__(self, d_in):
        super(Next_gmNN, self).__init__()
        self.fc1 = nn.Linear(d_in, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 32)
        self.output = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.output(x)
        return x

class basic_dataset:
    def __init__(self, data):
        self.data = data

class basic_dataset_gm:
    def __init__(self, data):
        self.data = data


class nn_class:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.value0 = ValueNN(4).to(self.device)
        self.policy = NextkNN(3).to(self.device)
        self.gm_model = GeneralizedMomModel(1).to(self.device)
        self.gm_model_policy = GeneralizedMomModel(1).to(self.device)
        self.next_gm_model = Next_gmNN(2).to(self.device)
        self.gm_model_price = GeneralizedMomModel(1).to(self.device)
        self.price_model = PriceNN(2).to(self.device)
        params_value = list(self.value0.parameters()) + list(self.gm_model.parameters())
        params_policy = list(self.policy.parameters()) + list(self.gm_model_policy.parameters())
        params_price = list(self.gm_model_price.parameters()) + list(self.price_model.parameters())
        params_next_gm = list(self.next_gm_model.parameters())
        self.optimizer_valueinit = optim.Adam(self.value0.parameters(), lr=0.001)
        self.optimizer_policyinit = optim.Adam(self.policy.parameters(), lr=0.001)
        self.optimizer_val = optim.Adam(params_value, lr=0.00001)
        self.optimizer_pol = optim.Adam(params_policy, lr=0.00001)
        self.optimizer_pri = optim.Adam(params_price, lr=0.00005)
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
n_model.price_model.apply(initialize_weights)

init_price = 3.2

vi.value_init(n_model, params, n_model.optimizer_valueinit, 1000, 10)
pred.next_gm_init(n_model, params, n_model.optimizer_next_gm, 10, 10, 1000)
vi.policy_iter_init2(params,n_model.optimizer_policyinit, n_model, 1000, 10)

dataset_grid = vi.get_dataset(params, 1000, n_model, 10, init_price)
train_ds = basic_dataset(dataset_grid)
vi.policy_iter(train_ds.data, params, n_model.optimizer_pol, n_model, 1000, 10, p_init=init_price)
train_ds.data = vi.get_dataset(params, 1000, n_model, 10, init_price)
pred.price_train(train_ds.data, params, n_model, n_model.optimizer_pri, 200, 64, 1000, 1e-5)
pred.next_gm_train(train_ds.data, n_model, params, n_model.optimizer_next_gm, 1000, 10, 30)
train_ds.data = vi.get_dataset(params, 1000, n_model, 10)
pred.price_train(train_ds.data, params, n_model, n_model.optimizer_pri, 200, 128, 900, 1e-5)
#pred.next_gm_train(train_ds.data, n_model, params, n_model.optimizer_next_gm, 1000, 10, 100)


#for _ in range(10):
count = 0
loss_value = []
loss_policy = []
for _ in range(50):
    #params.B = 0.06
    count += 1
    
    loss_v = vi.value_iter(train_ds.data, n_model, params, n_model.optimizer_val, 1000, 10)
    loss_p = vi.policy_iter(train_ds.data, params, n_model.optimizer_pol, n_model, 1000, 10)
    pred.next_gm_train(train_ds.data, n_model, params, n_model.optimizer_next_gm, 1000, 10, 100)
    loss_value.append(loss_v)
    #loss_policy.append(loss_p)
    if count % 3 == 0:
        for _ in range(2):
            pred.price_train(train_ds.data, params, n_model, n_model.optimizer_pri, 200, 64, 1000, 1e-5)
            train_ds.data = vi.get_dataset(params, 1000, n_model, 10)
    #pred.next_gm_train(train_ds.data, n_model, params, n_model.optimizer_next_gm, 1000, 10, 30)
    
    #train_ds.data = vi.get_dataset(params, 1000, n_model, 10)
    #params.B = 0.0083