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
        self.gm_model_price = GeneralizedMomModel(1)
        self.price_model = PriceNN(2)
        
nn = nn_class()
    
    

while diff > critout:
    value_iter()
    update_gm_price()
    update_gm_gm()