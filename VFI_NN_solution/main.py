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

DTYPE = "float32"
if DTYPE == "float64":
    NP_DTYPE = np.float64
    TORCH_DTYPE = torch.float64  # PyTorchのデータ型を指定
elif DTYPE == "float32":
    NP_DTYPE = np.float32
    TORCH_DTYPE = torch.float32  # PyTorchのデータ型を指定
else:
    raise ValueError("Unknown dtype.")



class ValueNN(nn.Module):
    def __init__(self, d_in):
        super(ValueNN, self).__init__()
        self.fc1 = nn.Linear(d_in, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.leakyrelu = nn.LeakyReLU()
    def forward(self, x):
        x = self.leakyrelu(self.fc1(x))
        x = self.leakyrelu(self.fc2(x))
        x = self.leakyrelu(self.fc3(x))
        x = self.fc4(x)
        return x
    
class TargetValueNN(nn.Module):
    def __init__(self, d_in):
        super(TargetValueNN, self).__init__()
        self.fc1 = nn.Linear(d_in, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.leakyrelu = nn.LeakyReLU()
    def forward(self, x):
        x = self.leakyrelu(self.fc1(x))
        x = self.leakyrelu(self.fc2(x))
        x = self.leakyrelu(self.fc3(x))
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
        self.leakyrelu = nn.LeakyReLU()
        self.softplus = nn.Softplus()
    def forward(self, x):
        x = self.leakyrelu(self.fc1(x))
        x = self.leakyrelu(self.fc2(x))
        x = self.leakyrelu(self.fc3(x))
        x = self.softplus(self.fc4(x))
        return x #このあとこれと分布の内積をとる。
    
class Price_GM(nn.Module):
    def __init__(self, d_in):
        super(Price_GM, self).__init__()
        self.fc1 = nn.Linear(d_in, 12)
        self.fc2 = nn.Linear(12, 12)
        self.fc3 = nn.Linear(12, 12)
        self.fc4 = nn.Linear(12, 5)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        self.leakyrelu = nn.LeakyReLU()
    def forward(self, x):
        x = self.leakyrelu(self.fc1(x))
        x = self.leakyrelu(self.fc2(x))
        x = self.leakyrelu(self.fc3(x))
        x = self.softplus(self.fc4(x))
        return x #このあとこれと分布の内積をとる。

class NextkNN(nn.Module):
    def __init__(self, d_in):
        super(NextkNN, self).__init__()
        self.fc1 = nn.Linear(d_in, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.leakyrelu = nn.LeakyReLU()
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.leakyrelu(self.fc1(x))
        x = self.leakyrelu(self.fc2(x))
        x = self.leakyrelu(self.fc3(x))
        x = self.leakyrelu(self.fc4(x))
        return x
    
class PriceNN(nn.Module):
    def __init__(self, d_in):
        super(PriceNN, self).__init__()
        self.fc1 = nn.Linear(d_in, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, 24)
        self.fc4 = nn.Linear(24, 24)
        self.output = nn.Linear(24, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        self.leakyrelu = nn.LeakyReLU()
    def forward(self, x):
        x = self.leakyrelu(self.fc1(x))
        x = self.leakyrelu(self.fc2(x))
        x = self.leakyrelu(self.fc3(x))
        #x = self.tanh(self.fc4(x))
        x = self.output(x)
        return x


class Next_gmNN(nn.Module):
    def __init__(self, d_in):
        super(Next_gmNN, self).__init__()
        self.fc1 = nn.Linear(d_in, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, 24)
        self.output = nn.Linear(24, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        self.leakyrelu = nn.LeakyReLU()
    def forward(self, x):
        x = self.leakyrelu(self.fc1(x))
        x = self.leakyrelu(self.fc2(x))
        x = self.leakyrelu(self.fc3(x))
        x = self.output(x)
        return x

class basic_dataset:
    def __init__(self, data):
        self.data = data

class BasicDatasetGM:
    def __init__(self, data, device=None):
        """
        初期化メソッド。データをGPUまたは指定されたデバイスに送ります。

        Args:
            data (dict): GPUに送信したいデータの辞書。
            device (torch.device, optional): データを送るデバイス。指定がない場合はCUDAが利用可能ならCUDAを使用し、それ以外はCPUを使用。
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.data = self.send_to_device(data)

    def send_to_device(self, data):
        """
        データを指定されたデバイスに送るメソッド。

        Args:
            data (dict): GPUに送信したいデータの辞書。

        Returns:
            dict: デバイスに送られたデータの辞書。
        """
        data_gpu = {}
        for key, value in data.items():
            if isinstance(value, list):
                if key == "ashock":
                    # スカラー値のリストをテンソルに変換
                    data_gpu[key] = torch.tensor(value, dtype=torch.float32).to(self.device)
                elif all(isinstance(v, torch.Tensor) for v in value):
                    try:
                        # テンソルのリストをスタックして1つのテンソルに
                        data_gpu[key] = torch.stack(value, dim=0).to(self.device)
                    except ValueError as e:
                        print(f"Error stacking tensors for key '{key}': {e}")
                        raise
                else:
                    print(f"Unsupported list element types for key '{key}'.")
                    raise TypeError(f"Unsupported list element types for key '{key}'.")
            elif isinstance(value, torch.Tensor):
                # 既にテンソルであれば直接デバイスに移動
                data_gpu[key] = value.to(self.device)
            else:
                print(f"Unsupported data type for key '{key}': {type(value)}")
                raise TypeError(f"Unsupported data type for key '{key}': {type(value)}")
        return data_gpu

    def get_data(self):
        """
        データを取得するメソッド。

        Returns:
            dict: デバイスに送られたデータの辞書。
        """
        return self.data

    def update_data(self, new_data):
        """
        データを更新するメソッド。新しいデータをデバイスに送ってself.dataを更新します。

        Args:
            new_data (dict): 新しいデータの辞書。
        """
        self.data = self.send_to_device(new_data)



class nn_class:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.value0 = ValueNN(4).to(self.device)
        self.target_value = TargetValueNN(4).to(self.device)
        self.policy = NextkNN(4).to(self.device)
        self.gm_model = GeneralizedMomModel(1).to(self.device)
        self.target_gm_model = GeneralizedMomModel(1).to(self.device)
        self.gm_model_policy = GeneralizedMomModel(1).to(self.device)
        self.next_gm_model = Next_gmNN(2).to(self.device)
        self.gm_model_price =Price_GM(5).to(self.device)
        self.price_model = PriceNN(6).to(self.device)
        self.params_value = list(self.value0.parameters()) + list(self.gm_model.parameters())
        self.params_policy = list(self.policy.parameters()) + list(self.gm_model_policy.parameters())
        self.params_price = list(self.gm_model_price.parameters()) + list(self.price_model.parameters())
        self.params_next_gm = list(self.next_gm_model.parameters())
        self.optimizer_valueinit = optim.Adam(self.value0.parameters(), lr=0.001)
        self.optimizer_policyinit = optim.Adam(self.policy.parameters(), lr=0.001)
        self.optimizer_val = optim.Adam(self.params_value, lr=0.0004)
        self.optimizer_pol = optim.Adam(self.params_policy, lr=0.00005)
        self.optimizer_pri = optim.Adam(self.params_price, lr=0.001)
        self.optimizer_next_gm = optim.Adam(self.params_next_gm, lr=0.001)

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
n_model.target_value.load_state_dict(n_model.value0.state_dict())
n_model.target_gm_model.load_state_dict(n_model.gm_model.state_dict())

init_price = 2.7
mean=None

vi.value_init(n_model, params, n_model.optimizer_valueinit, 1000, 10)
pred.next_gm_init(n_model, params, n_model.optimizer_next_gm, 10, 10, 1000)
vi.policy_iter_init2(params,n_model.optimizer_policyinit, n_model, 1000, 10, init_price)
with torch.no_grad():
    dataset_grid = vi.get_dataset(params, 1100, n_model, init_price, mean)
    vi.plot_mean_k(dataset_grid, 500, 600)
train_ds_gm = BasicDatasetGM(dataset_grid)
train_ds = basic_dataset(dataset_grid)
params.B = 0.0083
n_model.target_value.load_state_dict(n_model.value0.state_dict())
n_model.target_gm_model.load_state_dict(n_model.gm_model.state_dict())

vi.policy_iter(train_ds.data, params, n_model.optimizer_pol, n_model, 1000, 10, p_init=init_price, mean=mean)
#new_data = vi.get_dataset(params, 1100, n_model, init_price, mean)

#train_ds_gm.update_data(new_data)
#train_ds.data = new_data
with torch.no_grad():
    true_price, dist_new, params.price_size = pred.bisectp(n_model, params, train_ds_gm.data, init=init_price)
pred.price_train(train_ds.data, true_price, n_model, 200)
pred.next_gm_train(train_ds.data, dist_new, n_model, params, n_model.optimizer_next_gm, 1000, 10, 100)


count = 0
loss_value = []
loss_policy = []
previous_loss = 0
for _ in range(50):

    count += 1
    loss_p = vi.policy_iter(train_ds.data, params, n_model.optimizer_pol, n_model, 1000, 10, mean=mean)
    loss_v = vi.value_iter(train_ds.data, n_model, params, n_model.optimizer_val, 1000, 10, mean=mean)
    loss_p = vi.policy_iter(train_ds.data, params, n_model.optimizer_pol, n_model, 1000, 10, mean=mean)

    with torch.no_grad():
            true_price, dist_new, params.price_size = pred.bisectp(n_model, params, train_ds_gm.data)
    pred.price_train(train_ds.data, true_price, n_model, 100)
    pred.next_gm_train(train_ds.data, dist_new, n_model, params, n_model.optimizer_next_gm, 1000, 10, 100)
    #loss_p = vi.policy_iter(train_ds.data, params, n_model.optimizer_pol, n_model, 1000, 10, mean=mean)
    #if loss_v < 0.01:
        #n_model.optimizer_val = optim.Adam(n_model.params_value, lr=0.00001)
        #n_model.optimizer_pol = optim.Adam(n_model.params_policy, lr=0.00001)
    loss_value.append(loss_v)
    #loss_policy.append(loss_p)
    #loss_change = abs(loss_p - previous_loss)
    if count % 3 == 0:
        with torch.no_grad():
                new_data = vi.get_dataset(params, 2000, n_model, mean=mean, init_dist=True, last_dist=False)
                vi.plot_mean_k(dataset_grid, 500, 600)
    
    #previous_loss = loss_p
    #if count % 10 == 0:
        
        #vi.plot_mean_k(new_data, 500, 600)
        #train_ds_gm.update_data(new_data)
        #train_ds.data = new_data
        #with torch.no_grad():
            #true_price, dist_new = pred.bisectp(n_model, params, train_ds_gm.data)
        #pred.price_train(train_ds.data, true_price, n_model, 200)

loss_p = vi.policy_iter(train_ds.data, params, n_model.optimizer_pol, n_model, 1000, 10, mean=mean)
check_data = vi.get_dataset(params, 1000, n_model, mean=mean, init_dist=True, last_dist=False)