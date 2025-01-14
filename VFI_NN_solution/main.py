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
import simul as sim
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
        self.fc1 = nn.Linear(d_in, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        #self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.leakyrelu = nn.LeakyReLU()
    def forward(self, x):
        x = self.leakyrelu(self.fc1(x))
        x = self.leakyrelu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class TargetValueNN(nn.Module):
    def __init__(self, d_in):
        super(TargetValueNN, self).__init__()
        self.fc1 = nn.Linear(d_in, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        #self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.leakyrelu = nn.LeakyReLU()
    def forward(self, x):
        x = self.leakyrelu(self.fc1(x))
        x = self.leakyrelu(self.fc2(x))
        x = self.fc3(x)
        return x

class GeneralizedMomModel(nn.Module):
    def __init__(self, d_in):
        super(GeneralizedMomModel, self).__init__()
        self.fc1 = nn.Linear(d_in, 12)
        self.fc2 = nn.Linear(12, 12)
        self.fc3 = nn.Linear(12, 1)
        #self.fc4 = nn.Linear(12, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.leakyrelu = nn.LeakyReLU()
        self.softplus = nn.Softplus()
    def forward(self, x):
        x = self.leakyrelu(self.fc1(x))
        x = self.leakyrelu(self.fc2(x))
        x = self.softplus(self.fc3(x))
        return x #このあとこれと分布の内積をとる。
    
class Price_GM(nn.Module):
    def __init__(self, d_in):
        super(Price_GM, self).__init__()
        self.fc1 = nn.Linear(d_in, 12)
        self.fc2 = nn.Linear(12, 12)
        self.fc3 = nn.Linear(12, 5)
        #self.fc4 = nn.Linear(12, 5)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        self.leakyrelu = nn.LeakyReLU()
    def forward(self, x):
        x = self.leakyrelu(self.fc1(x))
        x = self.leakyrelu(self.fc2(x))
        x = self.softplus(self.fc3(x))
        
        return x #このあとこれと分布の内積をとる。

class NextkNN(nn.Module):
    def __init__(self, d_in):
        super(NextkNN, self).__init__()
        self.fc1 = nn.Linear(d_in, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        #self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.leakyrelu = nn.LeakyReLU()
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.leakyrelu(self.fc1(x))
        x = self.leakyrelu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
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
        #x = self.leakyrelu(self.fc3(x))
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
        #x = self.leakyrelu(self.fc3(x))
        x = self.output(x)
        return x

def append_and_trim_data(
    old_data: dict, 
    new_data: dict,
    start_append: int,
    end_append: int,
    remove_count: int
) -> dict:
    """
    Both old_data and new_data are expected to have the following structure:
        {
            'grid': [Tensor, Tensor, ...],
            'dist': [Tensor, Tensor, ...],
            'dist_k': [...],
            ...
        }
    where each key is associated with a list of Tensors in chronological order.

    This function removes the first 'remove_count' items from old_data
    and appends the slice [start_append : end_append] from new_data.

    Returns:
        A dict containing the updated data.
    """
    updated_data = {}
    for key in old_data.keys():
        # old_data[key] should be a list of Tensors
        # new_data[key] should be a list of Tensors
        old_list = old_data[key]  
        trimmed_old_list = old_list[remove_count:]  # discard the first 'remove_count' items
        slice_from_new = new_data[key][start_append:end_append]

        # If both are lists of Tensors, we can concatenate them with '+'
        merged_list = trimmed_old_list + slice_from_new
        updated_data[key] = merged_list

    return updated_data

class BasicDataset:
    """
    This class unifies the functionality of basic_dataset and BasicDatasetGM.
    It stores two attributes:
      - data_cpu: raw CPU data (a dict of lists of Tensors)
      - data_gm: the GPU/GM version of that data (same shape, but on device)

    The method 'update_cpu_data' automatically rebuilds data_gm
    by calling _convert_to_gm_shape (similar to BasicDatasetGM logic).
    """
    def __init__(self, data_cpu: dict, device=None):
        # If no device is specified, default to CUDA if available, else CPU
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Store CPU data
        self.data_cpu = data_cpu

        # Build GM (GPU) data from the CPU data
        self.data_gm = self._convert_to_gm_shape(data_cpu)

    def _convert_to_gm_shape(self, data_cpu: dict) -> dict:
        """
        Converts the CPU data to GM (GPU) data. 
        Essentially mimics BasicDatasetGM's send_to_device logic.
        """
        data_gpu = {}
        for key, value in data_cpu.items():
            if isinstance(value, list):
                if key == "ashock":
                    # スカラー値のリストをテンソルに変換
                    data_gpu[key] = torch.tensor(value, dtype=TORCH_DTYPE).to(self.device)
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

    def get_data_cpu(self) -> dict:
        """ Returns the CPU data dictionary. """
        return self.data_cpu

    def get_data_gm(self) -> dict:
        """ Returns the GM (GPU) data dictionary. """
        return self.data_gm

    def update_cpu_data(self, new_data_cpu: dict):
        """
        Updates the CPU data and rebuilds the GM data accordingly.
        This replaces the role of 'update_data' in BasicDatasetGM.
        """
        self.data_cpu = new_data_cpu
        self.data_gm = self._convert_to_gm_shape(new_data_cpu)

def conditionally_update_dataset(
    dataset: BasicDataset,
    condition_value: float,
    threshold: float,
    new_data_cpu: dict,
    start_append: int = 1001,
    end_append: int = 1500,
    remove_count: int = 500
):
    """
    If condition_value >= threshold, this function updates dataset.data_cpu
    by removing the first 'remove_count' items and appending items 
    from [start_append : end_append] of new_data_cpu.

    After updating data_cpu, we also regenerate data_gm automatically
    (due to dataset.update_cpu_data).

    Args:
        dataset (BasicDataset): an instance of the unified dataset class
        condition_value (float): the current value being monitored
        threshold (float): the threshold for triggering the data update
        new_data_cpu (dict): new data from something like get_dataset(...)
        start_append (int): first index for appending
        end_append (int): last index (non-inclusive) for appending
        remove_count (int): number of items to discard from the start
    """
    if condition_value <= threshold:
        old_cpu_data = dataset.get_data_cpu()
        updated_cpu_data = append_and_trim_data(
            old_cpu_data,
            new_data_cpu,
            start_append,
            end_append,
            remove_count
        )
        dataset.update_cpu_data(updated_cpu_data)
        print(f"Dataset updated: removed first {remove_count} steps and appended steps [{start_append}:{end_append}].")
    else:
        print("Condition not met. No update performed.")

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
        self.optimizer_val = optim.Adam(self.params_value, lr=0.001)
        self.optimizer_pol = optim.Adam(self.params_policy, lr=0.001)
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

init_price = 2.1
mean=None
simul_T = 600
vi.value_init(n_model, params, n_model.optimizer_valueinit, 1000, 10)
pred.next_gm_init(n_model, params, n_model.optimizer_next_gm, 10, 10, 1000)
vi.policy_iter_init2(params,n_model.optimizer_policyinit, n_model, 1000, 10, init_price)
n_model.target_value.load_state_dict(n_model.value0.state_dict())
n_model.target_gm_model.load_state_dict(n_model.gm_model.state_dict())


with torch.no_grad():
    dataset_grid = vi.get_dataset(params, 1100, n_model, init_price, mean)
    vi.plot_mean_k(dataset_grid, 500, 600)
train_ds = BasicDataset(dataset_grid)
with torch.no_grad():
    true_price, dist_new, params.price_size = pred.bisectp(n_model, params, train_ds.data_gm, init=init_price)
pred.price_train1(train_ds.data_cpu, true_price, n_model, 100)
#pred.next_gm_train1(train_ds.data_cpu, dist_new, n_model, params, n_model.optimizer_next_gm, 1000, 10, 100)



#with torch.no_grad():
    #new_data=sim.simulation(params, n_model, simul_T, init=init_price)
#train_ds = BasicDataset(new_data)
#pred.price_train(train_ds.data_cpu, n_model, 100)
#pred.next_gm_train(train_ds.data_cpu, n_model, params, n_model.optimizer_next_gm, 400, 10, 100)

params.B = 0.0083
#new_data = vi.get_dataset(params, 1100, n_model, init_price, mean)

#train_ds_gm.update_data(new_data)
#train_ds.data = new_data

outer_count = 0
count = 0
loss_value = []
loss_policy = []
previous_loss = 0
for _ in range(50):

    outer_count += 1
    count += 1
    loss_v, min_loss, max_loss = vi.value_iter(train_ds.data_cpu, n_model, params, n_model.optimizer_val, simul_T-100, 10, mean=mean, count=count, save_plot=True)
    
    if loss_v < 0.015:
        with torch.no_grad():
            new_data=sim.simulation(params, n_model, 1500, init=2.0, init_dist=True, last_dist=False)
        train_ds = BasicDataset(new_data)
        pred.price_train(train_ds.data_cpu, n_model, 50)
        pred.next_gm_train(train_ds.data_cpu, n_model, params, n_model.optimizer_next_gm, 400, 10, 50)
        #simul_T = 1500
        
        count = 0
        #vi.policy_iter(train_ds.data_cpu, params, n_model.optimizer_pol, n_model, 1000, 10, mean=mean)
    #loss_p = vi.policy_iter(train_ds.data, params, n_model.optimizer_pol, n_model, 1000, 10, mean=mean)
    #if loss_v < 0.01:
        #n_model.optimizer_val = optim.Adam(n_model.params_value, lr=0.00001)
        #n_model.optimizer_pol = optim.Adam(n_model.params_policy, lr=0.00001)
    loss_value.append(loss_v)
    #loss_policy.append(loss_p)
    #loss_change = abs(loss_p - previous_loss)
    
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