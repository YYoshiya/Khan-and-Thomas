import json
import os
import numpy as np
import torch
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import scipy.io as sio
import simulation_KT as KT

DTYPE = "float32"
if DTYPE == "float64":
    NP_DTYPE = np.float64
    TORCH_DTYPE = torch.float64  # PyTorchのデータ型を指定
elif DTYPE == "float32":
    NP_DTYPE = np.float32
    TORCH_DTYPE = torch.float32  # PyTorchのデータ型を指定
else:
    raise ValueError("Unknown dtype.")

class NumpyEncoder(json.JSONEncoder):
    def default(self, o):  # pylint: disable=E0202
        if isinstance(o, np.ndarray):
            return o.tolist()
        return json.JSONEncoder.default(self, o)
class CustomDataset(Dataset):
    def __init__(self, v_datadict):
        self.v_datadict = v_datadict
        self.keys = list(v_datadict.keys())
        self.n_samples = len(v_datadict[self.keys[0]])

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {key: torch.tensor(self.v_datadict[key][idx], dtype=TORCH_DTYPE) for key in self.keys}
    

class BasicDataSet():
    def __init__(self, datadict=None):
        self.datadict, self.keys = None, None
        self.size, self.idx_in_epoch, self.epoch_used = None, None, None
        if datadict:
            self.update_datadict(datadict)
    
    def update_datadict(self, datadict):
        self.datadict = datadict
        self.keys = datadict.keys()
        size_list = [datadict[k].shape[0] for k in self.keys]
        for i in range(1, len(size_list)):
            assert size_list[i] == size_list[0], "The size does not match."
        self.size = size_list[0]
        self.shuffle()
        self.epoch_used = 0

    def shuffle(self):
        idx = np.arange(0, self.size)
        np.random.shuffle(idx)
        self.datadict = dict((k, self.datadict[k][idx]) for k in self.keys)
        self.idx_in_epoch = 0
    #next_batchは完全に要らないので削除

    def next_batch(self, batch_size):
        if self.idx_in_epoch + batch_size > self.size:
            self.shuffle()
            self.epoch_used += 1
        idx = slice(self.idx_in_epoch, self.idx_in_epoch+batch_size)
        self.idx_in_epoch += batch_size
        return dict((k, self.datadict[k][idx]) for k in self.keys)

class DataSetwithStats(BasicDataSet):
    def __init__(self, stats_keys, datadict=None):
        super().__init__(datadict)
        self.stats_keys = stats_keys
        self.stats_dict, self.stats_dict_tf = {}, {}
        for k in stats_keys:
            self.stats_dict[k] = None
            self.stats_dict_tf[k] = None
    
    def update_stats(self, data, key, ma):
        # data can be of shape B * d or B * n_agt * d
        axis_for_mean = tuple(list(range(len(data.shape)-1)))
        if self.stats_dict[key] is None:
            mean, std = data.mean(axis=axis_for_mean), data.std(axis=axis_for_mean)
        else:
            mean_new, std_new = data.mean(axis=axis_for_mean), data.std(axis=axis_for_mean)
            mean, std = self.stats_dict[key]
            mean = mean * ma + mean_new * (1-ma)
            std = std * ma + std_new * (1-ma)
        self.stats_dict[key] = mean, std
        self.stats_dict_tf[key] = torch.tensor([mean, std], dtype=TORCH_DTYPE)

    def normalize_data(self, data, key, withtf=False):
        if withtf:
            mean, std = self.stats_dict_tf[key]
            mean = mean.to(data.device)
            std = std.to(data.device)
        else:
            mean, std = self.stats_dict[key]
        return (data - mean) / std

    def unnormalize_data(self, data, key, withtf=False):
        if withtf:
            mean, std = self.stats_dict_tf[key]
            mean = mean.to(data.device)
            std = std.to(data.device)
        else:
            mean, std = self.stats_dict[key]
        return data * std + mean
    
    def save_stats(self, path):
        with open(os.path.join(path, "stats.json"), "w") as fp:
            json.dump(self.stats_dict, fp, cls=NumpyEncoder)

    def load_stats(self, path):
        with open(os.path.join(path, "stats.json"), "r") as fp:
            saved_stats = json.load(fp)
        for key in saved_stats:
            assert key in self.stats_dict, "The key of stats_dict does not match!"
            mean, std = saved_stats[key]
            mean, std = np.asarray(mean).astype(NP_DTYPE), np.asarray(std).astype(NP_DTYPE)
            self.stats_dict[key] = (mean, std)

            self.stats_dict_tf[key] = (
            torch.tensor(mean, dtype=TORCH_DTYPE),
            torch.tensor(std, dtype=TORCH_DTYPE)
        )
            
class InitDataSet(DataSetwithStats):
    def __init__(self, mparam, config):
        super().__init__(stats_keys=["basic_s", "agt_s", "value"])
        self.mparam = mparam
        self.config = config
        self.n_basic = config["n_basic"]
        self.n_fm = config["n_fm"]  # fixed moments
        self.n_path = config["dataset_config"]["n_path"]
        self.t_burn = config["dataset_config"]["t_burn"]
        self.c_policy_const_share = lambda *args: config["init_const_share"]
        if not config["init_with_bchmk"]:
            assert config["policy_config"]["update_init"], \
                "Must update init data during learning if bchmk policy is not used for sampling init"

    def update_with_burn(self, policy, policy_type, t_burn=None, state_init=None):
        if t_burn is None:
            t_burn = self.t_burn
        if state_init is None:
            state_init = self.datadict
        simul_data = self.simul_k_func(
            self.n_path, t_burn, self.mparam,
            policy, policy_type, state_init=state_init
        )
        self.update_from_simul(simul_data)