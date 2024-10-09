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
    
    def update_from_simul(self, simul_data):
        init_datadict = dict((k, simul_data[k][..., -1].copy()) for k in self.keys)
        for k in self.keys:
            if len(init_datadict[k].shape) == 1:
                init_datadict[k] = init_datadict[k][:, None] # for macro init state like N in JFV
        notnan = ~(np.isnan(init_datadict["k_cross"]).any(axis=1))
        if np.sum(~notnan) > 0:
            num_nan = np.sum(~notnan)
            num_total = notnan.shape[0]
            print("Warning: {} of {} init samples are nan!".format(num_nan, num_total))
            idx = np.where(notnan)[0]
            idx = np.concatenate([idx, idx[:num_nan]])
            for k in self.keys:
                init_datadict[k] = init_datadict[k][idx]
        self.update_datadict(init_datadict)
    
    def process_vdatadict(self, v_datadict):
        idx_nan = np.logical_or(
            np.isnan(v_datadict["basic_s"]).any(axis=(1, 2)),
            np.isnan(v_datadict["value"]).any(axis=(1, 2))
        )
        ma = self.config["dataset_config"]["moving_average"]
        for key, array in v_datadict.items():
            array = array[~idx_nan].astype(NP_DTYPE)
            self.update_stats(array, key, ma)
            v_datadict[key] = self.normalize_data(array, key)
        print("Average of total utility %f." % (self.stats_dict["value"][0][0]))

        valid_size = self.config["value_config"]["valid_size"]
        n_sample = v_datadict["value"].shape[0]
        if valid_size > 0.2*n_sample:
            valid_size = int(0.2*n_sample)
            print("Valid size is reduced to %d according to small data size!" % valid_size)
        print("The dataset has %d samples in total." % n_sample)
        dataset = CustomDataset(v_datadict)
        train_size = n_sample - valid_size
        train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
        batch_size = self.config["value_config"]["batch_size"]
        train_vdataset = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        valid_vdataset = DataLoader(valid_dataset, batch_size=valid_size, shuffle=False)
        
        return train_vdataset, valid_vdataset
    
    def get_policydataset(self, policy, policy_type, update_init=False):
        policy_config = self.config["policy_config"]
        simul_data = self.simul_k_func(
            self.n_path, policy_config["T"], self.mparam, policy, policy_type,
            state_init=self.datadict
        )
        if update_init:
            self.update_from_simul(simul_data)
        p_datadict = {}
        idx_nan = False
        for k in self.keys:
            arr = simul_data[k].astype(NP_DTYPE)
            arr = arr[..., slice(-policy_config["t_sample"], -1, policy_config["t_skip"])]
            if len(arr.shape) == 3:
                arr = np.swapaxes(arr, 1, 2)
                arr = np.reshape(arr, (-1, self.mparam.n_agt))
                if k != "ishock":
                    idx_nan = np.logical_or(idx_nan, np.isnan(arr).any(axis=1))
            else:
                arr = np.reshape(arr, (-1, 1))
                if k != "ashock":
                    idx_nan = np.logical_or(idx_nan, np.isnan(arr[:, 0]))
            p_datadict[k] = arr
        for k in self.keys:
            p_datadict[k] = p_datadict[k][~idx_nan]
        if policy_config["opt_type"] == "game":
            p_datadict = crazyshuffle(p_datadict)
        policy_ds = BasicDataSet(p_datadict)
        return policy_ds
    
    def simul_k_func(self, n_sample, T, mparam, c_policy, policy_type, state_init=None, shocks=None):
        raise NotImplementedError
    
class KTInitDataSet(InitDataSet):
    def __init__(self, mparam, config):
        super().__init__(mparam, config)
        mats = sio.loadmat(mparam.mats_path)
        self.splines = KT.construct_bspl(mats)
        self.keys = ["k_cross", "ashock", "ishock"]
        self.k_policy_bchmk = lambda k_cross, ashock: KT.k_policy_bspl(k_cross, ashock, self.splines)
        # the first burn for initialization
        self.update_with_burn(self.k_policy_bchmk, "pde")

    def get_valuedataset(self, policy, policy_type, update_init=False):
        value_config = self.config["value_config"]
        t_count = value_config["t_count"]
        t_skip = value_config["t_skip"]
        simul_data = self.simul_k_func(
            self.n_path, value_config["T"], self.mparam, policy, policy_type,
            state_init=self.datadict
        )
        if update_init:
            self.update_from_simul(simul_data)
        
        ashock = simul_data["ashock"]
        k_cross = simul_data["k_cross"]
        profit = simul_data["v0"]
        k_mean = np.mean(k_cross, axis=1, keepdims=True)
        discount = np.power(self.mparam.beta, np.arange(t_count))

        basic_s = np.zeros(shape=[0, self.mparam.n_agt, self.n_basic+1])
        agt_s = np.zeros(shape=[0, self.mparam.n_agt, 1])
        value = np.zeros(shape=[0, self.mparam.n_agt, 1])
        t_idx = 0
        while t_idx + t_count < value_config["T"] - 1:
            k_tmp = k_cross[:, :, t_idx:t_idx+1]
            k_mean_tmp = np.repeat(k_mean[:, :, t_idx:t_idx+1], self.mparam.n_agt, axis=1)
            a_tmp = np.repeat(ashock[:, t_idx:t_idx+1], self.mparam.n_agt, axis=1)
            basic_s_tmp = np.concatenate([k_tmp, k_mean_tmp, a_tmp], axis=-1, keepdims=True)
            v_tmp = np.sum(profit[..., t_idx:t_idx+t_count] * discount, axis=-1, keepdims=True)
            basic_s = np.concatenate([basic_s, k_cross[:, :, t_idx:t_idx+t_count]], axis=0)
            agt_s = np.concatenate([agt_s, ashock[:, t_idx:t_idx+t_count, None]], axis=0)
            value = np.concatenate([value, profit[:, t_idx:t_idx+t_count, None]], axis=0)
            t_idx += t_skip
        
        v_datadict = {"basic_s": basic_s, "agt_s": agt_s, "value": value}
        train_vdataset, valid_vdataset = self.process_vdatadict(v_datadict)
        return train_vdataset, valid_vdataset
    
    def simul_k_func(self, n_sample, T, mparam, c_policy, policy_type, state_init=None, shocks=None):
        return KT.simul_k(n_sample, T, mparam, c_policy, policy_type, price_fn, state_init, shocks)
