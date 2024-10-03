import os
import numpy as np
from tqdm import tqdm
import util
import simulation_KT as KT
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import os


DTYPE = "float32"
if DTYPE == "float64":
    NP_DTYPE = np.float64
    TORCH_DTYPE = torch.float64  # PyTorchのデータ型を指定
elif DTYPE == "float32":
    NP_DTYPE = np.float32
    TORCH_DTYPE = torch.float32  # PyTorchのデータ型を指定
else:
    raise ValueError("Unknown dtype.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPSILON = torch.tensor(1e-3, dtype=TORCH_DTYPE, device=device)

class PolicyTrainer():
    def __init__(self, vtrainers, init_ds, policy_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = init_ds.config
        self.policy_config = self.config["policy_config"]
        self.t_unroll = self.policy_config["t_unroll"]
        self.vtrainers = vtrainers
        self.valid_size = self.policy_config["valid_size"]
        self.sgm_scale = self.policy_config["sgm_scale"] # scaling param in sigmoid
        self.init_ds = init_ds
        self.value_sampling = self.config["dataset_config"]["value_sampling"]
        self.num_vnet = len(vtrainers)
        self.decay_rate = self.policy_config["lr_end"] / self.policy_config["lr_beg"]
        self.mparam = init_ds.mparam
        d_in = self.config["n_basic"] + self.config["n_fm"] + self.config["n_gm"]
        self.model = util.FeedforwardModel(d_in, 1, self.policy_config, name="p_net").to(self.device)
        if self.config["n_gm"] > 0:
            self.gm_model = util.GeneralizedMomModel(1, self.config["n_gm"], self.config["gm_config"], name="v_gm").to(self.device)
            # 両方のモデルのパラメータを集める
            params = list(self.model.parameters()) + list(self.gm_model.parameters())
        else:
            params = self.model.parameters()
        self.optimizer = optim.Adam(
                params,
                lr=self.policy_config["lr_beg"],
                betas=(0.99, 0.99),
                eps=1e-8
            )
        if policy_path is not None:
            self.model.load_weights_after_init(policy_path)
            if self.config["n_gm"] > 0:
                self.gm_model.load_weights_after_init(policy_path.replace(".pth", "_gm.pth"))
            self.init_ds.load_stats(os.path.dirname(policy_path))
        lr_scheduler = ExponentialLR(self.optimizer, gamma=self.decay_rate)
        self.discount = torch.pow(self.mparam.beta, torch.arange(self.t_unroll)).to(self.device)
        self.policy_ds = None
    
    def prepare_state(self, input_data):
        state = torch.cat([input_data["basic_s"][..., 0:1], input_data["basic_s"][..., 2:]], dim=-1)
        gm = self.gm_model(input_data["agt_s"])
        state = torch.cat([state, gm], dim=-1)
        return state
    
    def policy_fn(self, input_data):
        state = self.prepare_state(input_data)
        policy = torch.sigmoid(self.model(state))
    
    def loss(self, input_data):
        raise NotImplementedError
    def get_valuedataset(self, update_init=False):
        raise NotImplementedError
    
    def train(self, n_epoch, batch_size=None):
        valid_data = {k: torch.tensor(self.init_ds.datadict[k], dtype=TORCH_DTYPE) for k in self.init_ds.keys}
        ashock, ishock = KS.simul_shocks(
            self.valid_size, self.t_unroll, self.mparam,
            state_init=self.init_ds.datadict
        )
        valid_data["ashock"] = torch.tensor(ashock, dtype=TORCH_DTYPE)
        valid_data["ishock"] = torch.tensor(ishock, dtype=TORCH_DTYPE)
        valid_data = {k: v.to(self.device) for k, v in valid_data.items()}
        
        update_init = False
        for n in tqdm(range(n_epoch), desc="Training Progress"):
            train_datasets = self.sampler(batch_size, update_init)
            for train_data in train_datasets:
                train_data = {key: value.to(self.device, dtype=TORCH_DTYPE) for key, value in train_data.items()}
                # トレーニングステップを実行
                self.optimizer.zero_grad()
                output_dict = self.loss(train_data)
                loss = output_dict["m_util"]
                loss.backward()
                self.optimizer.step()
            if n > 0 and n % 24 == 0:
                update_init = self.policy_config["update_init"]
                train_vds, valid_vds = self.get_valuedataset(update_init)
                for vtr in self.vtrainers:
                    vtr.train(
                        train_vds, valid_vds,
                        self.config["value_config"]["num_epoch"],
                        self.config["value_config"]["batch_size"]
                    )
    
class KTPolicyTrainer(PolicyTrainer):
    def __init__(self, vtrainers, init_ds, policy_path=None):
        super(KTPolicyTrainer, self).__init__(vtrainers, init_ds, policy_path)
        if self.config["init_with_bchmk"]:
            init_policy = self.init_ds.k_policy_bchmk
            policy_type = "pde"
        else:
            init_policy = self.init_ds.c_policy_const_share
            policy_type = "nn_share"
        self.policy_ds = self.init_ds.get_policydataset(init_policy, policy_type, update_init=False)
        
    def loss(self, input_data):
        k_cross = input_data["k_cross"]
        ashock = input_data["ashock"]
        price = input_data["price"]
        util_sum = 0
        for t in range(self.t_unroll):
            a_tmp = torch.repeat_interleave(ashock[:, t:t+1], 50, dim=1)
            a_tmp = torch.unsqueeze(a_tmp, 2)
            basic_s_tmp = torch.cat([torch.unsqueeze(k_cross, axis=-1), a_tmp], axis=-1)
            full_state_dict = {
                "basic_s": basic_s_tmp,
                "agt_s": self.init_ds.normalize_data(torch.unsqueeze(k_cross, axis=-1), key="agt_s", withtf=True)
            }
            if t == self.t_unroll - 1:
                value = 0
                for vtr in self.vtrainers:
                    value += self.init_ds.unnormalize_data(
                        vtr.value_fn(full_state_dict)[..., 0], key="value", withtf=True)
                value /= self.num_vnet
                util_sum += self.discount[t]*value
                continue
            
            price = price_fn(k_cross)
            wage = self.mparam.eta / price
            yterm = ashock[:, t] * k_cross[t, :]**self.mparam.theta
            n = (self.mparam.nu * yterm / wage)**(1 / (1 - self.mparam.nu))
            y = yterm * n**self.mparam.nu
            v0_temp = y - wage * n + (1 - self.mparam.delta) * k_cross
            v0 = v0_temp * price
            next_k = torch.argmax(value)
            
            k_cross = self.policy_fn(full_state_dict)
            
            e0 = 
    
    def expected_value(self, model, P):
        expec = model() @
        
            