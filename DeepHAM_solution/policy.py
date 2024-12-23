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
import matplotlib.pyplot as plt
from param import KTParam
import math


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

class CustomDataset(Dataset):
    def __init__(self, policy_ds):
        self.policy_ds = policy_ds
        self.keys = list(policy_ds.keys())
        self.data_length = len(policy_ds[self.keys[0]])  # データセットのサイズを取得

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        sample = {key: torch.tensor(self.policy_ds[key][idx]) for key in self.keys}
        return sample

class PriceDataset(Dataset):
    def __init__(self, basic_s):
        if isinstance(basic_s, np.ndarray):
            self.basic_s = torch.tensor(basic_s, dtype=TORCH_DTYPE)
        else:
            self.basic_s = basic_s
    
    def __len__(self):
        return len(self.basic_s)
    
    def __getitem__(self, idx):
        return self.basic_s[idx]

class PolicyTrainer():
    def __init__(self, vtrainers, init_ds, policy_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = init_ds.config
        self.policy_config = self.config["policy_config"]
        self.price_config = self.config["price_config"]
        self.t_unroll = self.policy_config["t_unroll"]
        self.valid_size = self.policy_config["valid_size"]
        self.sgm_scale = self.policy_config["sgm_scale"] # scaling param in sigmoid
        self.init_ds = init_ds
        self.vtrainers = vtrainers
        self.value_sampling = self.config["dataset_config"]["value_sampling"]
        self.num_vnet = len(vtrainers)
        self.decay_rate = self.policy_config["lr_end"] / self.policy_config["lr_beg"]
        self.mparam = init_ds.mparam
        self.n_sample_price = self.price_config["n_sample"]
        self.ashock_num = self.price_config["T"] * 384
        self.T_price = self.price_config["T"]
        d_in = self.config["n_basic"] + self.config["n_fm"] + self.config["n_gm"]
        self.policy = util.FeedforwardModel(2, 1, self.policy_config, name="p_net").to(self.device)
        self.policy_true = util.Policy(4, self.policy_config, "p_net_true").to(self.device)
        self.gm_model = util.GeneralizedMomModel(1, self.config["gm_config"], name="v_gm").to(self.device)
        self.gm_model_p = util.GeneralizedMomPrice(1, self.config["gm_config"], name="p_gm").to(self.device)
        self.price_model = util.PriceModel(2, 1, self.config["price_config"], name="price_net").to(self.device)
        self.policy_list = [self.policy, self.policy_true, self.gm_model]
        self.price_list = [self.price_model, self.gm_model_p]
        # 両方のモデルのパラメータを集める
        self.params = list(self.policy.parameters()) + list(self.gm_model.parameters())
        self.params_true = list(self.policy_true.parameters()) + list(self.gm_model.parameters())
        self.params_price = list(self.price_model.parameters()) + list(self.gm_model_p.parameters())
        self.optimizer = optim.Adam(
                self.params,
                lr=self.policy_config["lr_beg"],
                betas=(0.99, 0.99),
                eps=1e-8
            )
        self.optimizer_true = optim.Adam(
                self.params_true,
                lr=self.policy_config["lr_beg"],
                betas=(0.99, 0.99),
                eps=1e-8
            )
        self.optimizer_price = optim.Adam(
                self.params_price,
                lr=self.price_config["lr"],
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

    def set_requires_grad(self, models, requires_grad):
        """
        渡されたモデルのパラメーターの requires_grad を一括で設定します。

        Parameters:
            models (list): 対象のニューラルネットワークモデルのリスト。
            requires_grad (bool): パラメーターの requires_grad を True または False に設定するフラグ。
        """
        for model in models:
            for param in model.parameters():
                param.requires_grad = requires_grad

    
    def sampler(self, batch_size, init=None, update_init=False):
        if init is None:
            self.policy_ds = self.init_ds.get_policydataset(self.current_policy, "nn_share", self.price_fn, init=init, update_init=update_init)
        ashock, xi = KT.simul_shocks(self.ashock_num, self.t_unroll, self.mparam)
        self.policy_ds.datadict["ashock"] = torch.tensor(ashock, dtype=TORCH_DTYPE)
        self.policy_ds.datadict["xi"] = torch.tensor(xi, dtype=TORCH_DTYPE)
        dataset = CustomDataset(self.policy_ds.datadict)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return train_loader
        
        
        
    def prepare_state(self, input_data):
        state = torch.cat([input_data["basic_s"][..., 0:1], input_data["basic_s"][..., 2:]], dim=-1)
        gm = self.gm_model(input_data["agt_s"])
        state = torch.cat([state, gm], dim=-1)
        return state
    
    def prepare_state_policy(self, input_data):
        gm = self.gm_model(input_data["agt_s"])
        state = torch.cat([input_data["basic_s"], gm], dim=-1)
        return state
    
    def policy_fn(self, input_data):
        state = self.prepare_state_policy(input_data)
        policy = self.policy(state)#unnormalize_data必要と思う。
        return policy
    
    def policy_fn_true(self, input_data):
        state = self.prepare_state(input_data)
        policy = self.policy_true(state)#こっちもunnormalize_data必要と思う。
        return policy
    
    def loss(self, input_data):
        raise NotImplementedError
    
    def get_valuedataset(self, update_init=False):
        raise NotImplementedError
    
    def save_model(self, path="policy_model.pth"):
        torch.save(self.policy.state_dict(), path)
        torch.save(self.policy_true.state_dict(), path.replace(".pth", "_true.pth"))
        torch.save(self.gm_model.state_dict(), path.replace(".pth", "_gm.pth"))
        torch.save(self.price_model.state_dict(), path.replace(".pth", "_price.pth"))

        self.init_ds.save_stats(os.path.dirname(path))
    
    def train(self, n_epoch, batch_size=None):
        ashock, xi = KT.simul_shocks(
            self.valid_size, self.t_unroll, self.mparam,
            state_init=self.init_ds.datadict
        )
        init=True
        update_init = False
        for n in tqdm(range(n_epoch), desc="Training Progress"):
            with torch.no_grad():
                train_datasets = self.sampler(batch_size, init, update_init)
            init=None
            loss1_loop = 0.0
            loss2_loop = 0.0
            for train_data in train_datasets:
                train_data = {key: value.to(self.device, dtype=TORCH_DTYPE) for key, value in train_data.items()}
                # トレーニングステップを実行
                self.optimizer.zero_grad()
                output_dict = self.loss1(train_data)
                loss1 = output_dict["m_util"]
                loss1.backward()
                self.optimizer.step()
                self.optimizer_true.zero_grad()
                output_dict2 = self.loss2(train_data)
                loss2 = output_dict2["m_util"]
                loss2.backward()
                self.optimizer_true.step()#この後にpriceの学習入れるべきじゃない？
                loss1_loop += -loss1.item()
                loss2_loop += -loss2.item()
            avg_train_loss1 = loss1_loop / len(train_datasets)
            avg_train_loss2 = loss2_loop / len(train_datasets)
            print(f"Epoch {n+1}: Training Loss1 = {avg_train_loss1:.6f}, Training Loss2 = {avg_train_loss2:.6f}")
            update_init = self.policy_config["update_init"]
            #update_frequency = min(25, max(3, int(math.sqrt(n + 1))))
            #if n > 0 and n % update_frequency == 0:
            if n > 0 and n % 3 == 0:
                self.set_requires_grad(self.policy_list, False)
                self.optimizer_price = torch.optim.Adam(self.params_price, lr=self.price_config["lr"])
                self.price_loss_training_loop(self.n_sample_price, self.price_config["T"], self.mparam, self.current_policy, "nn_share", self.price_fn, self.optimizer_price, batch_size=256,  num_epochs=10, validation_size=64, threshold=1e-5)
                train_vds, valid_vds = self.get_valuedataset(init=init, update_init=update_init)
                self.set_requires_grad(self.policy_list, True)
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
        data_stats = KT.create_stats_init(384, 10, self.mparam, init_ds.policy_init_only, policy_type, self.price_model)
        init_ds.update_stats(data_stats, key="basic_s", ma=1)
        init_ds.stats_dict["agt_s"], init_ds.stats_dict_tf["agt_s"] = [x[0] for x in init_ds.stats_dict["basic_s"]], [x[0] for x in init_ds.stats_dict_tf["basic_s"]]
        #init_ds.stats_dict["value"], init_ds.stats_dict_tf["value"] = (5, 2), (torch.tensor(5, dtype=TORCH_DTYPE), torch.tensor(2, dtype=TORCH_DTYPE))
        self.price_loss_training_loop(self.n_sample_price, self.price_config["T"], self.mparam, KT.init_policy_fn_tf, "nn_share", self.price_fn, self.optimizer_price, batch_size=256, init=True, state_init=None, shocks=None, num_epochs=3) #self.price_config["T"]
        self.policy_ds = self.init_ds.get_policydataset(init_ds.policy_init_only, policy_type, self.price_fn, init=True, update_init=False)
        

    def create_data(self, input_data):
        k_cross = input_data["k_cross"]
        ashock = input_data["ashock"]
        for t in range(self.t_unroll):
            a_tmp = torch.repeat_interleave(ashock[:, t:t+1],self.mparam.n_agt, dim=1)
            a_tmp = torch.unsqueeze(a_tmp, 2)
            basic_s_tmp = torch.cat([torch.unsqueeze(k_cross, axis=-1), a_tmp], axis=-1)
            full_state_dict = {
                "basic_s": basic_s_tmp,
                "agt_s": self.init_ds.normalize_data(torch.unsqueeze(k_cross, axis=-1), key="agt_s", withtf=True)
            }
        return full_state_dict

    def loss1(self, input_data): #vを最大にするpolicyを学習するためのlossを計算。
        k_cross = input_data["k_cross"]
        ashock = input_data["ashock"]
        xi = input_data["xi"]
        util_sum = 0
        for t in range(self.t_unroll):
            k_tmp = torch.unsqueeze(k_cross, 2)
            a_tmp = torch.repeat_interleave(ashock[:, t:t+1], self.mparam.n_agt, dim=1)#samplerで作成したbatch_size, self.t_unrollのashockを使っている。
            a_tmp = torch.unsqueeze(a_tmp, 2)
            xi_tmp = xi[:,:,t:t+1]
        
            if t == self.t_unroll - 1:
                price = self.price_fn(k_cross_pre, ashock[:, t:t+1])
                k_mean_tmp = torch.mean(k_tmp, dim=1, keepdim=True).repeat(1, self.mparam.n_agt,1)
                basic_s_tmp_v = torch.cat([k_tmp, k_mean_tmp, a_tmp, xi_tmp], axis=-1)
                basic_s_v = self.init_ds.normalize_data(basic_s_tmp_v, key="basic_s", withtf=True)
                full_state_dict_v = {
                    "basic_s": basic_s_v,
                    "agt_s": self.init_ds.normalize_data(k_tmp, key="agt_s", withtf=True)
                }
                value = 0
                for vtr in self.vtrainers:
                    value += self.init_ds.unnormalize_data(
                        vtr.value_fn(full_state_dict_v)[..., 0], key="value", withtf=True)
                value /= self.num_vnet
                util_sum += -self.mparam.GAMY*price * k_cross + self.mparam.BETA * value
                continue
            basic_s_tmp = self.init_ds.normalize_data_ashock(a_tmp, key="basic_s", withtf=True)
            full_state_dict = {
                "basic_s": basic_s_tmp,
                "agt_s": self.init_ds.normalize_data(k_tmp, key="agt_s", withtf=True)
            }
            k_cross_pre = k_cross
            k_cross = self.init_ds.unnormalize_data_k_cross(self.policy_fn(full_state_dict), key="basic_s", withtf=True).clamp(min=0.01).squeeze(-1)

        output_dict = {"m_util": -torch.mean(util_sum[:, 0]), "k_end": torch.mean(k_cross)}
        #print(f"loss1:{-output_dict['m_util']}")
        return output_dict
            


    def loss2(self, input_data):
        k_cross = input_data["k_cross"]
        ashock = input_data["ashock"]
        xi = input_data["xi"]
        k_tmp = torch.unsqueeze(k_cross, 2)     
        k_mean = torch.mean(k_tmp, dim=1, keepdim=True).repeat(1, self.mparam.n_agt, 1)
        ashock_tmp = torch.repeat_interleave(ashock[:, 0:1], self.mparam.n_agt, dim=1)
        ashock_tmp = torch.unsqueeze(ashock_tmp, 2)
        next_k = self.current_policy(k_tmp, ashock[:,0:1], xi[:,:,0:1], withtf=True)
        next_k_v = torch.where(next_k==(1-self.mparam.delta)*k_tmp, next_k/self.mparam.GAMY, next_k)
        next_mean = torch.mean(next_k_v, dim=1, keepdim=True).repeat(1, self.mparam.n_agt, 1)
        ashock_next = torch.repeat_interleave(ashock[:, 1:2], self.mparam.n_agt, dim=1)
        ashock_next = torch.unsqueeze(ashock_next, 2)
        basic_s_tmp = torch.cat([next_k_v, next_mean, ashock_next, xi[:,:,1:2]], axis=-1)
        basic_s = self.init_ds.normalize_data(basic_s_tmp, key="basic_s", withtf=True)
        full_state_dict = {
            "basic_s": basic_s,
            "agt_s": self.init_ds.normalize_data(next_k_v, key="agt_s", withtf=True)
        }
        next_value = 0
        for vtr in self.vtrainers:
            next_value += self.init_ds.unnormalize_data(
                vtr.value_fn(full_state_dict)[..., 0], key="value", withtf=True)
        next_value /= self.num_vnet
        next_value = next_value.squeeze(-1)
        price = self.price_fn(k_cross, ashock[:,0:1])
        wage = self.mparam.eta / price
        yterm = ashock[:,0:1]*k_cross**self.mparam.theta
        n = (self.mparam.nu*yterm/wage)**(1/(1-self.mparam.nu))
        y = yterm*n**self.mparam.nu
        profit = (y - wage*n + (1-self.mparam.delta)*k_cross)*price
        changed = profit-xi[:,:,0]*wage*price - self.mparam.GAMY * price * next_k.squeeze(-1) + self.mparam.BETA * next_value
        unchanged = profit - price * (1-self.mparam.delta) * k_cross + self.mparam.BETA * next_value        
        util_sum = torch.where(next_k.squeeze(-1)==(1-self.mparam.delta)*k_cross, unchanged, changed)
        output_dict = {"m_util": -torch.mean(util_sum[:, 0]), "k_end": torch.mean(k_cross)}
        return output_dict
        

    def loss22(self, input_data):
        k_cross = input_data["k_cross"]
        ashock = input_data["ashock"]
        xi = input_data["xi"]
        for t in range(self.t_unroll):
            k_tmp = torch.unsqueeze(k_cross, 2)
            k_mean_tmp = torch.mean(k_tmp, dim=1, keepdim=True).repeat(1, 50, 1)
            a_tmp = torch.repeat_interleave(ashock[:, t:t+1], 50, dim=1)#samplerで作成したbatch_size, self.t_unrollのashockを使っている。
            a_tmp = torch.unsqueeze(a_tmp, 2)
            xi_tmp = xi[:,:,t:t+1]
            
            if t == self.t_unroll - 1:
                value0 = 0
                value1 = 0
                price = self.price_fn(k_cross_pre, ashock[:,t:t+1]) #ここおかしいかも。input_dataのpriceを使うべきか？
                k_mean_pre = torch.mean(k_cross_pre, dim=1, keepdim=True).unsqueeze(-1).repeat(1, 50, 1)
                
                basic_s_tmp_pre = torch.cat([k_tmp, k_mean_tmp,a_tmp, xi_tmp], axis=-1)
                basic_s_tmp_e0 = self.init_ds.normalize_data(basic_s_tmp_pre, key="basic_s", withtf=True)
                full_state_dict_e0 = {
                    "basic_s": basic_s_tmp_e0,
                    "agt_s": self.init_ds.normalize_data(k_tmp, key="agt_s", withtf=True)
                }
                wage = self.mparam.eta / price#この下のvalueはk_cross, gmも必要。
                for vtr in self.vtrainers:
                    value0 += self.init_ds.unnormalize_data(
                        vtr.value_fn(full_state_dict_e0)[..., 0], key="value", withtf=True)
                value0 /= self.num_vnet
                e0 = -xi[:,:,t]*wage*price - self.mparam.GAMY * price * k_cross + self.mparam.BETA * value0
                #a_tmp = torch.repeat_interleave(ashock[:, t:t+1], 50, dim=1)#samplerで作成したbatch_size, self.t_unrollのashockを使っている。
                #a_tmp = torch.unsqueeze(a_tmp, 2)
                k_pre = ((1-self.mparam.delta) / self.mparam.GAMY) * k_cross_pre 
                basic_s_tmp = torch.cat([torch.unsqueeze(k_pre, axis=-1), k_mean_pre, a_tmp, xi_tmp], axis=-1)#ここもおかしいかもpreのmeanにすべき
                basic_s_tmp_e1 = self.init_ds.normalize_data(basic_s_tmp, key="basic_s", withtf=True)
                full_state_dict_e1 = {
                    "basic_s": basic_s_tmp,
                    "agt_s": self.init_ds.normalize_data(k_tmp, key="agt_s", withtf=True)
                }
                for vtr in self.vtrainers:
                    value1 += self.init_ds.unnormalize_data(
                        vtr.value_fn(full_state_dict_e1)[..., 0], key="value", withtf=True)
                value1 /= self.num_vnet
                e1 = -price * (1-self.mparam.delta) * k_cross_pre + self.mparam.BETA * value1
                
                basic_s_policy_tmp = torch.cat([torch.unsqueeze(k_cross_pre, axis=-1), k_mean_pre, a_pre, xi_pre], axis=-1)
                basic_s_policy = self.init_ds.normalize_data(basic_s_policy_tmp, key="basic_s", withtf=True)
                full_state_dict_loss = {
                    "basic_s": basic_s_policy,
                    "agt_s": self.init_ds.normalize_data(k_cross_pre.unsqueeze(-1), key="agt_s", withtf=True)
                }
                loss_fn = nn.BCELoss()
                bigger = torch.where(e0>=e1, torch.zeros_like(e0, dtype=TORCH_DTYPE), torch.ones_like(e1, dtype=TORCH_DTYPE))
                target = self.policy_fn_true(full_state_dict_loss).squeeze(-1)
                loss = loss_fn(target, bigger)
                continue
            
            basic_s_tmp = self.init_ds.normalize_data_ashock(a_tmp, key="basic_s", withtf=True)
            full_state_dict = {
                "basic_s": basic_s_tmp,
                "agt_s": self.init_ds.normalize_data(torch.unsqueeze(k_cross, axis=-1), key="agt_s", withtf=True)
            }
            a_pre = a_tmp
            xi_pre = xi_tmp
            k_cross_pre = k_cross
            k_cross = self.init_ds.unnormalize_data_k_cross(self.policy_fn(full_state_dict), key="basic_s", withtf=True).clamp(min=0.01).squeeze(-1)
        #print(f"loss2:{loss}")
        return loss
    

    def sampler_p(self, batch_size, init=None):
        if init is None:
            self.price_ds = self.init_ds.get_pricedataset(self.current_policy, "nn_share", self.price_fn, init=init)
        else:
            self.price_ds = self.init_ds.get_pricedataset(self.init_ds.policy_init_only, "nn_share", self.price_fn, init=init)
            
        dataset = CustomDataset(self.price_ds.datadict)
        
        # Split the dataset into training and validation sets
        valid_size = 192
        train_size = len(dataset) - valid_size
        train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
        
        # Create DataLoaders for training and validation
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
        
        return train_loader, valid_loader

    def price_loss_training_loop(
            self,
            n_sample,
            T,
            mparam,
            policy_fn,
            policy_type,
            price_fn,
            optimizer,
            batch_size=256,
            init=None,
            state_init=None,
            shocks=None,
            num_epochs=3,
            validation_size=32,  # Added parameter for validation size
            threshold = 1e-6
        ):
        # ロスを保存するリスト
        train_losses = []
        val_losses = []
        epoch = 0
        avg_val_loss = 1.0
        if init is not None:
            loss_fn = self.loss_price_init
        else:
            loss_fn = self.loss_price
        # エポックループの追加
        with torch.no_grad():
            train_loader, val_loader = self.sampler_p(batch_size, init)
        while avg_val_loss > threshold and epoch < num_epochs: #avg_val_loss > threshold or epoch < num_epochs:
            epoch += 1
            epoch_train_loss = 0.0

            # トレーニングフェーズ
            self.price_model.train()  # Set model to training mode
            self.gm_model_p.train()
            for data in train_loader:
                data = {key: value.to(self.device, dtype=TORCH_DTYPE) for key, value in data.items()}
                
                optimizer.zero_grad()

                # ロス関数の計算
                loss = loss_fn(data, policy_fn, price_fn, mparam)

                # 損失の逆伝播と最適化
                loss.backward()
                optimizer.step()

                # ロスの累積
                epoch_train_loss += loss.item()

            # トレーニングの平均ロス
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # バリデーションフェーズ
            self.price_model.eval()  # Set model to evaluation mode
            self.gm_model_p.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for val_data in val_loader:
                    val_data = {key: value.to(self.device, dtype=TORCH_DTYPE) for key, value in val_data.items()}
                    
                    # ロス関数の計算
                    val_loss = loss_fn(val_data, policy_fn, price_fn, mparam)
                    
                    # ロスの累積
                    epoch_val_loss += val_loss.item()

            # バリデーションの平均ロス
            avg_val_loss = epoch_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            # ロスの出力
            print(f"  Training Loss: {avg_train_loss:.10f} | Validation Loss: {avg_val_loss:.10f}")
            
            
        print("トレーニング完了")
        for epoch_num, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), start=1):  
            print(f"Epoch {epoch_num}: Training Loss = {train_loss:.10f}, Validation Loss = {val_loss:.10f}")

        # Optionally, return the losses for further analysis
        return train_losses, val_losses



    
    def loss_price(self, data, policy_fn, price_fn, mparam):
        ashock = data["ashock"]
        k_cross = data["k_cross"]
        xi = data["xi"]
        k_tmp = k_cross.unsqueeze(2)
        xi_tmp = xi.unsqueeze(2)
        price = price_fn(k_cross, ashock)
        wage = mparam.eta / price

        yterm = ashock * k_cross ** mparam.theta
        n = (mparam.nu * yterm / wage)**(1/(1-mparam.nu))#最右項は*2
        k_new = policy_fn(k_tmp, ashock, xi_tmp, withtf=True).clamp(min=0.01).squeeze(2)
        inow = mparam.GAMY * k_new - (1 - mparam.delta) * k_cross
        ynow = ashock * k_cross**mparam.theta * (n**mparam.nu)
        Cnow = ynow.mean(dim=1, keepdim=True) - inow.mean(dim=1, keepdim=True)
        Cnow = Cnow.clamp(min=0.1)
        #print(f"k_cross:{k_cross[0,0]}, price:{price[0,0]}, yterm:{yterm[0,0]}, Cnow:{Cnow[0,0]}")
        price_target = 1 / Cnow
        mse_loss_fn = nn.HuberLoss()
        loss = mse_loss_fn(price, price_target)
        return loss


    def loss_price_init(self, data, policy_fn, price_fn, mparam):
        ashock = data["ashock"]
        k_cross = data["k_cross"]
        k_mean = torch.mean(k_cross, dim=1, keepdim=True)
        price = price_fn(k_cross, ashock)
        wage = mparam.eta / price

        yterm = ashock * k_cross ** mparam.theta
        n = (mparam.nu * yterm / wage)**(1/(1-mparam.nu))
        k_tmp = k_cross.unsqueeze(2)#128,50,1
        a_tmp = ashock.repeat(1, self.mparam.n_agt).unsqueeze(2)#128,50,1

        k_new = policy_fn(self.init_ds.policy_init_only ,k_tmp, k_mean, ashock).squeeze(2)
        inow = mparam.GAMY * k_new - (1 - mparam.delta) * k_cross
        ynow = ashock * k_cross**mparam.theta * (n**mparam.nu)
        Cnow = ynow.mean(dim=1, keepdim=True) - inow.mean(dim=1, keepdim=True)
        Cnow = Cnow.clamp(min=0.1)
        price_target = 1 / Cnow
        mse_loss_fn = nn.HuberLoss()
        loss = mse_loss_fn(price, price_target)
        return loss



        

    
    def current_policy(self, k_cross, ashock, xi, withtf=False):
        if withtf:
            k_mean = torch.mean(k_cross, dim=1, keepdim=True).repeat(1, self.mparam.n_agt, 1)
            ashock = ashock.repeat(1, self.mparam.n_agt).unsqueeze(2)
            basic_s = torch.cat([k_cross, k_mean, ashock, xi], dim=2)
            basic_p = self.init_ds.normalize_data_ashock(ashock, key="basic_s", withtf=True)
            
        else:
            k_mean = np.mean(k_cross, axis=1, keepdims=True)  # NumPy: 形状 (384, 1, 1)
            k_mean = np.repeat(k_mean, self.mparam.n_agt, axis=1)  # NumPy: 形状 (384, 50, 1)

            # ashockもNumPyで操作
            ashock = np.repeat(ashock, self.mparam.n_agt, axis=1)[:, :, np.newaxis]  # NumPy: 形状 (384, 50, 1)

            # k_cross, k_mean, ashockを結合 (NumPyで)
            basic_s = np.concatenate([k_cross, k_mean, ashock, xi], axis=-1)  # NumPy: 形状 (384, 50, X)

            # NumPy配列をTorchテンソルに変換
            basic_s = torch.tensor(basic_s, dtype=TORCH_DTYPE).to(self.device)  # Torch: 形状 (384, 50, X)
            basic_p = torch.tensor(ashock, dtype=TORCH_DTYPE).to(self.device)  # Torch: 形状 (384, 50, 1)
            # k_crossも同様にTorchテンソルに変換
            k_cross = torch.tensor(k_cross, dtype=TORCH_DTYPE).to(self.device)  # Torch: 形状 (384, 50)
        basic_s = self.init_ds.normalize_data(basic_s, key="basic_s", withtf=True)
        basic_p = self.init_ds.normalize_data_ashock(basic_p, key="basic_s", withtf=True)
        agt_s = self.init_ds.normalize_data(k_cross, key="agt_s", withtf=True)
        
        full_state_dict = {
            "basic_s": basic_s,
            "agt_s": agt_s
        }
        full_state_dict_p = {
            "basic_s": basic_p,
            "agt_s": agt_s
        }
        
        decision = self.policy_fn_true(full_state_dict)
        op_k = self.init_ds.unnormalize_data_k_cross(self.policy_fn(full_state_dict_p), key="basic_s", withtf=True)
        output = torch.where(decision > 0.5, (1-self.mparam.delta) * k_cross, op_k)
        return output
    
    def get_valuedataset(self, update_from=None, init=None, update_init=False):
        return self.init_ds.get_valuedataset(self.current_policy, "nn_share", self.price_fn, update_from, init, update_init)
    
    def init_policy_fn_tf(self, k_cross, k_mean, ashock):
        # PyTorchで処理する
        k_mean_tmp = k_mean.repeat(1, self.mparam.n_agt).unsqueeze(2)  # axis=1をPyTorchで再現
        ashock_tmp = ashock.repeat(1, self.mparam.n_agt).unsqueeze(2)  # axis=1をPyTorchで再現
        basic_s = torch.cat([k_cross, ashock_tmp, k_mean_tmp], dim=2)  # NumPyのconcatenateをtorch.catで再現
        
        # GPUでNNの計算を実行
        basic_s_torch = basic_s.to("cuda")  # GPUに移動
        output_torch = self.init_ds.policy_init_only(basic_s_torch)
        
        # 結果をそのまま返す（必要ならCPUに移動してNumPyに変換）
        output = output_torch  # 必要に応じて .to('cpu') を付けてCPUに戻す

        return output
    
    def price_fn(self, k_cross, ashock):#どっちもunsqueezeする→k_crossだけ
        if isinstance(k_cross, np.ndarray):
            k_cross = torch.tensor(k_cross, dtype=TORCH_DTYPE)  # ndarrayをTensorに変換
            ashock = torch.tensor(ashock, dtype=TORCH_DTYPE)  # ndarrayをTensorに変換
            
        k_cross = k_cross.unsqueeze(2).to(self.device)
        ashock = ashock.to(self.device)
        #input_data = input_data.to(self.device)
        #price_data = torch.mean(input_data, dim=1, keepdim=True)
        k_tmp = self.init_ds.normalize_data(k_cross, key="agt_s", withtf=True)
        a_tmp = self.init_ds.normalize_data_ashock(ashock, key="basic_s", withtf=True)
        gm_data = self.gm_model_p(k_tmp)
        state = torch.cat([gm_data, a_tmp], dim=1)
        price = self.price_model(state)
        return price

    #def price_fn(self, input_data):
        #price_data = self.init_ds.normalize_data_price(data_tmp, key="basic_s", withtf=True)
        #price = self.init_ds.unnormalize_data_k_cross(self.price_model(price_data), key="basic_s", withtf=True).clamp(min=0.01)
        #return price
    
    def get_bin_edges(self, num_bins=50, min_val=0.0, max_val=3.0):
        return torch.linspace(min_val, max_val, steps=num_bins+1)
            
    def assign_bins(self, k_cross, bin_edges):
        bin_indices = torch.bucketize(k_cross, bin_edges, right=False) - 1  # shape: (batch_size, num_agents)
        bin_indices = torch.clamp(bin_indices, 0, len(bin_edges) - 2) 
        return bin_indices
    
    def count_bins(self, bin_indices, num_bins=50):
        """
        bin_indices: Tensor of shape (batch_size, num_agents)
        """
        # ワンホットエンコーディング
        one_hot = F.one_hot(bin_indices, num_classes=num_bins)  # shape: (batch_size, num_agents, num_bins)
        
        # エージェント数をカウント
        bin_counts = one_hot.sum(dim=1).float()  # shape: (batch_size, num_bins)
        
        return bin_counts
    
    def prepare_price_input(self, k_cross, ashock, num_bins=50):
        """
        k_cross: Tensor of shape (batch_size, num_agents)
        ashock: Tensor of shape (batch_size, 1)
        """
        k_cross = torch.tensor(k_cross, dtype=TORCH_DTYPE).to(self.device)
        ashock = torch.tensor(ashock, dtype=TORCH_DTYPE).to(self.device)
        # ビンエッジを定義
        bin_edges = self.get_bin_edges().to(self.device)
        
        # ビンに割り当て
        bin_indices = self.assign_bins(k_cross, bin_edges)
        
        # ビンごとのカウント
        bin_counts = self.count_bins(bin_indices, num_bins)
        
        # ashockと結合
        price_input = torch.cat([bin_counts, ashock], dim=1)  # shape: (batch_size, num_bins + 1)
        
        price = self.price_model(price_input)  # shape: (batch_size, 1)
        return price
    
    def value_simul_k(self, k_cross, k_mean, ashock):
        k_cross = torch.tensor(k_cross, dtype=TORCH_DTYPE).to(self.device)
        k_mean = torch.tensor(k_mean, dtype=TORCH_DTYPE).to(self.device)
        ashock = torch.tensor(ashock, dtype=TORCH_DTYPE).to(self.device)
        k_tmp = k_cross.unsqueeze(2)
        k_mean_tmp = k_mean.repeat(1, 50).unsqueeze(2)
        a_tmp = ashock.repeat(1, 50).unsqueeze(2)
        basic_s_tmp = torch.cat([k_tmp, k_mean_tmp, a_tmp], dim=2)
        basic_s = self.init_ds.normalize_data(basic_s_tmp, key="basic_s", withtf=True)
        full_state_dict = {
            "basic_s": basic_s,
            "agt_s": self.init_ds.normalize_data(k_tmp, key="agt_s", withtf=True)
        }
        value = 0
        for vtr in self.vtrainers:
            value += self.init_ds.unnormalize_data(
                vtr.value_fn(full_state_dict)[..., 0], key="value", withtf=True)
        value /= self.num_vnet
        return value
        
# value, policyが学習されないようにする必要あり。
# 真のpolicyがalphaを考慮してるからここでは真のpolicyを流してよさそう。
#k_crossを384, 50, 32にして32*384, 50
#↑いや384, 50, 500にしてシャッフルしてやるわ。

    # def price_loss1(n_sample, T, mparam, policy, policy_type, price_fn, state_init=None, shocks=None):
    #     if shocks:
    #         ashock = shocks
    #     
    #         assert n_sample == ashock.shape[0], "n_sample is inconsistent with given shocks."
    #         assert T == ashock.shape[1], "T is inconsistent with given shocks."
    #         if state_init:
    #             assert np.array_equal(ashock[..., 0:1], state_init["ashock"]) and \
    #                 "Shock inputs are inconsistent with state_init"
    #     else:
    #         ashock = simul_shocks(n_sample, T, mparam, state_init)
    #     
    #     k_cross = np.zeros([n_sample, n_agt, T])
    #     if state_init:
    #         assert n_sample == state_init["k_cross"].shape[0], "n_sample is inconsistent with state_init."
    #         k_cross[:, :, 0] = state_init["k_cross"]
    #     else:
    #         k_cross[:, :, 0] = mparam.k_ss
    #         
    #     if policy_type == "nn":
    #         for t in range(1, T):
    #             price = price_fn(k_cross[:, :, t-1])# 384*1
    #             wage = mparam.eta / price # 384*1
    #             yterm = ashock[:, t-1] * k_cross[:, :, t-1]**mparam.theta # 384*50
    #             n = (mparam.nu * yterm / wage)**(1 / (1 - mparam.nu))
    #             y = yterm * n**mparam.nu
    #             k_cross_pre = k_cross[:, :, t-1]
    #             k_cross[:, :, t] = policy(k_cross[:, :, t - 1], ashock[:, t - 1])# 384*50
    #             inow = mparam.GAMY * k_cross - (1 - mparam.delta) * k_cross_pre
    #             ynow = ashock[:, t-1] * k_cross_pre**mparam.theta * n**mparam.nu
    #             
    #         Inow = torch.sum(inow, axis=1)
    #         Ynow = torch.sum(ynow, axis=1)
    #         Cnow = Ynow - Inow
    #         price1 = 1 / Cnow 
    #         loss = torch.mean((price - price1)**2)
    #
    #     return  loss
            
        
# def price_loss(n_sample, T, mparam, policy, policy_type, price_fn, state_init=None, shocks=None):
#     if shocks:
#         ashock = shocks
#     
#         assert n_sample == ashock.shape[0], "n_sample is inconsistent with given shocks."
#         assert T == ashock.shape[1], "T is inconsistent with given shocks."
#         if state_init:
#             assert np.array_equal(ashock[..., 0:1], state_init["ashock"]) and \
#                 "Shock inputs are inconsistent with state_init"
#     else:
#         ashock = simul_shocks(n_sample, T, mparam, state_init)
#     
#     k_cross = np.zeros([n_sample, n_agt, T])
#     if state_init:
#         assert n_sample == state_init["k_cross"].shape[0], "n_sample is inconsistent with state_init."
#         k_cross[:, :, 0] = state_init["k_cross"]
#     else:
#         k_cross[:, :, 0] = mparam.k_ss
#     
#     if policy_type == "nn":
#         for t in range(1, T):
#             price = price_fn(k_cross[:, :, t-1])# 384*1
#             wage = mparam.eta / price # 384*1
#             yterm = ashock[:, t-1] * k_cross[:, :, t-1]**mparam.theta # 384*50
#             n = (mparam.nu * yterm / wage)**(1 / (1 - mparam.nu))
#             y = yterm * n**mparam.nu
#             k_cross_pre = k_cross[:, :, t-1]
#             k_cross[:, :, t] = policy(k_cross[:, :, t - 1], ashock[:, t - 1])# 384*50
#             a_tmp = torch.repeat_interleave(ashock[:, t:t+1], 50, dim=1)#samplerで作成したbatch_size, self.t_unrollのashockを使っている。
#             a_tmp = torch.unsqueeze(a_tmp, 2)
#             basic_s_tmp = torch.cat([torch.unsqueeze(k_cross[:,:,t], axis=-1), a_tmp], axis=-1)
#             full_state_dict = {
#                 "basic_s": basic_s_tmp,
#                 "agt_s": self.init_ds.normalize_data(torch.unsqueeze(k_cross[:,:,t], axis=-1), key="agt_s", withtf=True)
#             }
#             e0 = -mparam.GAMY * price * k_cross + mparam.BETA * value(full_state_dict)
#             basic_s_tmp_e1 = torch.cat([torch.unsqueeze(k_cross_pre, axis=-1), a_tmp], axis=-1)
#             full_state_dict_e1 = {
#                 "basic_s": basic_s_tmp_pre,
#                 "agt_s": self.init_ds.normalize_data(torch.unsqueeze(k_cross_pre, axis=-1), key="agt_s", withtf=True)
#             }
#             e1 = mparam.p * (1 - mparam.delta) * k_cross_pre + mparam.BETA * value(full_state_dict_e1)
#             xitemp = (e0 - e1)/(price * wage)
#             xi = min(B, max(0, xitemp))
#             alpha = xi / B
#             inow = alpha * (mparam.GAMY * k_cross - (1 - mparam.delta) * k_cross_pre)
#             ynow = ashock[:, t-1] * k_cross_pre**mparam.theta * n**mparam.nu
#             nnow = n + xi**2/(2*B)
#         
#         Inow = torch.sum(inow, axis=1)
#         Ynow = torch.sum(ynow, axis=1)
#         Cnow = Ynow - Inow
#         price1 = 1 / Cnow #n_sample, Tになってて欲しい。
#         loss = torch.mean((price - price1)**2)
# 
#     return  loss