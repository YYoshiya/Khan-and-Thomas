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

class CustomDataset(Dataset):
    def __init__(self, policy_ds):
        self.policy_ds = policy_ds
        self.keys = list(policy_ds.keys())  # データのキー (e.g., "k_cross", "ashock", "ishock")
        self.data_length = len(policy_ds[self.keys[0]])  # データセットのサイズを取得

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):
        sample = {key: torch.tensor(self.policy_ds[key][idx]) for key in self.keys}
        return sample

class PolicyTrainer():
    def __init__(self, vtrainers, init_ds, policy_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = init_ds.config
        self.policy_config = self.config["policy_config"]
        self.price_config = self.config["price_config"]
        self.t_unroll = self.policy_config["t_unroll"]
        self.value = util.FeedforwardModel(d_in, 1, self.policy_config, name="value_net").to(self.device)
        self.valid_size = self.policy_config["valid_size"]
        self.sgm_scale = self.policy_config["sgm_scale"] # scaling param in sigmoid
        self.init_ds = init_ds
        self.value_sampling = self.config["dataset_config"]["value_sampling"]
        self.num_vnet = len(vtrainers)
        self.decay_rate = self.policy_config["lr_end"] / self.policy_config["lr_beg"]
        self.mparam = init_ds.mparam
        self.n_sample_price = self.price_config["n_sample"]
        self.T_price = self.price_config["T"]
        d_in = self.config["n_basic"] + self.config["n_fm"] + self.config["n_gm"]
        self.policy = util.FeedforwardModel(d_in, 1, self.policy_config, name="p_net").to(self.device)
        self.policy_true = util.FeedforwardModel(d_in, 1, self.policy_config, name="p_net_true").to(self.device)
        self.gm_model = util.GeneralizedMomModel(1, self.config["n_gm"], self.config["gm_config"], name="v_gm").to(self.device)
        self.price_model = util.PriceModel(1, 1, self.policy["price_config"], name="price_net").to(self.device)
        # 両方のモデルのパラメータを集める
        params = list(self.policy.parameters()) + list(self.gm_model.parameters())
        params_true = list(self.policy_true.parameters()) + list(self.gm_model.parameters())
        self.optimizer = optim.Adam(
                params,
                lr=self.policy_config["lr_beg"],
                betas=(0.99, 0.99),
                eps=1e-8
            )
        self.optimizer_true = optim.Adam(
                params_true,
                lr=self.policy_config["lr_beg"],
                betas=(0.99, 0.99),
                eps=1e-8
            )
        self.optimizer_price = optim.Adam(
                self.price_model.parameters(),
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
    
    def sampler(self, batch_size, update_init=False):
        self.policy_ds = self.init_ds.get_policydataset(self.current_c_policy, "nn_share", self.price_model, update_init)
        dataset = CustomDataset(self.policy_ds.datadict)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        first_batch = next(iter(train_loader))
        ashock = KT.simul_shocks(batch_size, self.t_unroll, self.mparam.Z, self.mparam.Pi, state_init=None)
        new_data = {
            "k_cross": first_batch["k_cross"],
            "ashock": torch.tensor(ashock, dtype=TORCH_DTYPE)
        }
        
        for train_data in train_loader:
            ashock = KT.simul_shocks(batch_size, self.t_unroll, self.mparam.Z, self.mparam.Pi, state_init=None)
            new_data["k_cross"] = torch.cat([new_data["k_cross"], train_data["k_cross"]], dim=0)
            new_data["ashock"] = torch.cat([new_data["ashock"], torch.tensor(ashock, dtype=TORCH_DTYPE)], dim=0)
        
        new_dataset = CustomDataset(new_data)
        new_train_loader = torch.utils.data.DataLoader(new_dataset, batch_size=batch_size, shuffle=True)
        return new_train_loader
        
        
        
    def prepare_state(self, input_data):
        state = torch.cat([input_data["basic_s"][..., 0:1], input_data["basic_s"][..., 2:]], dim=-1)
        gm = self.gm_model(input_data["agt_s"])
        state = torch.cat([state, gm], dim=-1)
        return state
    
    def policy_fn(self, input_data):
        state = self.prepare_state(input_data)
        policy = torch.sigmoid(self.policy(state))
    
    def policy_fn_true(self, input_data):
        state = self.prepare_state(input_data)
        policy = torch.sigmoid(self.policy_true(state))
    
    def loss(self, input_data):
        raise NotImplementedError
    
    def get_valuedataset(self, update_init=False):
        raise NotImplementedError
    
    def train(self, n_epoch, batch_size=None):
        valid_data = {k: torch.tensor(self.init_ds.datadict[k], dtype=TORCH_DTYPE) for k in self.init_ds.keys}
        ashock = KT.simul_shocks(
            self.valid_size, self.t_unroll, self.mparam,
            state_init=self.init_ds.datadict
        )
        valid_data["ashock"] = torch.tensor(ashock, dtype=TORCH_DTYPE)
        valid_data = {k: v.to(self.device) for k, v in valid_data.items()}
        
        update_init = False
        for n in tqdm(range(n_epoch), desc="Training Progress"):
            train_datasets = self.sampler(batch_size, update_init)
            for train_data in train_datasets:
                train_data = {key: value.to(self.device, dtype=TORCH_DTYPE) for key, value in train_data.items()}
                # トレーニングステップを実行
                self.optimizer.zero_grad()
                loss1 = self.loss1(train_data)
                loss1.backward()
                self.optimizer.step()
            for train_data in train_datasets:#データセットはシャッフルしなくてよい？
                train_data = {key: value.to(self.device, dtype=TORCH_DTYPE) for key, value in train_data.items()}
                # トレーニングステップを実行
                self.optimizer_true.zero_grad()
                loss2 = self.loss2(train_data)
                loss2.backward()
                self.optimizer_true.step()#この後にpriceの学習入れるべきじゃない？
            self.price_loss_training_loop(self.n_sample_price, self.T_price, self.mparam, batch_size=64, state_init=None, shocks=None,)
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
        self.price_loss_training_loop(self.n_sample_price, self.T_price, self.mparam, init_ds.policy_init_only, self.price_model, batch_size=64, state_init=None, shocks=None)
        self.policy_ds = self.init_ds.get_policydataset(init_ds.policy_init_only, policy_type, self.price_model, init=True, update_init=False)
        

    def create_data(self, input_data):
        k_cross = input_data["k_cross"]
        ashock = input_data["ashock"]
        for t in range(self.t_unroll):
            a_tmp = torch.repeat_interleave(ashock[:, t:t+1], 50, dim=1)
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
        price = input_data["price"]
        for t in range(self.t_unroll):
            price = price[:, t:t+1]
            a_tmp = torch.repeat_interleave(ashock[:, t:t+1], 50, dim=1)#samplerで作成したbatch_size, self.t_unrollのashockを使っている。
            a_tmp = torch.unsqueeze(a_tmp, 2)
            basic_s_tmp = a_tmp
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
                util_sum += -price * k_cross + self.discount[t] * value
                continue
            k_cross = self.policy_fn(full_state_dict)

        loss1 = util_sum
            




    def loss2(self, input_data):
        k_cross = input_data["k_cross"]
        ashock = input_data["ashock"]
        price = input_data["price"]
        for t in range(self.t_unroll):
            a_tmp = torch.repeat_interleave(ashock[:, t:t+1], 50, dim=1)#samplerで作成したbatch_size, self.t_unrollのashockを使っている。
            a_tmp = torch.unsqueeze(a_tmp, 2)
            basic_s_tmp = a_tmp
            full_state_dict = {
                "basic_s": basic_s_tmp,
                "agt_s": self.init_ds.normalize_data(torch.unsqueeze(k_cross, axis=-1), key="agt_s", withtf=True)
            }
            
            if t == self.t_unroll - 1:
                wage = self.mparam.eta / price
                e0 = -self.mparam.GAMY * price * k_cross + self.mparam.BETA * self.init_ds.unnormalize_data(value(full_state_dict), withth=True)
                a_tmp = torch.repeat_interleave(ashock[:, t:t+1], 50, dim=1)#samplerで作成したbatch_size, self.t_unrollのashockを使っている。
                a_tmp = torch.unsqueeze(a_tmp, 2)
                basic_s_tmp = torch.cat([torch.unsqueeze(k_cross_pre, axis=-1), a_tmp], axis=-1)
                full_state_dict_e1 = {
                    "basic_s": basic_s_tmp,
                    "agt_s": self.init_ds.normalize_data(torch.unsqueeze(k_cross_pre, axis=-1), key="agt_s", withtf=True)
                }
                e1 = -price * (1-self.mparam.delta) * k_cross_pre + self.mparam.BETA * self.init_ds.unnormalize_data(value(full_state_dict_e1), withth=True)
                xitemp = (e0 - e1)/(price * wage)
                xi = min(self.mparam.B, max(0, xitemp))
                alpha = xi / self.mparam.B
                value = 0
                true_policy = alpha * k_cross + (1 - alpha) * (1-self.mparam.delta) * k_cross_pre
                
                full_state_dict_loss = {
                    "basic_s": basic_s_tmp,
                    "agt_s": self.init_ds.normalize_data(torch.unsqueeze(k_cross, axis=-1), key="agt_s", withtf=True)
                }
                
                loss = torch.mean((true_policy - self.policy_fn_true(full_state_dict_loss))**2)
                continue
            
            price = self.price_model(k_cross)
            k_cross_pre = k_cross
            k_cross = self.policy_fn(full_state_dict)
        
        return loss
    

    def price_loss_training_loop(self, n_sample, T, mparam, policy_fn, price_fn, optimizer, batch_size=64, state_init=None, shocks=None):
        # デバイスの設定
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # ショックの設定
        if shocks is not None:
            ashock = shocks.to(device)
            assert n_sample == ashock.shape[0], "n_sample is inconsistent with given shocks."
            assert T == ashock.shape[1], "T is inconsistent with given shocks."
            if state_init:
                assert torch.all(ashock[:, 0:1] == state_init["ashock"].to(device)), "Shock inputs are inconsistent with state_init"
        else:
            ashock = KT.simul_shocks(n_sample, T, mparam.Z, mparam.Pi, state_init).to(device)
        
        # エージェント数の取得
        n_agt = mparam.n_agt  # 例: エージェント数
        
        # k_crossの初期化
        k_cross = torch.zeros(n_sample, n_agt, T, device=device)
        if state_init:
            assert n_sample == state_init["k_cross"].shape[0], "n_sample is inconsistent with state_init."
            k_cross[:, :, 0] = state_init["k_cross"].to(device)
        else:
            k_cross[:, :, 0] = mparam.k_ss.to(device)
        
        for t in range(1, T):
            k_prev = k_cross[:, :, t-1]  # 前の時間ステップの資本 [n_sample, n_agt]
            price = price_fn(k_prev)      # 価格 [n_sample, n_agt]
            wage = mparam.eta / price      # 賃金 [n_sample, n_agt]
            yterm = ashock[:, t-1].unsqueeze(1) * k_prev**mparam.theta  
            n = (mparam.nu * yterm / wage)**(1 / (1 - mparam.nu))      
            y = yterm * n**mparam.nu                                    
            
            # 政策関数を用いて次期資本を決定
            k_cross[:, :, t] = self.policy_fn(k_prev, ashock[:, t-1])                   

            # 投資と消費の計算
            inow = mparam.GAMY * k_cross[:,:,t] - (1 - mparam.delta) * k_prev   
            ynow = ashock[:, t-1].unsqueeze(1) * k_prev**mparam.theta * n**mparam.nu  # 消費 [n_sample, n_agt]
            Cnow = ynow - inow                                           # 消費量 [n_sample, n_agt]
            
            # 目標価格の計算（例として1/Cnowとする）
            price_target = 1 / Cnow                                       # 目標価格 [n_sample, n_agt]
            
            # 現在の価格と目標価格の二乗誤差を計算
            loss_t = (price - price_target)**2                            
        
        # DataLoaderの作成
        dataset = CustomDataset(loss_t)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

        # トレーニングループ
        self.price_model.train()  # モデルをトレーニングモードに設定
        for epoch in range(mparam.num_epochs):
            print(f"Epoch {epoch+1}/{mparam.num_epochs}")
            for batch_idx, (batch_loss,) in enumerate(dataloader):
                optimizer.zero_grad()
                loss = batch_loss.mean()
                loss.backward()
                optimizer.step()
                
                if batch_idx % 100 == 0:
                    print(f"  Batch {batch_idx}/{len(dataloader)}: Loss = {loss.item():.6f}")
        
        print("Training completed.")
    
    def current_policy(self, k_cross, ashock):
        k_mean = torch.mean(k_cross, dim=1, keepdim=True)
        k_mean = torch.repeat_interleave(k_mean, self.mparam.n_agt, dim=1)
        ashock = torch.repeat_interleave(ashock, self.mparam.n_agt, dim=1)
        basic_s = torch.cat([k_cross, k_mean, ashock], dim=-1)
        basic_s = self.normalize_data(basic_s, key="basic_s", withtf=True)
        agt_s = self.normalize_data(k_cross, key="agt_s", withtf=True)
        
        full_state_dict = {
            "basic_s": basic_s,
            "agt_s": agt_s
        }
        
        output = self.policy_fn(full_state_dict)[..., 0]
        return output


            

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