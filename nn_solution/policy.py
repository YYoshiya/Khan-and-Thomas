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
        self.policy_true = util.FeedforwardModel(d_in, 1, self.policy_config, "p_net_true").to(self.device)
        self.gm_model = util.GeneralizedMomModel(1, self.config["gm_config"], name="v_gm").to(self.device)
        self.price_model = util.PriceModel(51, 1, self.config["price_config"], name="price_net").to(self.device)
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
    
    def sampler(self, batch_size, init=None, update_init=False):
        if init is None:
            self.policy_ds = self.init_ds.get_policydataset(self.current_policy, "nn_share", self.prepare_price_input, init=init, update_init=update_init)
        ashock = KT.simul_shocks(self.ashock_num, self.t_unroll, self.mparam.Z, self.mparam.Pi)
        self.policy_ds.datadict["ashock"] = torch.tensor(ashock, dtype=TORCH_DTYPE)
        dataset = CustomDataset(self.policy_ds.datadict)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return train_loader
        
        
        
    def prepare_state(self, input_data):
        state = torch.cat([input_data["basic_s"][..., 0:1], input_data["basic_s"][..., 2:]], dim=-1)
        gm = self.gm_model(input_data["agt_s"])
        state = torch.cat([state, gm], dim=-1)
        return state
    
    def policy_fn(self, input_data):
        state = self.prepare_state(input_data)
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
    
    def train(self, n_epoch, batch_size=None):
        loss1_list = []
        valid_data = {k: torch.tensor(self.init_ds.datadict[k], dtype=TORCH_DTYPE) for k in self.init_ds.keys}
        ashock = KT.simul_shocks(
            self.valid_size, self.t_unroll, self.mparam.Z, self.mparam.Pi,
            state_init=self.init_ds.datadict
        )
        valid_data["ashock"] = torch.tensor(ashock, dtype=TORCH_DTYPE)
        valid_data = {k: v.to(self.device) for k, v in valid_data.items()}
        init=True
        update_init = False
        for n in tqdm(range(n_epoch), desc="Training Progress"):
            epoch_loss1 = 0.0
            train_datasets = self.sampler(batch_size, init, update_init)
            init=None
            for train_data in train_datasets:
                train_data = {key: value.to(self.device, dtype=TORCH_DTYPE) for key, value in train_data.items()}
                # トレーニングステップを実行
                self.optimizer.zero_grad()
                output_dict = self.loss1(train_data)
                loss1 = output_dict["m_util"]
                loss1.backward()
                self.optimizer.step()
                epoch_loss1 += loss1.item()
                
                self.optimizer_true.zero_grad()
                loss2 = self.loss2(train_data)
                loss2.backward()
                self.optimizer_true.step()#この後にpriceの学習入れるべきじゃない？
            
            avg_loss1 = epoch_loss1 / len(train_datasets)
            loss1_list.append(avg_loss1)
            self.price_loss_training_loop(self.n_sample_price, self.price_config["T"], self.mparam, self.current_policy, "nn_share", self.prepare_price_input, self.optimizer_price, batch_size=64,  num_epochs=2)
            update_frequency = min(25, max(3, int(math.sqrt(n + 1))))
            if n > 0 and n % update_frequency == 0:
                update_init = self.policy_config["update_init"]
                train_vds, valid_vds = self.get_valuedataset(init=init, update_init=update_init)
                for vtr in self.vtrainers:
                    vtr.train(
                        train_vds, valid_vds,
                        self.config["value_config"]["num_epoch"],
                        self.config["value_config"]["batch_size"]
                    )
        
        plt.figure(figsize=(12, 5))
        # Loss1のプロット
        plt.subplot(1, 2, 1)
        plt.plot(loss1_list, label='Loss1')
        plt.xlabel('Epoch')
        plt.ylabel('Loss1')
        plt.title('Loss1 over Epochs')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.init_ds.value_mean, label='Value', color='orange')
        plt.xlabel('Update Step')
        plt.ylabel('Value')
        plt.title('Value over Update Steps')
        plt.legend()
        plt.tight_layout()
        plt.show()

        
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
        self.price_loss_training_loop(self.n_sample_price, self.price_config["T"], self.mparam, init_ds.policy_init_only, "nn_share", self.prepare_price_input, self.optimizer_price,batch_size=64, init=True, state_init=None, shocks=None, num_epochs=10) #self.price_config["T"]
        self.policy_ds = self.init_ds.get_policydataset(init_ds.policy_init_only, policy_type, self.prepare_price_input, init=True, update_init=False)
        

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
        util_sum = 0
        for t in range(self.t_unroll):
            price = price[:, t:t+1]
            k_tmp = torch.unsqueeze(k_cross, 2)
            a_tmp = torch.repeat_interleave(ashock[:, t:t+1], 50, dim=1)#samplerで作成したbatch_size, self.t_unrollのashockを使っている。
            a_tmp = torch.unsqueeze(a_tmp, 2)
        
            if t == self.t_unroll - 1:
                price = self.prepare_price_input(k_cross, ashock[:, t:t+1])
                k_mean_tmp = torch.mean(k_tmp, dim=1, keepdim=True).repeat(1, 50,1)
                basic_s_tmp_v = torch.cat([k_tmp, k_mean_tmp, a_tmp], axis=-1)
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
                util_sum += -price * k_cross + self.discount[t] * value
                continue
            basic_s_tmp = self.init_ds.normalize_data_ashock(a_tmp, key="basic_s", withtf=True)
            full_state_dict = {
                "basic_s": basic_s_tmp,
                "agt_s": self.init_ds.normalize_data(k_tmp, key="agt_s", withtf=True)
            }
            k_cross = self.init_ds.unnormalize_data_ashock(self.policy_fn(full_state_dict), key="basic_s", withtf=True).clamp(min=0.01).squeeze(-1)

        output_dict = {"m_util": -torch.mean(util_sum[:, 0]), "k_end": torch.mean(k_cross)}
        print(f"loss1:{-output_dict['m_util']}")
        return output_dict
            




    def loss2(self, input_data):
        k_cross = input_data["k_cross"]
        ashock = input_data["ashock"]
        price = input_data["price"]
        for t in range(self.t_unroll):
            k_tmp = torch.unsqueeze(k_cross, 2)
            k_mean_tmp = torch.mean(k_tmp, dim=1, keepdim=True).repeat(1, 50, 1)
            a_tmp = torch.repeat_interleave(ashock[:, t:t+1], 50, dim=1)#samplerで作成したbatch_size, self.t_unrollのashockを使っている。
            a_tmp = torch.unsqueeze(a_tmp, 2)
            
            
            if t == self.t_unroll - 1:
                value0 = 0
                value1 = 0
                price = self.prepare_price_input(k_cross, ashock[:, t:t+1])
                
                basic_s_tmp_pre = torch.cat([k_tmp, k_mean_tmp,a_tmp], axis=-1)
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
                e0 = -self.mparam.GAMY * price * k_cross + self.mparam.BETA * value0
                a_tmp = torch.repeat_interleave(ashock[:, t:t+1], 50, dim=1)#samplerで作成したbatch_size, self.t_unrollのashockを使っている。
                a_tmp = torch.unsqueeze(a_tmp, 2)
                basic_s_tmp = torch.cat([torch.unsqueeze(k_cross_pre, axis=-1), k_mean_tmp, a_tmp], axis=-1)
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
                xitemp = (e0 - e1)/(price * wage)
                xi = torch.min(torch.tensor(self.mparam.B), torch.max(torch.tensor(0.0), xitemp))
                alpha = xi / self.mparam.B
                true_policy = alpha * k_cross + (1 - alpha) * (1-self.mparam.delta) * k_cross_pre
                
                
                full_state_dict_loss = {
                    "basic_s": basic_s_tmp_e1,
                    "agt_s": self.init_ds.normalize_data(k_tmp, key="agt_s", withtf=True)
                }
                
                loss = torch.mean((true_policy - self.policy_fn_true(full_state_dict_loss).squeeze(-1))**2)
                continue
            
            basic_s_tmp = self.init_ds.normalize_data_ashock(a_tmp, key="basic_s", withtf=True)
            full_state_dict = {
                "basic_s": basic_s_tmp,
                "agt_s": self.init_ds.normalize_data(torch.unsqueeze(k_cross, axis=-1), key="agt_s", withtf=True)
            }
            
            k_cross_pre = k_cross
            k_cross = self.init_ds.unnormalize_data_ashock(self.policy_fn(full_state_dict), key="basic_s", withtf=True).clamp(min=0.01).squeeze(-1)
        print(f"loss2:{loss}")
        return loss
    

    def price_loss_training_loop(
        self,
        n_sample,
        T,
        mparam,
        policy_fn,
        policy_type,
        price_fn,
        optimizer,
        batch_size=64,
        init=None,
        state_init=None,
        shocks=None,
        num_epochs=3
    ):
        
        
        # データ生成（1回だけ実行）
        with torch.no_grad():
            if init is not None:
                input_data = KT.init_simul_k(
                    n_sample, T, mparam, policy_fn, policy_type, price_fn, state_init=None, shocks=None)
                loss_fn = self.loss_price_init
                #for param in policy_fn.parameters():
                    #param.requires_grad = False
            else:
                input_data = KT.simul_k(
                    n_sample, T, mparam, policy_fn, policy_type, price_fn, state_init=self.init_ds.datadict)
                loss_fn = self.loss_price
                #for param in self.policy.parameters():
                    #param.requires_grad = False
                #for param in self.gm_model.parameters():
                    #param.requires_grad = False
                #for param in self.policy_true.parameters():
                    #param.requires_grad = False

        # データの整形
        k_cross = input_data["k_cross"]
        ashock = input_data["ashock"]
        k_tmp = np.reshape(k_cross, (-1, 50))  # 384*T, 50
        a_tmp = np.reshape(ashock, (-1, 1))    # 384*T, 1
        basic_s = torch.tensor(
            np.concatenate([k_tmp, a_tmp], axis=1),
            dtype=TORCH_DTYPE
        ).to(self.device)  # データを結合し、デバイスに移動

        # データセットの作成
        dataset = PriceDataset(basic_s)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        losses = []  # ロスを保存するリスト

        # エポックループの追加
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            epoch_loss = 0.0

            # 各エポック内のバッチループ
            for batch_idx, data in enumerate(dataloader):
                data = data.to(self.device)
                optimizer.zero_grad()

                # ロス関数の計算
                loss = loss_fn(data, policy_fn, price_fn, mparam)

                # 損失の逆伝播と最適化
                loss.backward()
                optimizer.step()

                # ロスの累積と保存
                epoch_loss += loss.item()
                losses.append(loss.item())

                # ロスの出力
                #print(f"Epoch {epoch + 1}, Step {batch_idx + 1}, Loss: {loss.item()}")

            # エポックごとの平均ロスを表示
            avg_epoch_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch + 1} の平均ロス: {avg_epoch_loss}\n")

        print("トレーニング完了")

        # トレーニング後にロスをプロット
        #plt.plot(losses)
        #plt.xlabel('Iteration')
        #plt.ylabel('Loss')
        #plt.title('Training Loss')
        #plt.show()


    
    def loss_price(self, data, policy_fn, price_fn, mparam):
        ashock = data[:,50:].to(self.device)#128,1
        k_cross = data[:, :50].to(self.device)#128,50
        price = price_fn(k_cross, ashock)
        wage = mparam.eta / price

        yterm = ashock * k_cross ** mparam.theta
        n = (mparam.nu * yterm / wage)**(1/(1-mparam.nu))#最右項は*2
        k_tmp = k_cross.unsqueeze(2)#128,50,1
        k_new = policy_fn(k_tmp, ashock, withtf=True).clamp(min=0.01).squeeze(2)
        inow = mparam.GAMY * k_new - (1 - mparam.delta) * k_cross
        ynow = ashock * k_cross**mparam.theta * (n**mparam.nu)
        Cnow = ynow.sum(dim=1, keepdim=True) - inow.sum(dim=1, keepdim=True)
        Cnow = Cnow.clamp(min=0.1)
        #print(f"k_cross:{k_cross[0,0]}, price:{price[0,0]}, yterm:{yterm[0,0]}, Cnow:{Cnow[0,0]}")
        price_target = 1 / Cnow
        mse_loss_fn = nn.L1Loss()
        loss = mse_loss_fn(price, price_target)
        return loss


    def loss_price_init(self, data, policy_fn, price_fn, mparam):
        ashock = data[:,50:]#128,1
        k_cross = data[:, :50]#128,50
        k_mean = torch.mean(k_cross, dim=1, keepdim=True).repeat(1, 50).unsqueeze(2)#128,50,1
        price = price_fn(k_cross, ashock)
        wage = mparam.eta / price

        yterm = ashock * k_cross ** mparam.theta
        n = (mparam.nu * yterm / wage)**(1/(1-mparam.nu))
        k_tmp = k_cross.unsqueeze(2)#128,50,1
        a_tmp = ashock.repeat(1, 50).unsqueeze(2)#128,50,1
        basic_s = self.init_ds.normalize_data(torch.cat([k_tmp, k_mean, a_tmp], dim=-1), key="basic_s", withtf=True)

        k_new = self.init_ds.unnormalize_data_k_cross(policy_fn(basic_s).squeeze(2), key="basic_s", withtf=True)
        inow = mparam.GAMY * k_new - (1 - mparam.delta) * k_cross
        ynow = ashock * k_cross**mparam.theta * (n**mparam.nu)
        Cnow = ynow.sum(dim=1, keepdim=True) - inow.sum(dim=1, keepdim=True)
        Cnow = Cnow.clamp(min=0.1)
        price_target = 1 / Cnow
        mse_loss_fn = nn.L1Loss()
        loss = mse_loss_fn(price, price_target)
        if torch.isnan(loss):
            print("Loss is NaN")
            print(f"price: {price}")
            print(f"k_cross: {k_cross}")
            print(f"Cnow: {Cnow}")
    
        return loss



        

    
    def current_policy(self, k_cross, ashock, withtf=False):
        if withtf:
            k_mean = torch.mean(k_cross, dim=1, keepdim=True).repeat(1, 50, 1)
            ashock = ashock.repeat(1, 50).unsqueeze(2)
            basic_s = torch.cat([k_cross, k_mean, ashock], dim=2)
            agt_s = k_cross
        else:
            k_mean = np.mean(k_cross, axis=1, keepdims=True)  # NumPy: 形状 (384, 1, 1)
            k_mean = np.repeat(k_mean, self.mparam.n_agt, axis=1)  # NumPy: 形状 (384, 50, 1)

            # ashockもNumPyで操作
            ashock = np.repeat(ashock, self.mparam.n_agt, axis=1)[:, :, np.newaxis]  # NumPy: 形状 (384, 50, 1)

            # k_cross, k_mean, ashockを結合 (NumPyで)
            basic_s = np.concatenate([k_cross, k_mean, ashock], axis=-1)  # NumPy: 形状 (384, 50, X)

            # NumPy配列をTorchテンソルに変換
            basic_s = torch.tensor(basic_s, dtype=TORCH_DTYPE).to(self.device)  # Torch: 形状 (384, 50, X)

            # k_crossも同様にTorchテンソルに変換
            agt_s = torch.tensor(k_cross, dtype=TORCH_DTYPE).to(self.device)  # Torch: 形状 (384, 50)
        basic_s = self.init_ds.normalize_data(basic_s, key="basic_s", withtf=True)
        agt_s = self.init_ds.normalize_data(agt_s, key="agt_s", withtf=True)
        
        full_state_dict = {
            "basic_s": basic_s,
            "agt_s": agt_s
        }
        
        output = self.init_ds.unnormalize_data_k_cross(self.policy_fn_true(full_state_dict), "basic_s", withtf=True)
        return output
    
    def get_valuedataset(self, update_from=None, init=None, update_init=False):
        return self.init_ds.get_valuedataset(self.current_policy, "nn_share", self.prepare_price_input, update_from, init, update_init)
    
    def init_policy_fn_tf(self, k_cross, k_mean, ashock):
        # PyTorchで処理する
        k_mean_tmp = k_mean.repeat(1, 50).unsqueeze(2)  # axis=1をPyTorchで再現
        ashock_tmp = ashock.repeat(1, 50).unsqueeze(2)  # axis=1をPyTorchで再現
        basic_s = torch.cat([k_cross, ashock_tmp, k_mean_tmp], dim=2)  # NumPyのconcatenateをtorch.catで再現
        
        # GPUでNNの計算を実行
        basic_s_torch = basic_s.to("cuda")  # GPUに移動
        output_torch = self.init_ds.policy_init_only(basic_s_torch)
        
        # 結果をそのまま返す（必要ならCPUに移動してNumPyに変換）
        output = output_torch  # 必要に応じて .to('cpu') を付けてCPUに戻す

        return output

    def price_fn(self, input_data):
        price_data = self.init_ds.normalize_data_price(data_tmp, key="basic_s", withtf=True)
        price = self.init_ds.unnormalize_data_k_cross(self.price_model(price_data), key="basic_s", withtf=True).clamp(min=0.01)
        return price
    
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