import util
import simulation_KT as KT
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from param import KTParam
import numpy as np
import json

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

config_path = 'game_nn_n50_0fm1gm.json'
with open(config_path, 'r') as f:
        config = json.load(f)
policy_config = config["policy_config"]
value_config = config["value_config"]
price_config = config["price_config"]

policy = util.FeedforwardModel(2, 1, policy_config, name="p_test").to(device)
policy_true = util.FeedforwardModel(3, 1, policy_config, "p_net_true_test").to(device)
policy_gm = util.GeneralizedMomModel(1, config["gm_config"], name="v_gm_test").to(device)
price_model = util.PriceModel(51, 1, config["price_config"], name="price_net_test").to(device)
value = util.FeedforwardModel(3, 1, value_config, name="v_net_test").to(device)
value_gm = util.GeneralizedMomModel(1, config["gm_config"], name="v_gm_test").to(device)

model_dir = r"C:\Users\yuka\Yoshiya\Khan and Thomas result\1026_51_ver\game_nn_n50_test"
policy_path = model_dir + r"\policy.pth"
policy_true_path = model_dir + r"\policy_true.pth"
policy_gm_path = model_dir + r"\policy_gm.pth"
price_path = model_dir + r"\policy_price.pth"
value_path = model_dir + r"\value0.pth"
value_gm_path = model_dir + r"\value0_gm.pth"

policy.load_state_dict(torch.load(policy_path, map_location=device))
policy_true.load_state_dict(torch.load(policy_true_path, map_location=device))
policy_gm.load_state_dict(torch.load(policy_gm_path, map_location=device))
price_model.load_state_dict(torch.load(price_path, map_location=device))
value.load_state_dict(torch.load(value_path, map_location=device))
value_gm.load_state_dict(torch.load(value_gm_path, map_location=device))
# 評価モードに設定
policy.eval()
policy_true.eval()
policy_gm.eval()
price_model.eval()
value.eval()
value_gm.eval()

mparam = KTParam(50, 0.9540, None)

def valid_simul(T, mparam):
    # 初期化: 時間ステップT+1に拡張
    k_cross = np.zeros((1, 50, T+1))
    price = np.zeros((1, T))
    n = np.zeros((1, 50, T))
    inow = np.zeros((1, 50, T))
    ynow = np.zeros((1, 50, T))
    Cnow = np.zeros((1, T))
    wage = np.zeros((1, T))
    
    # ショックのシミュレーション
    ashock = KT.simul_shocks(1, 200, mparam.Z, mparam.Pi, state_init=None)
    
    for t in range(T):
        # 価格の計算 T
        price[:, t] = prepare_price_input(k_cross[:, :, t], ashock[:, t:t+1]).detach().cpu().squeeze(-1).numpy()
        
        # 労働供給の計算 T
        xi = np.random.uniform(0, mparam.B, size=(1, 50))  # xiが使用されていない場合は削除可能
        wage[:, t] = mparam.eta / price[:, t]
        
        # 生産要素の計算
        yterm = ashock[:, t:t+1] * k_cross[:, :, t] ** mparam.theta#1,50
        n[:, :, t] = (mparam.nu * yterm / wage[:, t:t+1]) ** (1 / (1 - mparam.nu))#1,50
        
        # 資本の次期値を政策関数から計算
        k_cross[:, :, t+1] = policy_true(k_cross[:, :, t:t+1], ashock[:, t:t+1]).detach().cpu().squeeze(-1).numpy()
        
        # 現在の投資と生産の計算
        inow[:, :, t] = mparam.GAMY * k_cross[:, :, t+1] - (1 - mparam.delta) * k_cross[:, :, t]
        ynow[:, :, t] = ashock[:, t:t+1] * k_cross[:, :, t] ** mparam.theta * (n[:, :, t] ** mparam.nu)
        
        # 現在の消費の計算
        Cnow[:, t] = np.mean(ynow[:, :, t], axis=1) - np.mean(inow[:, :, t], axis=1)
    
    # 必要に応じて結果を返す
    return {
        "k_cross": k_cross,
        "price": price,
        "n": n,
        "inow": inow,
        "ynow": ynow,
        "Cnow": Cnow,
        "wage": wage
    }



def get_bin_edges(num_bins=50, min_val=0.0, max_val=3.0):
        return torch.linspace(min_val, max_val, steps=num_bins+1)
            
def assign_bins(k_cross, bin_edges):
    bin_indices = torch.bucketize(k_cross, bin_edges, right=False) - 1  # shape: (batch_size, num_agents)
    bin_indices = torch.clamp(bin_indices, 0, len(bin_edges) - 2) 
    return bin_indices

def count_bins(bin_indices, num_bins=50):
    """
    bin_indices: Tensor of shape (batch_size, num_agents)
    """
    # ワンホットエンコーディング
    one_hot = F.one_hot(bin_indices, num_classes=num_bins)  # shape: (batch_size, num_agents, num_bins)
    
    # エージェント数をカウント
    bin_counts = one_hot.sum(dim=1).float()  # shape: (batch_size, num_bins)
    
    return bin_counts

def prepare_price_input(k_cross, ashock, num_bins=50):
    """
    k_cross: Tensor of shape (batch_size, num_agents)
    ashock: Tensor of shape (batch_size, 1)
    """
    k_cross = torch.tensor(k_cross, dtype=TORCH_DTYPE).to(device)
    ashock = torch.tensor(ashock, dtype=TORCH_DTYPE).to(device)
    # ビンエッジを定義
    bin_edges = get_bin_edges().to(device)
    
    # ビンに割り当て
    bin_indices = assign_bins(k_cross, bin_edges)
    
    # ビンごとのカウント
    bin_counts = count_bins(bin_indices, num_bins)
    
    # ashockと結合
    price_input = torch.cat([bin_counts, ashock], dim=1)  # shape: (batch_size, num_bins + 1)
    
    price = price_model(price_input)  # shape: (batch_size, 1)
    return price


result = valid_simul(200, mparam)