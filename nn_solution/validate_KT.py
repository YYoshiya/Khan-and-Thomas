import util
import simulation_KT as KT
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from policy import KTPolicyTrainer
from value import ValueTrainer
from dataset import KTInitDataSet
from param import KTParam
import numpy as np
import json
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

config_path = 'game_nn_n50_0fm1gm.json'
with open(config_path, 'r') as f:
        config = json.load(f)

model_dir_yuka = r"C:\Users\yuka\Yoshiya\Khan and Thomas result\1026_51_ver\game_nn_n50_test"
model_dir = r"C:\Users\Owner\OneDrive\デスクトップ\Github\Khan-and-Thomas\results\game_nn_n50_test"

mparam = KTParam(50, 0.9540, None)
init_ds = KTInitDataSet(mparam, config)
policy_config = config["policy_config"]
value_config = config["value_config"]
price_config = config["price_config"]
vtrainers = [ValueTrainer(config) for i in range(value_config["num_vnet"])]
for i, vtr in enumerate(vtrainers):
        vtr.load_model(os.path.join(model_dir, "value{}.pth".format(i)))

ptrainer = KTPolicyTrainer(vtrainers, init_ds)

policy_path = model_dir + r"\policy.pth"
policy_true_path = model_dir + r"\policy_true.pth"
policy_gm_path = model_dir + r"\policy_gm.pth"
price_path = model_dir + r"\policy_price.pth"
value_path = model_dir + r"\value0.pth"
value_gm_path = model_dir + r"\value0_gm.pth"

ptrainer.policy.load_state_dict(torch.load(policy_path, map_location=device))
ptrainer.policy_true.load_state_dict(torch.load(policy_true_path, map_location=device))
ptrainer.gm_model.load_state_dict(torch.load(policy_gm_path, map_location=device))
ptrainer.price_model.load_state_dict(torch.load(price_path, map_location=device))

data = init_ds.get_valuedataset(init_ds.policy_init_only, "nn_share", ptrainer.prepare_price_input, ptrainer.value_simul_k, init=True, update_init=False)

ykSS = (mparam.GAMY - mparam.beta * (1 - mparam.delta)) / mparam.beta / mparam.theta
ckSS = ykSS + (1 - mparam.GAMY - mparam.delta)
ycSS = ykSS / ckSS
nSS = mparam.nu / mparam.eta * ycSS
kSS = (ykSS * nSS ** (-mparam.nu)) ** (1 / (mparam.theta - 1))
grid = np.linspace(0.1, 3.0, 50).reshape(1, 50, 1)

def valid_simul(T, mparam):
    # 初期化: 時間ステップT+1に拡張
    k_cross = np.zeros((1, 50, T+1))
    
    k_cross[:, :, 0:1] = grid
    
    price = np.zeros((1, T))
    n = np.zeros((1, 50, T))
    inow = np.zeros((1, 50, T))
    ynow = np.zeros((1, 50, T))
    Cnow = np.zeros((1, T))
    wage = np.zeros((1, T))
    
    # ショックのシミュレーション
    ashock = KT.simul_shocks(1, T, mparam, state_init=None)
    
    for t in range(T):
        # 価格の計算 T
        price[:, t] = ptrainer.prepare_price_input(k_cross[:, :, t], ashock[:, t:t+1]).detach().cpu().squeeze(-1).numpy()
        
        # 労働供給の計算 T
        xi = np.random.uniform(0, mparam.B, size=(1, 50))  # xiが使用されていない場合は削除可能
        wage[:, t] = mparam.eta / price[:, t]
        
        # 生産要素の計算
        yterm = ashock[:, t:t+1] * k_cross[:, :, t] ** mparam.theta#1,50
        n[:, :, t] = (mparam.nu * yterm / wage[:, t:t+1]) ** (1 / (1 - mparam.nu))#1,50
        
        # 資本の次期値を政策関数から計算
        k_cross[:, :, t+1] = ptrainer.current_policy(k_cross[:, :, t:t+1], ashock[:, t:t+1]).detach().cpu().squeeze(-1).numpy()
        
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



result = valid_simul(200, mparam)