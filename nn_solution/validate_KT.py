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



def valid_simul(T):
        k_cross = np.zeros((1,50,T))
        price = np.zeros((1, T))

        for t in range(T):
                price
