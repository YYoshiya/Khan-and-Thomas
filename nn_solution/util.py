import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import numpy as np

class FeedforwardModel(nn.Module):
    def __init__(self, d_in, d_out, config, name="agentmodel"):
        super(FeedforwardModel, self).__init__()
        layers = []
        for w in config["net_width"]:
            layers.append(nn.Linear(d_in, w))
            layers.append(nn.Tanh())  # Tanh アクティベーションを追加
            d_in = w  # 次のレイヤーの入力は現在の出力次元になる
        layers.append(nn.Linear(d_in, d_out))  # 最終出力層はアクティベーションなし
        self.dense_layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.dense_layers(x)

    def load_weights_after_init(self, path):
        # モデルの重みをロード
        self.load_state_dict(torch.load(path))

class GeneralizedMomModel(FeedforwardModel):
    def __init__(self, d_in, config, name="generalizedmomentmodel"):
        super(GeneralizedMomModel, self).__init__(d_in, d_out=1, config=config, name=name)

    def basis_fn(self, x):
        return self.dense_layers(x)

    def forward(self, x):
        x = self.basis_fn(x)  # [batch_size, 1]
        gm = torch.mean(x, dim=-2, keepdim=True)  # 平均を計算
        gm = gm.repeat(1, x.shape[-2], 1)  # 必要に応じて繰り返し
        return gm

    
class PriceModel(nn.Module):
    def __init__(self, d_in, d_out, config, name="pricemodel"):
        super(PriceModel, self).__init__()
        layers = []
        for w in config["net_width"]:
            layers.append(nn.Linear(d_in, w))
            layers.append(nn.ReLU())  # Tanh アクティベーションを追加
            d_in = w  # 次のレイヤーの入力は現在の出力次元になる
        layers.append(nn.Linear(d_in, d_out))  # 最終出力層はアクティベーションなし
        self.dense_layers = nn.Sequential(*layers)

    def forward(self, x):

        return self.dense_layers(x)

    
    
def print_elapsedtime(delta):
    hours, rem = divmod(delta, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Elapsed time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))