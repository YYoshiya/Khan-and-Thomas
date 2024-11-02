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
        layers.append(nn.Linear(d_in, d_out))
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
        x = self.basis_fn(x)  
        gm = torch.mean(x, dim=-2, keepdim=True)  
        gm = gm.repeat(1, x.shape[-2], 1)  #384,50,1
        return gm
    
    def _initialize_weights(self):
        for layer in self.dense_layers:
            if isinstance(layer, nn.Linear):
                # Xavier（Glorot）初期化を使用
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
                

class Policy(FeedforwardModel):
    def __init__(self, d_in, config, name="policy"):
        super(Policy, self).__init__(d_in, d_out=1, config=config, name=name)
        
    def forward(self, x):
        x = self.dense_layers(x)
        policy = torch.sigmoid(x)
        return policy
    
    def _initialize_weights(self):
        for layer in self.dense_layers:
            if isinstance(layer, nn.Linear):
                # Xavier（Glorot）初期化を使用
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

class GeneralizedMomPrice(nn.Module):
    def __init__(self, d_in, config, name="generalizedmomprice"):
        super(GeneralizedMomPrice, self).__init__()

        self.price_layer = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64,1),
            nn.Softplus()
        )
        self.dense_layers = nn.Sequential(
            nn.Linear(1, 12),
            nn.Tanh(),
            nn.Linear(12, 12),
            nn.Tanh(),
            nn.Linear(12, 1),
        )
        # 重みの初期化を追加
        self._initialize_weights()
        
    
    def forward(self, x):
        x = self.dense_layers(x)
        gm = torch.mean(x, dim=-2)
        price = self.price_layer(gm)
        return price
    
    def _initialize_weights(self):
        for layer in self.dense_layers:
            if isinstance(layer, nn.Linear):
                # Xavier（Glorot）初期化を使用
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        for layer in self.price_layer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    
class PriceModel(nn.Module):
    def __init__(self, d_in, d_out, config, name="pricemodel"):
        super(PriceModel, self).__init__()
        layers = []
        for w in config["net_width"]:
            layers.append(nn.Linear(d_in, w))
            layers.append(nn.Tanh())
            d_in = w
        layers.append(nn.Linear(d_in, d_out))
        layers.append(nn.Softplus())
        self.dense_layers = nn.Sequential(*layers)
        
        # 重みの初期化を追加
        self._initialize_weights()

    def forward(self, x):
        return self.dense_layers(x)
    
    def _initialize_weights(self):
        for layer in self.dense_layers:
            if isinstance(layer, nn.Linear):
                # Xavier（Glorot）初期化を使用
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    
    
def print_elapsedtime(delta):
    hours, rem = divmod(delta, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Elapsed time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))