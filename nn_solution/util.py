import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import numpy as np

DTYPE = "float64"
if DTYPE == "float64":
    NP_DTYPE = np.float64
    TORCH_DTYPE = torch.float64  # PyTorchのデータ型を指定
elif DTYPE == "float32":
    NP_DTYPE = np.float32
    TORCH_DTYPE = torch.float32  # PyTorchのデータ型を指定
else:
    raise ValueError("Unknown dtype.")

class FeedforwardModel(nn.Module):
    def __init__(self, d_in, d_out, config, name="agentmodel", dtype=TORCH_DTYPE):
        super(FeedforwardModel, self).__init__()
        layers = []
        for w in config["net_width"]:
            layers.append(nn.Linear(d_in, w))
            layers.append(nn.Tanh())  # Tanh アクティベーションを追加
            d_in = w  # 次のレイヤーの入力は現在の出力次元になる
        layers.append(nn.Linear(d_in, d_out))
        self.dense_layers = nn.Sequential(*layers).to(dtype)

    def forward(self, x):
        return self.dense_layers(x)

    def load_weights_after_init(self, path):
        # モデルの重みをロード
        self.load_state_dict(torch.load(path))

class GeneralizedMomModel(FeedforwardModel):
    def __init__(self, d_in, config, name="generalizedmomentmodel", dtype=TORCH_DTYPE):
        super(GeneralizedMomModel, self).__init__(d_in, d_out=1, config=config, name=name)

    def basis_fn(self, x):
        return self.dense_layers(x)

    def forward(self, x):
        x = self.basis_fn(x)  
        gm = torch.mean(x, dim=-2, keepdim=True)  
        gm = gm.repeat(1, x.shape[-2], 1)  #384,50,1
        return gm

    
class PriceModel(nn.Module):
    def __init__(self, d_in, d_out, config, name="pricemodel", dtype=TORCH_DTYPE):
        super(PriceModel, self).__init__()
        layers = []
        for w in config["net_width"]:
            layers.append(nn.Linear(d_in, w))
            layers.append(nn.Tanh())
            d_in = w
        layers.append(nn.Linear(d_in, d_out))
        layers.append(nn.Softplus())
        self.dense_layers = nn.Sequential(*layers).to(dtype)
        
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