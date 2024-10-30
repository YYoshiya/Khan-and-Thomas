import numpy as np
import util
import torch
import torch.optim as optim
import torch.nn.functional as F

DTYPE = "float32"
if DTYPE == "float64":
    NP_DTYPE = np.float64
    TORCH_DTYPE = torch.float64  # PyTorchのデータ型を指定
elif DTYPE == "float32":
    NP_DTYPE = np.float32
    TORCH_DTYPE = torch.float32  # PyTorchのデータ型を指定
else:
    raise ValueError("Unknown dtype.")

class ValueTrainer():
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.value_config = config["value_config"]
        d_in = config["n_basic"] + config["n_fm"] + config["n_gm"]
        self.model = util.FeedforwardModel(d_in, 1, self.value_config, name="v_net").to(self.device)
        self.gm_model = util.GeneralizedMomModel(1, config["gm_config"], name="v_gm").to(self.device)
            # 両方のモデルのパラメータを集める
        params = list(self.model.parameters()) + list(self.gm_model.parameters())

        self.train_vars = None
        self.optimizer = optim.Adam(
            params, 
            lr=self.value_config["lr"], 
            eps=1e-8, 
            betas=(0.99, 0.99)
        )
    
    def prepare_state(self, input_data):
        state = torch.cat([input_data["basic_s"][..., 0:1], input_data["basic_s"][..., 2:]], dim=-1)
        gm = self.gm_model(input_data["agt_s"])
        state = torch.cat([state, gm], dim=-1)
        return state
    
    def value_fn(self, input_data):
        state = self.prepare_state(input_data)
        value = self.model(state)
        return value
    
    def loss(self, input_data):
        y_pred = self.value_fn(input_data)
        y = input_data["value"]
        loss = F.mse_loss(y_pred, y)
        return loss
    
    def train(self, train_dataset, valid_dataset, num_epoch=None, batch_size=None):
            for epoch in range(num_epoch + 1):
                for train_data in train_dataset:
                    train_data = {key: value.to(device=self.device, dtype=TORCH_DTYPE) for key, value in train_data.items()}
                    self.optimizer.zero_grad()  # 勾配をゼロリセット
                    loss_dict = self.loss(train_data)  # lossメソッドを使用して損失を計算
                    loss_dict.backward()  # 勾配計算
                    self.optimizer.step()  # パラメータ更新

                # Validation loop
                if epoch % 20 == 0:
                    with torch.no_grad():  # 検証時には勾配計算を行わない
                        for valid_data in valid_dataset:
                            valid_data = {key: value.to(self.device, dtype=TORCH_DTYPE) for key, value in valid_data.items()}
                            val_loss_dict = self.loss(valid_data)  # lossメソッドを使用
                            print(f"Value function learning epoch: {epoch}, validation loss: {val_loss_dict}")

    def save_model(self, path="value_model.pth"):
    # モデルの重みを保存
        torch.save(self.model.state_dict(), path)
        # n_gmが0より大きい場合、gm_modelも保存
        if self.config["n_gm"] > 0:
            torch.save(self.gm_model.state_dict(), path.replace(".pth", "_gm.pth"))

    def load_model(self, path):
    # モデルの重みをCPU上にロード
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)  # モデル自体もCPUに移動
        
        # n_gmが0より大きい場合、gm_modelも読み込み
        if self.config["n_gm"] > 0:
            self.gm_model.load_state_dict(torch.load(path.replace(".pth", "_gm.pth"), map_location=self.device))
            self.gm_model.to(self.device)  # gm_modelもCPUに移動
