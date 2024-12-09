import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import seaborn as sns
import value_iter as vi
from param import params

sns.set(style="whitegrid")


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

def bisectp(nn, params, data):
    diff = 1e-4
    iter_count = 0
    pL = params.pL
    pH = params.pH

    while diff > params.critbp:
        p0 = (pL + pH) / 2
        pnew, thetanew  = eq_price(nn, data, params, p0)
        B0 = p0 - pnew
        if B0 < 0:
            pL = p0
        else:
            pH = p0
        diff = pH - pL
        iter_count += 1
    
    return pnew, thetanew
    



def eq_price(nn, data, params, price):
    i_size = params.ishock_gpu.size(0)
    max_cols = data["grid"].size(1)
    ashock_3d = params.ishock.view(1, 1, i_size).expand(data["grid"].size(0), max_cols, -1)
    ishock_3d = params.ishock.view(1, 1, i_size).expand(data["grid"].size(0), max_cols, -1)
    price = price.view(-1, 1, 1).expand(-1, max_cols, i_size)
    wage = params.eta/price
    e0, e1 = next_value_price(data, nn, params, max_cols)#作らなきゃいけない。
    threshold = (e0 - e1) / params.eta
    xi = torch.min(torch.tensor(params.B, dtype=TORCH_DTYPE), torch.max(torch.tensor(0, dtype=TORCH_DTYPE), threshold))
    alpha = (xi / params.B).squeeze(-1)
    k_next = vi.policy_fn_sim(data["ashock"], data["ishock"],data["grid_k"], data["dist_k"], nn).view(-1,1, i_size).expand(-1, max_cols, i_size)
    yterm = ashock_3d * ishock_3d  * data["grid"]**params.theta
    numerator = params.nu * yterm / (wage + 1e-6)
    numerator = torch.clamp(numerator, min=1e-6, max=1e8)  # 数値の範囲を制限
    nnow = torch.pow(numerator, 1 / (1 - params.nu))
    inow = alpha * (k_next - (1-params.delta) * data["grid"])
    ynow = ashock_3d*ishock_3d * data["grid"]**params.theta * nnow**params.nu
    Iagg = torch.sum(data["dist"] * inow, dim=(1,2))
    Yagg = torch.sum(data["dist"]* ynow, dim=(1,2))
    Cagg = Yagg - Iagg
    target = 1 / Cagg
    return target
    




def price_loss(nn, data, params, mean):#k_gridに関してxiを求める他は適当でよい。
    eps = 1e-6
    i_size = params.ishock_gpu.size(0)
    max_cols = data["grid"].size(1)
    ashock_3d = data["ashock"].unsqueeze(1).expand(-1, max_cols, -1)#batch, max_cols, i_size
    ishock_3d = data["ishock"].unsqueeze(1).expand(-1, max_cols, -1)
    price = vi.price_fn(data["grid"], data["dist"], data["ashock"][:, 0],nn, mean)
    wage = params.eta/price
    wage = wage.unsqueeze(-1).expand(-1, max_cols, i_size)#batch, max_cols, i_size
    e0, e1 = next_value_gm(data, nn,params, data["grid"].size(1))#batch, max_cols, i_size
    e0_check = e0[0, :, :]
    e1_check = e1[0, :, :]
    threshold = (e0 - e1) / params.eta#batch, max_cols,i_size
    threshold_check = threshold[0, :, :]
    xi = torch.min(torch.tensor(params.B, dtype=TORCH_DTYPE), torch.max(torch.tensor(0, dtype=TORCH_DTYPE), threshold))#batch, max_cols,i_size
    alpha = (xi / params.B).squeeze(-1)#batch, max_cols,i_size
    alpha_check = alpha[0, :, :]
    k_next = vi.policy_fn_sim(data["ashock"], data["ishock"],data["grid_k"], data["dist_k"], nn).view(-1,1, i_size).expand(-1, max_cols, i_size)#batch, i_sizeで出てくるからmax_colsに合わせる
    yterm = ashock_3d * ishock_3d  * data["grid"]**params.theta#batch, max_cols, i_size
    numerator = params.nu * yterm / (wage + eps)
    numerator = torch.clamp(numerator, min=eps, max=1e8)  # 数値の範囲を制限
    nnow = torch.pow(numerator, 1 / (1 - params.nu))
    inow = alpha * (k_next - (1-params.delta) * data["grid"])
    inow_check = inow[0, :, :]
    ynow = ashock_3d*ishock_3d * data["grid"]**params.theta * nnow**params.nu
    ynow_check = ynow[0, :, :]
    Iagg = torch.sum(data["dist"] * inow, dim=(1,2))#batch
    Yagg = torch.sum(data["dist"]* ynow, dim=(1,2))#batch
    Cagg = Yagg - Iagg#batch
    Cagg = torch.clamp(Cagg, min=0.1)
    target = 1 / Cagg
    loss = F.huber_loss(price, target.unsqueeze(-1))
    #min_Iagg = params.min_Iagg  # Define a minimum threshold in params
    #penalty_weight = params.penalty_weight  # Define penalty weight in params
    #penalty = torch.relu(min_Iagg - Iagg)  # batch
    #penalty = penalty * penalty_weight 
    #penalty_Iagg = torch.mean(penalty)

    # Combine the main loss with the penalty
    #loss_total = loss + penalty_Cagg
    return loss

def price_train(data, params, nn, optimizer, num_epochs, batch_size, T, threshold, mean=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 学習率の設定
    initial_lr = 0.0005
    mid_lr = 0.00005
    final_lr = 0.00001

    # オプティマイザの設定
    if mean is None:
        optimizer = torch.optim.Adam(nn.params_price, lr=initial_lr)
    else:
        optimizer = torch.optim.Adam(nn.price_model.parameters(), lr=initial_lr)

    # データ準備
    ashock_data = vi.generate_ashock(1, T, params.ashock, params.pi_a).squeeze(0).unsqueeze(-1).expand(-1, params.ishock.size(0))
    ishock_data = params.ishock.unsqueeze(0).expand(T, -1)
    dataset = MyDataset(grid=data["grid"], dist=data["dist"], grid_k=data["grid_k"], dist_k=data["dist_k"],
                       ashock=ashock_data, ishock=ishock_data)
    valid_size = 64
    train_size = len(dataset) - valid_size
    train_data, valid_data = random_split(dataset, [train_size, valid_size])
    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, 32, shuffle=False)

    avg_val_loss = float('inf')
    epoch = 0

    # 学習率調整ステージ管理
    lr_stage = 0  # 0: initial_lr, 1: mid_lr, 2: final_lr

    while avg_val_loss > threshold and epoch < num_epochs:
        epoch += 1
        loss_list = []

        # トレーニングフェーズ
        nn.price_model.train()  
        nn.gm_model_price.train()
        for i, batch_data in enumerate(train_loader):
            # データをデバイスに移動
            batch_data = {key: value.to(device, dtype=TORCH_DTYPE) for key, value in batch_data.items()}
            optimizer.zero_grad()
            loss = price_loss(nn, batch_data, params, mean)
            loss.backward()
            optimizer.step()

        # バリデーションフェーズ
        nn.price_model.eval()  
        nn.gm_model_price.eval()
        with torch.no_grad():
            val_losses = []
            for v_data in valid_loader:
                v_data = {key: value.to(device, dtype=TORCH_DTYPE) for key, value in v_data.items()}
                val_loss = price_loss(nn, v_data, params, mean)
                val_losses.append(val_loss.item())
            avg_val_loss = sum(val_losses) / len(val_losses)

        print(f"Epoch: {epoch}, Avg Val Loss: {avg_val_loss:.6f}, LR: {optimizer.param_groups[0]['lr']}")

        # 学習率の調整
        if lr_stage == 0 and avg_val_loss < 0.005:
            for param_group in optimizer.param_groups:
                param_group['lr'] = mid_lr
            lr_stage = 1
            print(f"Learning rate adjusted to mid_lr: {mid_lr}")
        elif lr_stage == 1 and avg_val_loss < 0.002:
            for param_group in optimizer.param_groups:
                param_group['lr'] = final_lr
            lr_stage = 2
            print(f"Learning rate adjusted to final_lr: {final_lr}")

        # 学習率をさらに調整する条件を追加する場合はここに記述

    print("Training completed.")
        
def gm_fn(grid, dist, nn):
    grid_norm = (grid - params.k_grid_min) / (params.k_grid_max - params.k_grid_min)
    gm_tmp = nn.gm_model(grid_norm.unsqueeze(-1))#batch, k_grid, 1
    gm = torch.sum(gm_tmp * dist.unsqueeze(-1), dim=-2)#batch, 1
    return gm.squeeze(-1)

def policy_fn(ashock, ishock, grid_k, dist_k, nn):
    grid_norm = (grid_k - params.k_grid_mean) / params.k_grid_std
    gm_tmp = nn.gm_model_policy(grid_norm.unsqueeze(-1))
    gm = torch.sum(gm_tmp * dist_k.unsqueeze(-1), dim=-2).expand(-1, ishock.size(1)).unsqueeze(-1)
    state = torch.cat([ashock.unsqueeze(-1), ishock.unsqueeze(-1), gm], dim=-1)
    next_k = nn.policy(state)
    return next_k


def next_gm_fn(gm, ashock, nn):
    state = torch.cat([ashock, gm], dim=1)
    next_gm = nn.next_gm_model(state)
    return next_gm

def next_value_gm(data, nn, params, max_cols):#batch, max_cols, i_size, i*a, 4
    G = data["grid"].size(0)
    i_size = params.ishock_gpu.size(0)
    price = vi.price_fn(data["grid"], data["dist"], data["ashock"][:,0], nn)#batch, 1
    next_gm = vi.dist_gm(data["grid_k"], data["dist_k"], data["ashock"][:,0],nn)#batch, 1
    ashock_idx = [torch.where(params.ashock_gpu == val)[0].item() for val in data["ashock"][:,0]]#batch
    ashock_exp = params.pi_a_gpu[ashock_idx].to(device)#batch, 5
    prob = torch.einsum('ik,nj->nijk', params.pi_i_gpu, ashock_exp).unsqueeze(1).expand(G, max_cols, i_size, i_size, i_size)#batch, max_cols, i_size, a, i
    
    next_k = vi.policy_fn_sim(data["ashock"], data["ishock"], data["grid_k"], data["dist_k"], nn)#batch, i_size, 1
    next_k_expa = next_k.squeeze(-1).unsqueeze(1).expand(-1, max_cols, -1)#batch, max_cols, i_size, 
    a_mesh, i_mesh = torch.meshgrid(params.ashock_gpu, params.ishock_gpu, indexing='ij')  # indexing='ij' を明示的に指定
    a_flat = a_mesh.flatten()  # shape: [I*A]
    i_flat = i_mesh.flatten()  # shape: [I*A]
    a_5d = a_flat.view(1, 1, 1, -1, 1).expand(G, max_cols, i_size, -1 ,1)#batch, max_cols, i_size, i*a, 1
    i_5d = i_flat.view(1, 1, 1, -1, 1).expand(G, max_cols, i_size, -1, 1)#batch, max_cols, i_size, 1, i*a
    next_k_flat = next_k_expa.view(G, max_cols, i_size, 1, 1).expand(-1, -1, -1, a_flat.size(0), 1)#batch, max_cols, i_size, i*a, 1
    next_gm_flat = next_gm.view(G, 1, 1, 1, 1).expand(G, max_cols, i_size, a_flat.size(0), 1)#batch, max_cols, i_size, i*a, 1

    k_cross_flat = data["grid"].view(G, max_cols, i_size, 1, 1).expand(-1, -1, -1, a_flat.size(0), 1)#batch, max_cols, i_size, i*a, 1
    pre_k_flat = (1-params.delta) * k_cross_flat#batch, max_cols, i*a
    
    data_e0 = torch.stack([next_k_flat, a_5d, i_5d, next_gm_flat], dim=-1)#batch, max_cols, i_size, i*a, 4
    data_e1 = torch.stack([pre_k_flat, a_5d, i_5d, next_gm_flat], dim=-1)#batch, max_cols, i_size, i*a, 4
    
    value0 = nn.value0(data_e0).view(G, max_cols, i_size, len(params.ashock), len(params.ishock))#batch, max_cols, i_size, a, i
    value1 = nn.value0(data_e1).view(G, max_cols, i_size, len(params.ashock), len(params.ishock))#batch, max_cols, i_size, a, i

    expected_v0 = (value0 *  prob).sum(dim=(3,4))#batch, max_cols, i_size,
    expected_v1 = (value1 *  prob).sum(dim=(3,4))#batch, max_cols, i_size
    check0 = expected_v0[0, :, :]
    check1 = expected_v1[0, :, :]
    
    e0 = -next_k_expa * price.expand(-1, max_cols).unsqueeze(-1) + params.beta * expected_v0
    e1 = -(1-params.delta)*data["grid"] * price.expand(-1, max_cols).unsqueeze(-1) + params.beta * expected_v1
    
    return e0, e1
    
    
class NextGMDataset(Dataset):
    def __init__(self, gm, ashock):
        # 入力: gm[:, :-1], ターゲット: gm[:, 1:]
        inputs = gm[:-1]
        targets = gm[1:]
        ashock_reshaped = ashock[:-1]
        
        # 2列のテンソルに結合
        self.data = torch.stack((inputs, ashock_reshaped, targets), dim=-1)  # shape: (num_samples * (T-1), 3)
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        """
        Args:
            idx (int): サンプルのインデックス
        Returns:
            tuple: (input, target) のタプル
        """
        input_val = self.data[idx, 0:1].to(device)    # shape: ()
        ashock_val = self.data[idx, 1:2].to(device)   # shape: ()
        target_val = self.data[idx, 2:3].to(device)   # shape: ()
        return input_val, ashock_val, target_val
    

def next_gm_train(data, nn, params, optimizer, T, num_sample, epochs, save_plot_dir='plots'):
    # プロット用ディレクトリの作成
    os.makedirs(save_plot_dir, exist_ok=True)
    
    with torch.no_grad():
        ashock = torch.tensor(data["ashock"], dtype=TORCH_DTYPE)
        dist = [torch.tensor(value, dtype=TORCH_DTYPE) for value in data["dist_k"]]
        dist = torch.stack(dist, dim=0)
        grid = [torch.tensor(value, dtype=TORCH_DTYPE) for value in data["grid_k"]]
        grid = torch.stack(grid, dim=0)
        nn.gm_model.to("cpu")
        gm = gm_fn(grid, dist, nn)
        dataset = NextGMDataset(gm, ashock)
        valid_size = 64
        train_size = len(dataset) - valid_size
        train_data, valid_data = random_split(dataset, [train_size, valid_size])
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        valid_loader = DataLoader(valid_data, batch_size=32, shuffle=True)
        loss_list = []
    
    nn.gm_model.to(device)
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # トレーニングループ
        nn.gm_model.train()
        epoch_train_loss = 0
        for input, ashock_val, target in train_loader:
            optimizer.zero_grad()
            loss = next_gm_loss(nn, input, ashock_val, target)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
    
        # 検証ループ
        nn.gm_model.eval()
        with torch.no_grad():
            epoch_val_loss = 0
            val_inputs = []
            val_targets = []
            val_predictions = []
            for input, ashock_val, target in valid_loader:
                loss = next_gm_loss(nn, input, ashock_val, target)
                epoch_val_loss += loss.item()
    
                # データを収集
                val_inputs.append(input.cpu())
                val_targets.append(target.cpu())
                next_gm = next_gm_fn(input, ashock_val, nn)
                val_predictions.append(next_gm.cpu())
    
            avg_val_loss = epoch_val_loss / len(valid_loader)
            val_losses.append(avg_val_loss)
            print(f"Epoch: {epoch+1}, Train Loss: {avg_train_loss:.10f}, Validation Loss: {avg_val_loss:.10f}")
    
        # プロット（10エポックごとまたは最後のエポック）
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            val_inputs_tensor = torch.cat(val_inputs, dim=0).view(-1)
            val_targets_tensor = torch.cat(val_targets, dim=0).view(-1)
            val_predictions_tensor = torch.cat(val_predictions, dim=0).view(-1)
    
            plt.figure(figsize=(14, 6))
    
            # 真の値と予測値のプロット
            plt.subplot(1, 2, 1)
            sns.scatterplot(x=val_inputs_tensor.numpy(), y=val_targets_tensor.numpy(), label='True', alpha=0.5, s=40)
            sns.scatterplot(x=val_inputs_tensor.numpy(), y=val_predictions_tensor.numpy(), label='Predicted', alpha=0.5, s=40)
            plt.xlabel('Input')
            plt.ylabel('Output')
            plt.title(f'Epoch {epoch+1} Predictions')
            plt.legend()
    
            # ロスのプロット
            plt.subplot(1, 2, 2)
            epochs_range = range(1, epoch+2)
            sns.lineplot(x=epochs_range, y=train_losses, label='Train Loss')
            sns.lineplot(x=epochs_range, y=val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Loss over Epochs')
            plt.legend()
    
            plt.tight_layout()
            plot_filename = os.path.join(save_plot_dir, f'epoch_{epoch+1}.png')
            plt.savefig(plot_filename)
            plt.close()  # メモリを節約するためにプロットを閉じる
    
            print(f"Plot saved to {plot_filename}")
    
    # 最終的な損失曲線をプロット
    plt.figure(figsize=(8, 6))
    sns.lineplot(x=range(1, epochs+1), y=train_losses, label='Train Loss')
    sns.lineplot(x=range(1, epochs+1), y=val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Final Loss over Epochs')
    plt.legend()
    final_loss_plot = os.path.join(save_plot_dir, 'final_loss.png')
    plt.savefig(final_loss_plot)
    plt.close()
    print(f"Final loss plot saved to {final_loss_plot}")

def next_gm_loss(nn, input, ashock, target):
    next_gm = next_gm_fn(input, ashock, nn)
    loss = torch.mean((next_gm - target)**2)
    return loss



class MyDataset(Dataset):
    def __init__(self,k_cross=None, ashock=None, ishock=None, grid=None, dist=None, grid_k=None, dist_k=None):
        self.data = {}
        if k_cross is not None:
            if isinstance(k_cross, np.ndarray):
                k_cross = torch.tensor(k_cross, dtype=TORCH_DTYPE)
            k_cross = k_cross.view(-1, 1).squeeze(-1)
            self.data['k_cross'] = k_cross
        if ashock is not None:
            if isinstance(ashock, np.ndarray):
                ashock = torch.tensor(ashock, dtype=TORCH_DTYPE)
            #ashock = ashock.view(-1, 1)
            self.data['ashock'] = ashock
        if ishock is not None:
            if isinstance(ishock, np.ndarray):
                ishock = torch.tensor(ishock, dtype=TORCH_DTYPE)
            #ishock = ishock.view(-1, 1)
            self.data['ishock'] = ishock
        if grid is not None:
            grid = [torch.tensor(data, dtype=TORCH_DTYPE) for data in grid]
            self.data['grid'] = torch.stack(grid, dim=0)
            
        if dist is not None:
            dist = [torch.tensor(data, dtype=TORCH_DTYPE) for data in dist]
            self.data['dist'] = torch.stack(dist, dim=0)
        
        if grid_k is not None:
            grid_k = [torch.tensor(data, dtype=TORCH_DTYPE) for data in grid_k]
            self.data['grid_k'] = torch.stack(grid_k, dim=0)
        
        if dist_k is not None:
            dist_k = [torch.tensor(data, dtype=TORCH_DTYPE) for data in dist_k]
            self.data['dist_k'] = torch.stack(dist_k, dim=0)
        

    def __len__(self):
        # 使用しているデータの最初の項目の長さを返す
        return next(iter(self.data.values())).shape[0]

    def __getitem__(self, idx):
        # データが存在する場合のみ項目を返す
        return {key: value[idx] for key, value in self.data.items()}

def padding(list_of_arrays):
    max_row = max(array.size(0) for array in list_of_arrays)
    padded_arrays = []
    for array in list_of_arrays:
        # 行方向にパディングを追加
        pad_size = max_row - array.size(0)
        padded_array = F.pad(array, (0, 0, 0, pad_size), mode='constant', value=0)
        padded_arrays.append(padded_array)
    
    data = torch.stack(padded_arrays, dim=0)
    return data

def next_gm_init(nn, params, optimizer, num_epochs, num_sample,T):
    K_cross = np.random.choice(params.K_grid_np, num_sample* T)
    ashock = vi.generate_ashock(num_sample, T, params.ashock, params.pi_a).view(-1, 1).squeeze(-1)
    dataset = Valueinit(ashock=ashock, K_cross=K_cross,target_attr='K_cross', input_attrs=['ashock', 'K_cross'])
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    for epoch in range(num_epochs):
        for train_data in dataloader:
            train_data['X'] = train_data['X'].to(device, dtype=TORCH_DTYPE)
            train_data['y'] = train_data['y'].to(device, dtype=TORCH_DTYPE)
            optimizer.zero_grad()
            next_gm = next_gm_fn(train_data['X'][:, 0:1], train_data['X'][:, 1:2], nn).squeeze(-1)
            loss = F.mse_loss(next_gm, train_data['y'])
            loss.backward()
            optimizer.step()


class Valueinit(Dataset):
    def __init__(self, k_cross=None, ashock=None, ishock=None, K_cross=None, target_attr='k_cross', input_attrs=None):
        
        if k_cross is not None:
            if not isinstance(k_cross, torch.Tensor):
                k_cross = torch.tensor(k_cross, dtype=TORCH_DTYPE)
            self.k_cross = k_cross.view(-1, 1).squeeze(-1)

        if ashock is not None:
            if not isinstance(ashock, torch.Tensor):
                ashock = torch.tensor(ashock, dtype=TORCH_DTYPE)
            self.ashock = ashock

        if ishock is not None:
            if not isinstance(ishock, torch.Tensor):
                ishock = torch.tensor(ishock, dtype=TORCH_DTYPE)
            self.ishock = ishock

        if K_cross is not None:
            if not isinstance(K_cross, torch.Tensor):
                K_cross = torch.tensor(K_cross, dtype=TORCH_DTYPE)
            self.K_cross = K_cross.view(-1, 1).squeeze(-1)

        # Validate target_attr and set it
        if target_attr not in ['k_cross', 'ashock', 'ishock', 'K_cross']:
            raise ValueError(f"Invalid target_attr: {target_attr}. Must be one of 'k_cross', 'ashock', 'ishock', 'K_cross'.")
        self.target_attr = target_attr

        # Set input attributes
        if input_attrs is None:
            # Default to using all attributes if not specified
            self.input_attrs = ['k_cross', 'ashock', 'ishock', 'K_cross']
        else:
            # Validate input attributes
            for attr in input_attrs:
                if attr not in ['k_cross', 'ashock', 'ishock', 'K_cross']:
                    raise ValueError(f"Invalid input attribute: {attr}. Must be one of 'k_cross', 'ashock', 'ishock', 'K_cross'.")
            self.input_attrs = input_attrs

    def __len__(self):
        # Find the first non-None attribute and return its length
        for attr in ['k_cross', 'ashock', 'ishock', 'K_cross']:
            data = getattr(self, attr, None)
            if data is not None:
                return len(data)
        raise ValueError("No valid data attributes were provided. Dataset length cannot be determined.")
    
    def __getitem__(self, idx):
        # Stack only the attributes specified in input_attrs
        inputs = [getattr(self, attr)[idx] for attr in self.input_attrs]
        X = torch.stack(inputs, dim=-1)
        y = getattr(self, self.target_attr)[idx]  # Use the attribute specified by target_attr
        return {'X': X, 'y': y}