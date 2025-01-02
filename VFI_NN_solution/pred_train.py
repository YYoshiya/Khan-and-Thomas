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

    
def map_to_grid(k_prime, k_grid):
    """
    Map k_prime (B,G,I) to the capital grid (G,) using linear interpolation.
    Returns idx_lower, idx_upper, weight all as (B,G,I) tensors.

    Parameters:
    - k_prime: (B,G,I) Tensor of new capital values
    - k_grid: (G,) 1D capital grid (sorted, ascending)

    Returns:
    - idx_lower: (B,G,I)
    - idx_upper: (B,G,I)
    - weight: (B,G,I)
    """
    B, G, I = k_prime.shape
    grid_size = k_grid.size(0)
    k_min = k_grid[0]
    k_max = k_grid[-1]

    # Flatten k_prime for search
    k_prime_flat = k_prime.reshape(-1)
  # (B*G*I,)

    # Searchsorted
    idx = torch.searchsorted(k_grid, k_prime_flat)
    idx = idx.view(B, G, I)

    # Clamp indices
    idx = torch.clamp(idx, 0, grid_size - 1)

    # Define lower and upper indices
    idx_lower = torch.clamp(idx - 1, 0, grid_size - 1)
    idx_upper = idx

    # Expand k_grid for easy gathering: (B,G,I) 
    # We replicate the 1D k_grid along B and I dims so we can gather directly.
    k_grid_expanded = k_grid.view(1, grid_size, 1).expand(B, grid_size, I)  # (B,G,I)

    # Gather k_lower and k_upper
    # gather along dim=1 (the grid dimension), idx_* must have the same shape as we gather from
    k_lower = torch.gather(k_grid_expanded, 1, idx_lower)
    k_upper = torch.gather(k_grid_expanded, 1, idx_upper)

    # Compute weights
    denom = k_upper - k_lower
    zero_denom_mask = denom.abs() < 1e-8
    denom = denom + zero_denom_mask * 1e-8
    weight = (k_prime - k_lower) / denom

    # Handle out-of-bound cases
    weight = torch.where(k_prime <= k_min, torch.zeros_like(weight), weight)
    weight = torch.where(k_prime >= k_max, torch.ones_like(weight), weight)

    idx_lower = torch.where(k_prime <= k_min, torch.zeros_like(idx_lower), idx_lower)
    idx_upper = torch.where(k_prime <= k_min, torch.zeros_like(idx_upper), idx_upper)

    idx_lower = torch.where(k_prime >= k_max, (grid_size - 1) * torch.ones_like(idx_lower), idx_lower)
    idx_upper = torch.where(k_prime >= k_max, (grid_size - 1) * torch.ones_like(idx_upper), idx_upper)

    # Ensure integer indices
    idx_lower = idx_lower.long()
    idx_upper = idx_upper.long()

    # Clamp weights to [0, 1]
    weight = torch.clamp(weight, 0.0, 1.0)

    return idx_lower, idx_upper, weight
#これpi_iのとこにエラー出る
def update_distribution(dist_new, dist_now, alpha, idx_lower, idx_upper, weight, pi_i, adjusting):
    """
    Update the distribution with capital transitions and state transitions in a (B,G,I) setting.

    Parameters:
    - dist_new: (B,G,I) Tensor, distribution at next iteration (will be updated)
    - dist_now: (B,G,I) Tensor, current distribution
    - alpha: scalar or (B,G,I)-broadcastable Tensor, adjustment rate
    - idx_lower: (B,G,I) Tensor, lower grid indices
    - idx_upper: (B,G,I) Tensor, upper grid indices
    - weight: (B,G,I) Tensor, interpolation weights
    - pi_i: (I,I) Tensor, transition probabilities between states i->i_prime
    - adjusting: bool, if True use dist_now*alpha else dist_now*(1-alpha)

    Returns:
    - None (dist_new is updated in-place)
    """
    B, G, I = dist_now.shape

    if adjusting:
        dist_adjust = dist_now * alpha  # (B,G,I)
    else:
        dist_adjust = dist_now * (1 - alpha)  # (B,G,I)

    # Loop over i and i_prime as in the original logic
    # We'll do scatter_add_ on dist_new for each i_prime
    # dist_new is (B,G,I), we select dist_new for each i_prime: dist_new[..., i_prime] is (B,G)
    for i in range(I):
        # Extract the indexing and weight for this i
        idx_l_i = idx_lower[:, :, i]  # (B,G)
        idx_u_i = idx_upper[:, :, i]  # (B,G)
        w_i = weight[:, :, i]         # (B,G)
        
        # dist_adjust[..., i] is (B,G)
        # For each i_prime we distribute mass according to pi_i[i, i_prime]
        for i_prime in range(I):
            pi_val = pi_i[i, i_prime]

            # dist_contrib: how much mass flows from state i to i_prime
            dist_contrib = dist_adjust[:, :, i] * pi_val  # (B,G)

            # dist_new_i_prime is (B,G), slice out along i_prime dimension
            dist_new_i_prime = dist_new[..., i_prime]

            # Add mass to lower index
            dist_new_i_prime.scatter_add_(1, idx_l_i, dist_contrib * (1 - w_i))
            # Add mass to upper index
            dist_new_i_prime.scatter_add_(1, idx_u_i, dist_contrib * w_i)


def bisectp(nn, params, data, init=None):
    diff = torch.full((data["grid"].size(0),), 1, dtype=TORCH_DTYPE).to(device)
    if init is not None:
        p_init = torch.full((data["grid"].size(0),), init, dtype=TORCH_DTYPE).to(device)
        # -0.1から0.1のランダムな値を生成
        random_values = (torch.rand(data["grid"].size(0), dtype=TORCH_DTYPE) * 0.2 - 0.1).to(device)
        
        # p_initにランダムな値を加算
        p_init = p_init + random_values
        pL = p_init - 0.1
        pH = p_init + 0.1
    else:
        p_init = vi.price_fn(data["grid"], data["dist"], data["ashock"], nn).squeeze(-1)
        pL = p_init - 0.1
        pH = p_init + 0.1

    # 「大きく外れているところのために範囲を再設定する」工程を何度か繰り返す
    # （ここでは最大2回やる例）
    max_outer_loop = 5
    outer_iter = 0

    while diff.max() > params.critbp and outer_iter < max_outer_loop:
        iter_count = 0  # 内側のバイセクション反復回数をカウント
        while diff.max() > params.critbp and iter_count < 20:
            p0 = (pL + pH) / 2
            pnew, dist_new = eq_price(nn, data, params, p0)
            B0 = p0 - pnew

            # ① pnew < 0 の箇所は pL = p0
            negative_mask = (pnew < 0)
            pL = torch.where(negative_mask, p0, pL)
            # pH は変えず

            # ② pnew >= 0 の箇所だけ B0 に基づいて更新
            nonnegative_mask = (pnew >= 0)
            pL = torch.where(nonnegative_mask & (B0 < 0), p0, pL)
            pH = torch.where(nonnegative_mask & (B0 >= 0), p0, pH)

            diff = torch.abs(B0)
            iter_count += 1

        # ここで 20 回試してもまだ diff > 0.1 のところがある場合、探索範囲を広げる
        outer_iter += 1
        still_large_mask = (diff > 0.0001)
        if still_large_mask.any():
            pL[still_large_mask] = p_init[still_large_mask] - 0.1 * (outer_iter+1)
            pH[still_large_mask] = p_init[still_large_mask] + 0.1 * (outer_iter+1)

          # 外側ループを回す
    print(f"Outer Iteration: {outer_iter}")
    return pnew.to("cpu"), dist_new.to("cpu"), outer_iter


def eq_price(nn, data, params, price):
    i_size = params.ishock_gpu.size(0)
    max_cols = data["grid"].size(1)
    ashock_3d = params.ishock_gpu.view(1, 1, i_size).expand(data["grid"].size(0), max_cols, -1)
    ishock_3d = params.ishock_gpu.view(1, 1, i_size).expand(data["grid"].size(0), max_cols, -1)
    price_policy = price.view(-1, 1).expand(-1, i_size)
    price = price.view(-1, 1, 1).expand(-1, max_cols, i_size)
    
    wage = params.eta/price
    e0, e1 = next_value_price(data, nn, params, max_cols, price_policy)#作らなきゃいけない。
    threshold = (e0 - e1) / params.eta
    xi = torch.min(torch.tensor(params.B, dtype=TORCH_DTYPE).to(device), torch.max(torch.tensor(0, dtype=TORCH_DTYPE).to(device), threshold))
    alpha = (xi / params.B).squeeze(-1)
    ishock_2d = params.ishock_gpu.unsqueeze(0).expand(data["grid"].size(0), -1)
    ashock_2d = data["ashock"].unsqueeze(1).expand(-1, i_size)
    next_k = vi.policy_fn_sim(ashock_2d, ishock_2d, data["grid_k"], data["dist_k"], price_policy, nn).view(-1,1, i_size).expand(-1, max_cols, i_size)
    k_next_non_adj = (1-params.delta) * params.k_grid_gpu.view(1, -1, i_size).expand(data["grid"].size(0), -1, -1)
    idx_adj_lower, idx_adj_upper, weight_adj = map_to_grid(next_k, params.k_grid_1d_gpu)
    idx_non_adj_lower, idx_non_adj_upper, weight_non_adj = map_to_grid(k_next_non_adj, params.k_grid_1d_gpu)
    dist_new = torch.zeros_like(data["dist"])
    update_distribution(dist_new, data["dist"], alpha, idx_adj_lower, idx_adj_upper, weight_adj, params.pi_i_gpu, True)
    update_distribution(dist_new, data["dist"], alpha, idx_non_adj_lower, idx_non_adj_upper, weight_non_adj, params.pi_i_gpu, False)
    yterm = ashock_3d * ishock_3d  * data["grid"]**params.theta
    numerator = params.nu * yterm / (wage + 1e-6)
    numerator = torch.clamp(numerator, min=1e-6, max=1e8)  # 数値の範囲を制限
    nnow = torch.pow(numerator, 1 / (1 - params.nu))
    inow = alpha * (next_k - (1-params.delta) * data["grid"])
    ynow = ashock_3d*ishock_3d * data["grid"]**params.theta * nnow**params.nu
    Iagg = torch.sum(data["dist"] * inow, dim=(1,2))
    Yagg = torch.sum(data["dist"]* ynow, dim=(1,2))
    Cagg = Yagg - Iagg
    target = 1 / Cagg
    return target, dist_new

def next_value_price(data, nn, params, max_cols, price):#batch, max_cols, i_size, i*a, 4
    G = data["grid"].size(0)
    i_size = params.ishock_gpu.size(0)
    
    next_gm = vi.dist_gm(data["grid_k"], data["dist_k"], data["ashock"],nn)#batch, 1
    ashock_idx = [torch.where(params.ashock_gpu == val)[0].item() for val in data["ashock"]]#batch
    ashock_exp = params.pi_a_gpu[ashock_idx].to(device)#batch, 5
    prob = torch.einsum('ik,nj->nijk', params.pi_i_gpu, ashock_exp).unsqueeze(1).expand(G, max_cols, i_size, i_size, i_size)#batch, max_cols, i_size, a, i
    
    ishock_2d = params.ishock_gpu.unsqueeze(0).expand(data["grid"].size(0), -1)
    ashock_2d = data["ashock"].unsqueeze(1).expand(-1, i_size)
    next_k = vi.policy_fn_sim(ashock_2d, ishock_2d, data["grid_k"], data["dist_k"],price, nn)#batch, i_size, 1
    next_k_expa = next_k.squeeze(-1).unsqueeze(1).expand(-1, max_cols, -1)#batch, max_cols, i_size, 
    a_mesh, i_mesh = torch.meshgrid(params.ashock_gpu, params.ishock_gpu, indexing='ij')  # indexing='ij' を明示的に指定
    a_mesh_norm = (a_mesh - params.ashock_min) / (params.ashock_max - params.ashock_min)
    i_mesh_norm = (i_mesh - params.ishock_min) / (params.ishock_max - params.ishock_min)
    a_flat = a_mesh_norm.flatten()  # shape: [I*A]
    i_flat = i_mesh_norm.flatten()  # shape: [I*A]
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
    
    price = price.view(-1, 1, i_size).expand(-1, max_cols, i_size)
    e0 = -next_k_expa * price + params.beta * expected_v0
    e1 = -(1-params.delta)*data["grid"] * price + params.beta * expected_v1
    
    return e0, e1

class Pred_Dataset(Dataset):
    def __init__(self, grid, dist, ashock, price):
        grid = [torch.tensor(data, dtype=TORCH_DTYPE) for data in grid]
        self.grid = torch.stack(grid, dim=0)
        dist = [torch.tensor(data, dtype=TORCH_DTYPE) for data in dist]
        self.dist = torch.stack(dist, dim=0)
        self.ashock = torch.tensor(ashock, dtype=TORCH_DTYPE)
        price = [torch.tensor(data, dtype=TORCH_DTYPE) for data in price]
        self.target = torch.stack(price, dim=0)
    
    def __len__(self):
        return self.grid.size(0)
    
    def __getitem__(self, idx):
        return self.grid[idx].to(device), self.dist[idx].to(device), self.ashock[idx].to(device), self.target[idx].to(device)
    

def price_train(data, nn, num_epochs):
    with torch.no_grad():
        train_dataset = Pred_Dataset(data["grid"], data["dist"], data["ashock"], data["price"])
        valid_size = 64
        train_size = len(train_dataset) - valid_size
        train_data, valid_data = random_split(train_dataset, [train_size, valid_size])
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        valid_loader = DataLoader(valid_data, batch_size=32, shuffle=True)
        optimizer = optim.Adam(nn.params_price, lr=0.01)
        valid_loss = 1
        epoch = 0
    while epoch < num_epochs and valid_loss > 1e-6:
        epoch += 1
        for grid, dist, ashock, target in train_loader:
            optimizer.zero_grad()
            pred = vi.price_fn(grid, dist, ashock, nn).squeeze(-1)
            loss = F.mse_loss(pred, target)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            for grid, dist, ashock, target in valid_loader:
                pred = vi.price_fn(grid, dist, ashock, nn).squeeze(-1)
                valid_loss = F.mse_loss(pred, target)
            if epoch % 10 == 0:
                print(f"Epoch: {epoch}, Train Loss: {loss.item()}, Validation Loss: {valid_loss.item()}")
    
def gm_fn(grid, dist, nn):
    grid_norm = (grid - params.k_grid_min) / (params.k_grid_max - params.k_grid_min)
    gm_tmp = nn.gm_model(grid_norm.unsqueeze(-1))#batch, k_grid, 1
    gm = torch.sum(gm_tmp * dist.unsqueeze(-1), dim=-2)#batch, 1
    return gm.squeeze(-1)



def next_gm_fn(gm, ashock, nn):
    ashock_norm = (ashock - params.ashock_min) / (params.ashock_max - params.ashock_min)
    state = torch.cat([ashock_norm, gm], dim=1)
    next_gm = nn.next_gm_model(state)
    return next_gm
    
    
class NextGMDataset(Dataset):
    def __init__(self, gm, ashock, next_gm):
        # 入力: gm[:, :-1], ターゲット: gm[:, 1:]
        self.gm = gm
        self.ashock = ashock
        self.next_gm = next_gm
    
    def __len__(self):
        return self.gm.size(0)
    
    def __getitem__(self, idx):
        return self.gm[idx].to(device), self.ashock[idx].to(device), self.next_gm[idx].to(device)
    

def next_gm_train(data, nn, params, optimizer, T, num_sample, epochs, save_plot_dir='results/next_gm'):
    # プロット用ディレクトリの作成
    os.makedirs(save_plot_dir, exist_ok=True)
    
    with torch.no_grad():
        ashock = torch.tensor(data["ashock"], dtype=TORCH_DTYPE)
        dist = [torch.tensor(value, dtype=TORCH_DTYPE) for value in data["dist_k"]]
        dist = torch.stack(dist, dim=0)
        grid = [torch.tensor(value, dtype=TORCH_DTYPE) for value in data["grid_k"]]
        grid = torch.stack(grid, dim=0)
        nn.gm_model.to("cpu")
        gm = gm_fn(grid, dist, nn)  # 全体を取得
        next_gm = gm[1:]  # 次期値
        gm = gm[:-1]  # 現在値
        dataset = NextGMDataset(gm, ashock, next_gm)
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
                next_gm = next_gm_fn(input.unsqueeze(-1), ashock_val.unsqueeze(-1), nn)
                val_predictions.append(next_gm.cpu())

            avg_val_loss = epoch_val_loss / len(valid_loader)
            val_losses.append(avg_val_loss)
            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch: {epoch+1}, Train Loss: {avg_train_loss:.10f}, Validation Loss: {avg_val_loss:.10f}")
    
    # 最後のエポックのプロットを保存
    nn.gm_model.eval()
    with torch.no_grad():
        val_inputs = []
        val_targets = []
        val_predictions = []
        for input, ashock_val, target in valid_loader:
            val_inputs.append(input.cpu())
            val_targets.append(target.cpu())
            next_gm = next_gm_fn(input.unsqueeze(-1), ashock_val.unsqueeze(-1), nn)
            val_predictions.append(next_gm.cpu())
    
    val_inputs_tensor = torch.cat(val_inputs, dim=0).view(-1)
    val_targets_tensor = torch.cat(val_targets, dim=0).view(-1)
    val_predictions_tensor = torch.cat(val_predictions, dim=0).view(-1)

    plt.figure(figsize=(8, 6))

    # 真の値と予測値のプロット
    sns.scatterplot(x=val_inputs_tensor.numpy(), y=val_targets_tensor.numpy(), label='True', alpha=0.5, s=40)
    sns.scatterplot(x=val_inputs_tensor.numpy(), y=val_predictions_tensor.numpy(), label='Predicted', alpha=0.5, s=40)
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.title(f'Final Epoch Predictions (Epoch {epochs})')
    plt.legend()

    plt.tight_layout()
    plot_filename = os.path.join(save_plot_dir, f'epoch_{epochs}.png')
    plt.savefig(plot_filename)
    plt.close()  # メモリを節約するためにプロットを閉じる

    print(f"Final epoch plot saved to {plot_filename}")

    # 最終的な損失曲線をプロット（オプション）
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

def next_gm_loss(nn, gm, ashock, target):
    next_gm = next_gm_fn(gm.unsqueeze(-1), ashock.unsqueeze(-1), nn).squeeze(-1)
    loss = F.mse_loss(next_gm, target)
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