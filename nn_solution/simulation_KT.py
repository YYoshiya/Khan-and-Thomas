import numpy as np
from scipy.interpolate import RectBivariateSpline
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


def simul_shocks(n_sample, T, Z, Pi, state_init=None):
    nz = len(Z)
    ashock = np.zeros([n_sample, T], dtype=int)  # ショックインデックスの格納用
    Pi = Pi / Pi.sum(axis=1, keepdims=True)

    if state_init is not None:
        ashock[:, 0] = state_init  # 初期状態を設定
    else:
        # 初期状態をランダムに決定（均等分布）
        ashock[:, 0] = np.random.choice(nz, size=n_sample)

    for t in range(1, T):
        current_states = ashock[:, t - 1]
        ashock[:, t] = [np.random.choice(nz, p=Pi[state]) for state in current_states]

    # ショックインデックスから実際のショック値に変換
    ashock_values = Z[ashock]

    return ashock_values

#CPUに移すのとか忘れずに。
def simul_k(n_sample, T, mparam, policy_fn_true, policy_type, price_fn, state_init=None, shocks=None): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if shocks is not None:
        ashock = torch.tensor(shocks, device=device)
        assert n_sample == ashock.shape[0], "n_sample is inconsistent with given shocks."
        assert T == ashock.shape[1], "T is inconsistent with given shocks."
        if state_init:
            assert torch.equal(ashock[..., 0:1], torch.tensor(state_init["ashock"], device=device)) and \
                "Shock inputs are inconsistent with state_init"
    else:
        ashock = simul_shocks(n_sample, T, mparam.Z, mparam.Pi, state_init).to(device)
    
    n_agt = mparam.n_agt
    k_cross = torch.zeros(n_sample, n_agt, T, device=device)
    
    if state_init:
        assert n_sample == state_init["k_cross"].shape[0], "n_sample is inconsistent with state_init."
        k_cross[:, :, 0] = torch.tensor(state_init["k_cross"], device=device)
    else:
        k_cross[:, :, 0] = mparam.k_ss.to(device)
    
    #price_fn.to(device)  
    #policy.to(device)  policy_fnにするときに面倒だからここじゃなくて最初に送っとくべき。
    
    if policy_type == "nn_share":
        for t in range(1, T):
            price = price_fn(k_cross[:, :, t-1])
            wage = mparam.eta / price
            yterm = ashock[:, t-1].unsqueeze(1) * k_cross[:, :, t-1]**mparam.theta
            n = (mparam.nu * yterm / wage.unsqueeze(1))**(1 / (1 - mparam.nu))
            y = yterm * n**mparam.nu
            v0_temp = y - wage.unsqueeze(1) * n + (1 - mparam.delta) * k_cross[:, :, t-1]
            v0 = v0_temp * price.unsqueeze(1)
            k_cross[:, :, t] = policy_fn_true(k_cross[:, :, t - 1], ashock[:, t - 1])
    
    simul_data = {
        "price": price.cpu().numpy(),
        "v0": v0.cpu().numpy(),
        "k_cross": k_cross.cpu().numpy(),
        "ashock": ashock.cpu().numpy()
    }
    
    return simul_data

def init_simul_k(n_sample, T, mparam, policy, policy_type, price_fn, state_init=None, shocks=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if shocks is not None:
        ashock = torch.tensor(shocks, device=device)
        assert n_sample == ashock.shape[0], "n_sample is inconsistent with given shocks."
        assert T == ashock.shape[1], "T is inconsistent with given shocks."
        if state_init:
            assert torch.equal(ashock[..., 0:1], torch.tensor(state_init["ashock"], device=device)) and \
                "Shock inputs are inconsistent with state_init"
    else:
        ashock = torch.tensor(simul_shocks(n_sample, T, mparam.Z, mparam.Pi, state_init), device=device)
    
    n_agt = mparam.n_agt
    k_cross = torch.zeros(n_sample, n_agt, T, device=device)
    k_mean = torch.zeros(n_sample, T, device=device)
    if state_init:
        assert n_sample == state_init["k_cross"].shape[0], "n_sample is inconsistent with state_init."
        k_cross[:, :, 0] = torch.tensor(state_init["k_cross"], device=device)
    else:
        k_cross[:, :, 0] = torch.tensor(mparam.k_ss).to(device)
    k_mean[:, 0] = k_cross[:, :, 0].mean(dim=1)

    if policy_type == "nn_share":
        for t in range(1, T):
            price = price_fn(k_cross[:, :, t-1])
            wage = mparam.eta / price
            yterm = ashock[:, t-1].unsqueeze(1) * k_cross[:, :, t-1]**mparam.theta
            n = (mparam.nu * yterm / wage.unsqueeze(1))**(1 / (1 - mparam.nu))
            y = yterm * n**mparam.nu
            v0_temp = y - wage.unsqueeze(1) * n + (1 - mparam.delta) * k_cross[:, :, t-1]
            v0 = v0_temp * price.unsqueeze(1)
            k_cross[:, :, t] = policy(k_cross[:, :, t - 1], k_mean[:, t-1], ashock[:, t - 1])
            k_mean[:, t] = k_cross[:, :, t].mean(dim=1)
    
    simul_data = {
    "price": price.cpu().numpy(),
    "v0": v0.cpu().numpy(),
    "k_cross": k_cross.cpu().numpy(),
    "ashock": ashock.cpu().numpy()
}
    
    return simul_data

class PolicyDataset(Dataset):
    def __init__(self, data):
        """
        data: numpy配列
        入力 (X): grid_k, grid_K, ashock
        ターゲット (y): grid_k
        """
        self.X = torch.tensor(data, dtype=torch.float32)        # grid_k, grid_K, ashock
        self.y = torch.tensor(data[:, 0], dtype=torch.float32) # grid_k

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def initial_policy(model, mparam, num_epochs=100, batch_size=50):
    # ショックをシミュレーション
    ashock = simul_shocks(n_sample=1, T=500, Z=mparam.Z, Pi=mparam.Pi, state_init=None)

    # グリッドの作成
    grid_k = np.linspace(0.1, 3.0, 100)
    grid_K = np.linspace(1.0, 5.0, 10)

    # grid_k と grid_K を500個に繰り返す
    repeats_k = int(np.ceil(500 / len(grid_k)))
    grid_k_seq = np.tile(grid_k, repeats_k)[:500]

    repeats_K = int(np.ceil(500 / len(grid_K)))
    grid_K_seq = np.tile(grid_K, repeats_K)[:500]

    # ashock をフラットにして500個に切り取る
    ashock_seq = ashock.flatten()[:500]

    # 各シーケンスを列として結合
    data = np.column_stack([grid_k_seq, grid_K_seq, ashock_seq])

    # カスタムデータセットの作成
    dataset = PolicyDataset(data)

    # DataLoaderの作成
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # モデル、損失関数、オプティマイザーの設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.to(device)
    # 学習ループの実装
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # フォワードパス
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)

            # バックワードパスとオプティマイザーのステップ
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_X.size(0)

        epoch_loss /= len(dataloader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    