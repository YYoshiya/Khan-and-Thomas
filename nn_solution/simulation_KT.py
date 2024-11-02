import numpy as np
from scipy.interpolate import RectBivariateSpline
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import random


DTYPE = "float32"
if DTYPE == "float64":
    NP_DTYPE = np.float64
    TORCH_DTYPE = torch.float64  # PyTorchのデータ型を指定
elif DTYPE == "float32":
    NP_DTYPE = np.float32
    TORCH_DTYPE = torch.float32  # PyTorchのデータ型を指定
else:
    raise ValueError("Unknown dtype.")

def simul_shocks(n_sample, T, mparam, state_init=None):
    nz = len(mparam.Z)
    ashock = np.zeros([n_sample, T], dtype=int)  # ショックインデックスの格納用
    mparam.Pi = mparam.Pi / mparam.Pi.sum(axis=1, keepdims=True)
    n_agt = mparam.n_agt
    if state_init is not None:
        ashock[:, 0:1] = state_init["ashock"]  # 初期状態を設定
    else:
        # 初期状態をランダムに決定（均等分布）
        ashock[:, 0] = np.random.choice(nz, size=n_sample)

    for t in range(1, T):
        current_states = ashock[:, t - 1]
        ashock[:, t] = [np.random.choice(nz, p=mparam.Pi[state]) for state in current_states]

    # ショックインデックスから実際のショック値に変換
    ashock_values = mparam.Z[ashock]
    
    xi = np.random.uniform(0, mparam.B, size=(n_sample, n_agt, T))

    return ashock_values, xi

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
        ashock, xi = simul_shocks(n_sample, T, mparam, state_init)
    
    n_agt = mparam.n_agt
    k_cross = np.zeros((n_sample, n_agt, T))
    price = np.zeros((n_sample, T))
    v0 = np.zeros((n_sample, n_agt, T-1))
    
    if state_init:
        assert n_sample == state_init["k_cross"].shape[0], "n_sample is inconsistent with state_init."
        k_cross[:, :, 0] = state_init["k_cross"]
    else:
        k_cross[:, :, 0:1] = mparam.k_ss
    
    #price_fn.to(device)  
    #policy.to(device)  policy_fnにするときに面倒だからここじゃなくて最初に送っとくべき。
    
    if policy_type == "nn_share":
        for t in range(1, T):
            price[:,t-1:t] = price_fn(k_cross[:,:,t-1], ashock[:,t-1:t]).detach().cpu().numpy()
            wage = mparam.eta / price[:, t-1:t]#384,1
            yterm = ashock[:, t-1:t] * k_cross[:, :, t-1]**mparam.theta#384,50
            n = (mparam.nu * yterm / wage)**(1 / (1 - mparam.nu))
            y = yterm * n**mparam.nu
            v0_temp = y - wage * n + (1 - mparam.delta) * k_cross[:, :, t-1]
            k_cross[:, :, t:t+1] = policy_fn_true(k_cross[:,:,t-1:t], ashock[:, t-1:t], xi[:,:,t-1:t]).detach().cpu().clamp(min=0.1).numpy()
            v0[:,:,t-1] = np.where((1-mparam.delta)*k_cross[:,:,t-1]==k_cross[:,:,t], (y-wage*n)* price[:, t-1:t], v0_temp*price[:, t-1:t]-xi[:,:,t-1]*wage*price[:,t-1:t] - mparam.GAMY*price[:, t-1:t]*k_cross[:,:,t])
    
    simul_data = {
        "price": price,
        "v0": v0,
        "k_cross": k_cross,
        "ashock": ashock,
        "xi": xi
    } 
    
    return simul_data

def init_simul_k(n_sample, T, mparam, policy_fn_true, policy_type, price_fn, state_init=None, shocks=None): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if shocks is not None:
        ashock = torch.tensor(shocks, device=device)
        assert n_sample == ashock.shape[0], "n_sample is inconsistent with given shocks."
        assert T == ashock.shape[1], "T is inconsistent with given shocks."
        if state_init:
            assert torch.equal(ashock[..., 0:1], torch.tensor(state_init["ashock"], device=device)) and \
                "Shock inputs are inconsistent with state_init"
    else:
        ashock, xi = simul_shocks(n_sample, T, mparam, state_init)
    
    n_agt = mparam.n_agt
    k_cross = np.zeros((n_sample, n_agt, T))
    k_mean = np.zeros((n_sample, T))
    price = np.zeros((n_sample, T))
    v0 = np.zeros((n_sample, n_agt, T-1))
    
    if state_init:
        assert n_sample == state_init["k_cross"].shape[0], "n_sample is inconsistent with state_init."
        k_cross[:, :, 0] = state_init["k_cross"]
    else:
        k_cross[:, :, 0:1] = mparam.k_ss
    k_mean[:, 0] = k_cross[:, :, 0].mean(axis=1)
    
    #price_fn.to(device)  
    #policy.to(device)  policy_fnにするときに面倒だからここじゃなくて最初に送っとくべき。
    
    if policy_type == "nn_share":
        for t in range(1, T):
            price[:,t-1:t] = price_fn(k_cross[:,:,t-1], ashock[:,t-1:t]).detach().cpu().numpy()
            wage = mparam.eta / price[:, t-1:t]#384,1
            yterm = ashock[:, t-1:t] * k_cross[:, :, t-1]**mparam.theta#384,50
            n = (mparam.nu * yterm / wage)**(1 / (1 - mparam.nu))
            y = yterm * n**mparam.nu
            v0_temp = y - wage * n + (1 - mparam.delta) * k_cross[:, :, t-1]
            k_cross[:, :, t:t+1] = init_policy_fn(policy_fn_true, k_cross[:, :, t-1:t], k_mean[:, t-1:t], ashock[:, t-1:t]).detach().cpu().clamp(min=0.1).numpy()
            k_mean[:, t] = k_cross[:, :, t].mean(axis=1)
            xi_ex = price[:, t-1:t]*wage*(xi[:,:,t-1]**2) / (2*mparam.B)
            v0[:,:,t-1] = np.where((1-mparam.delta)*k_cross[:,:,t-1]==k_cross[:,:,t], (y-wage*n)* price[:, t-1:t] - xi_ex, v0_temp*price[:, t-1:t]-xi_ex-price[:, t-1:t]*k_cross[:,:,t])
            
            
    simul_data = {
        "price": price,
        "v0": v0,
        "k_cross": k_cross,
        "ashock": ashock,
        "xi": xi
    } 
    
    return simul_data

def simul_k_init_update(n_sample, T, mparam, policy_true, policy_type, price_fn, state_init=None, shocks=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if shocks is not None:
        ashock = shocks
        assert n_sample == ashock.shape[0], "n_sample is inconsistent with given shocks."
        assert T == ashock.shape[1], "T is inconsistent with given shocks."
        if state_init:
            assert np.array_equal(ashock[..., 0:1], state_init["ashock"]) and \
                "Shock inputs are inconsistent with state_init"
    else:
        ashock, xi = simul_shocks(n_sample, T, mparam, state_init)
    
    n_agt = mparam.n_agt
    k_cross = np.zeros((n_sample, n_agt, T))
    k_mean = np.zeros((n_sample, T))
    price = np.zeros((n_sample, T))
    v0 = np.zeros((n_sample, n_agt, T-1))
    if state_init:
        assert n_sample == state_init["k_cross"].shape[0], "n_sample is inconsistent with state_init."
        k_cross[:, :, 0] = state_init["k_cross"]
    else:
        k_cross[:, :, 0:1] = mparam.k_ss
    k_mean[:, 0] = k_cross[:, :, 0].mean(axis=1)

    if policy_type == "nn_share":
        for t in range(1, T):
            price_data = torch.cat((torch.tensor(k_cross[:, :, t-1], dtype=TORCH_DTYPE), torch.tensor(ashock[:, t-1:t], dtype=TORCH_DTYPE)), dim=1)
            price[:, t-1] = price_fn(price_data.to(device)).detach().cpu().clamp(min=0.01).numpy().squeeze(-1)
            wage = mparam.eta / price[:, t-1:t]#384,1
            yterm = ashock[:, t-1:t] * k_cross[:, :, t-1]**mparam.theta#384,50
            n = (mparam.nu * yterm / wage)**(1 / (1 - mparam.nu))
            y = yterm * n**mparam.nu
            v0_temp = y - wage * n + (1 - mparam.delta) * k_cross[:, :, t-1]
            v0[:,:,t-1] = v0_temp * price[:, t-1:t]
            k_cross[:, :, t:t+1] = init_policy_fn(policy, k_cross[:, :, t-1:t], k_mean[:, t-1:t], ashock[:, t-1:t]).detach().cpu().clamp(min=0.1).numpy()
            k_mean[:, t] = k_cross[:, :, t].mean(axis=1)
    
    simul_data = {
        "price": price,
        "v0": v0,
        "k_cross": k_cross,
        "ashock": ashock
    } 
    
    return simul_data


def init_policy_fn(init_policy, k_cross, k_mean, ashock):
    # NumPyで処理する
    k_mean_tmp = np.repeat(k_mean, 50, axis=1)[:, :, np.newaxis]
    ashock_tmp = np.repeat(ashock, 50, axis=1)[:, :, np.newaxis]
    basic_s = np.concatenate([k_cross, k_mean_tmp, ashock_tmp], axis=2)
    
    # GPUでNNの計算を実行
    basic_s_torch = torch.tensor(basic_s, device="cuda", dtype=TORCH_DTYPE)
    output_torch = init_policy(basic_s_torch)
    
    # 結果をCPU上でNumPyに変換
    output = output_torch

    return output

def init_policy_fn_tf(init_policy, k_cross, k_mean, ashock):
    # PyTorchで処理する
    k_mean_tmp = k_mean.repeat(1, 50).unsqueeze(2)  # axis=1をPyTorchで再現
    ashock_tmp = ashock.repeat(1, 50).unsqueeze(2)  # axis=1をPyTorchで再現
    basic_s = torch.cat([k_cross, k_mean_tmp,ashock_tmp], dim=2)  # NumPyのconcatenateをtorch.catで再現
    
    # GPUでNNの計算を実行
    basic_s_torch = basic_s.to("cuda")  # GPUに移動
    output_torch = init_policy(basic_s_torch)
    
    # 結果をそのまま返す（必要ならCPUに移動してNumPyに変換）
    output = output_torch  # 必要に応じて .to('cpu') を付けてCPUに戻す

    return output

def create_stats_init(n_sample, T, mparam, policy, policy_type, price_fn, state_init=None, shocks=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if shocks is not None:
        ashock = shocks
        assert n_sample == ashock.shape[0], "n_sample is inconsistent with given shocks."
        assert T == ashock.shape[1], "T is inconsistent with given shocks."
        if state_init:
            assert np.array_equal(ashock[..., 0:1], state_init["ashock"]) and \
                "Shock inputs are inconsistent with state_init"
    else:
        ashock, xi = simul_shocks(n_sample, T, mparam, state_init)
    
    n_agt = mparam.n_agt
    k_cross = np.zeros((n_sample, n_agt, T))
    k_mean = np.zeros((n_sample, T))
    basic_s = np.zeros(shape=[0, n_agt, 4])  # Updated to accommodate the additional element
    if state_init:
        assert n_sample == state_init["k_cross"].shape[0], "n_sample is inconsistent with state_init."
        k_cross[:, :, 0] = state_init["k_cross"]
    else:
        k_cross[:, :, 0:1] = mparam.k_ss
    k_mean[:, 0] = k_cross[:, :, 0].mean(axis=1)
    if policy_type == "nn_share":
        for t in range(1, T):
            k_cross[:, :, t:t+1] = init_policy_fn(
                policy, k_cross[:, :, t-1:t], k_mean[:, t-1:t], ashock[:, t-1:t]
            ).detach().cpu().numpy()
            a_tmp = np.repeat(ashock[:, None, t-1:t], n_agt, axis=1)
            xi_tmp = xi[:,:,t-1:t]
            k_tmp = k_cross[:, :, t-1:t]
            k_mean_tmp = np.mean(k_tmp, axis=1, keepdims=True)
            k_mean_tmp_expanded = np.tile(k_mean_tmp, (1, n_agt, 1))
            basic_s_tmp = np.concatenate([k_tmp, k_mean_tmp_expanded, a_tmp, xi_tmp], axis=-1)
            basic_s = np.concatenate([basic_s, basic_s_tmp], axis=0)
    return basic_s




class PolicyDataset(Dataset):
    def __init__(self, data):
        """
        data: numpy配列
        入力 (X): grid_k, grid_K, ashock
        ターゲット (y): grid_k
        """
        self.X = torch.tensor(data, dtype=TORCH_DTYPE)        # grid_k, grid_K, ashock
        self.y = torch.tensor(data[:, 0], dtype=TORCH_DTYPE) # grid_k

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def initial_policy(model, mparam, num_epochs=200, batch_size=50):
    # ショックをシミュレーション
    ashock, xi = simul_shocks(n_sample=1, T=500, mparam=mparam, state_init=None)

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
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    model.to(device)
    # 学習ループの実装
    t=0
    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in dataloader:
            t+=1
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # フォワードパス
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)

            # バックワードパスとオプティマイザーのステップ
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if t % 10 == 0:
                print(f'Step [{t}], Loss: {loss.item()}')


def plot_policy(model, mparam, grid_K_fixed=3.0, ashock_fixed=1.0):
    """
    モデルの出力を grid_k に対してプロットする関数

    Parameters:
    - model: 学習済みのPyTorchモデル
    - mparam: モデルパラメータオブジェクト（ZやPiなどを含む）
    - grid_K_fixed: 固定する grid_K の値
    - ashock_fixed: 固定する ashock の値
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    # grid_k の範囲を設定
    grid_k = np.linspace(0.1, 3.0, 100)

    # grid_K と ashock を固定してデータを作成
    grid_K = np.full_like(grid_k, grid_K_fixed)
    ashock = np.full_like(grid_k, ashock_fixed)

    # 入力データを結合
    data = np.column_stack([grid_k, grid_K, ashock])

    # カスタムデータセットとデータローダーの作成
    dataset = PolicyDataset(data)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=False)

    outputs = []

    with torch.no_grad():
        for batch_X, _ in dataloader:
            batch_X = batch_X.to(device)
            out = model(batch_X).squeeze().cpu().numpy()
            outputs.append(out)

    # 予測結果を結合
    outputs = np.concatenate(outputs)

    # プロットの作成
    plt.figure(figsize=(10, 6))
    plt.plot(grid_k, outputs, label='Model Output')
    plt.xlabel('grid_k')
    plt.ylabel('Output')
    plt.title('Policy Output vs grid_k')
    plt.legend()
    plt.grid(True)
    plt.show()


def seed_everything(seed=42):
    torch.manual_seed(seed)  # PyTorchのCPU乱数シードを固定
    torch.cuda.manual_seed_all(seed)  # PyTorchのGPU乱数シードを固定
    np.random.seed(seed)  # NumPyの乱数シードを固定
    random.seed(seed)  # Pythonの乱数シードを固定

    # CuDNNの再現性を確保
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

