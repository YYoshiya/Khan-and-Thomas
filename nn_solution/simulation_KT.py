import numpy as np
from scipy.interpolate import RectBivariateSpline
import torch

def simul_shocks(n_sample, T, Z, Pi, state_init=None):
    nz = len(Z)
    ashock = np.zeros([n_sample, T], dtype=int)  # ショックインデックスの格納用
    
    if state_init is not None:
        ashock[:, 0] = state_init  # 初期状態を設定
    else:
        # 初期状態をランダムに決定（均等分布）
        ashock[:, 0] = np.random.choice(nz, size=n_sample)
    
    for t in range(1, T):
        for i in range(n_sample):
            current_state = ashock[i, t - 1]
            # 確率遷移行列 Pi に従って次の状態を決定
            ashock[i, t] = np.random.choice(nz, p=Pi[current_state])
    
    # ショックインデックスから実際のショック値に変換
    ashock_values = Z[ashock]
    
    return ashock_values

def simul_k(n_sample, T, mparam, policy, policy_type, price_fn, state_init=None, shocks=None): 
    if shocks:
        ashock = shocks
    
        assert n_sample == ashock.shape[0], "n_sample is inconsistent with given shocks."
        assert T == ashock.shape[1], "T is inconsistent with given shocks."
        if state_init:
            assert np.array_equal(ashock[..., 0:1], state_init["ashock"]) and \
                "Shock inputs are inconsistent with state_init"
    else:
        ashock = simul_shocks(n_sample, T, mparam, state_init)
    
    k_cross = np.zeros([n_sample, n_agt, T])
    if state_init:
        assert n_sample == state_init["k_cross"].shape[0], "n_sample is inconsistent with state_init."
        k_cross[:, :, 0] = state_init["k_cross"]
    else:
        k_cross[:, :, 0] = mparam.k_ss
    
    if policy_type == "nn":
        for t in range(1, T):
            price = price_fn(k_cross[:, :, t-1])# 384*1
            wage = mparam.eta / price # 384*1
            yterm = ashock[:, t-1] * k_cross[:, :, t-1]**mparam.theta # 384*50
            n = (mparam.nu * yterm / wage)**(1 / (1 - mparam.nu))
            y = yterm * n**mparam.nu
            v0_temp = y - wage * n + (1 - mparam.delta) * k_cross[:, :, t-1]
            v0 = v0_temp * price # 384*50
            k_cross[:, :, t] = policy(k_cross[:, :, t - 1], ashock[:, t - 1])# 384*50
            
    
    simul_data = {"price": price, "wage": wage, "v0": v0, "k_cross": k_cross, "ashock": ashock}
            # 384*T, 384*T, 384*50*T, 384*50*T, 384*T
    return simul_data
def price_train(n_sample, T, mparam, policy, policy_type, price_fn, state_init=None, shocks=None):
    loss = price_loss(n_sample, T, mparam, policy, policy_type, price_fn, state_init, shocks)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    


# value, policyが学習されないようにする必要あり。
def price_loss(n_sample, T, mparam, policy, policy_type, price_fn, state_init=None, shocks=None):
    if shocks:
        ashock = shocks
    
        assert n_sample == ashock.shape[0], "n_sample is inconsistent with given shocks."
        assert T == ashock.shape[1], "T is inconsistent with given shocks."
        if state_init:
            assert np.array_equal(ashock[..., 0:1], state_init["ashock"]) and \
                "Shock inputs are inconsistent with state_init"
    else:
        ashock = simul_shocks(n_sample, T, mparam, state_init)
    
    k_cross = np.zeros([n_sample, n_agt, T])
    if state_init:
        assert n_sample == state_init["k_cross"].shape[0], "n_sample is inconsistent with state_init."
        k_cross[:, :, 0] = state_init["k_cross"]
    else:
        k_cross[:, :, 0] = mparam.k_ss
    
    if policy_type == "nn":
        for t in range(1, T):
            price = price_fn(k_cross[:, :, t-1])# 384*1
            wage = mparam.eta / price # 384*1
            yterm = ashock[:, t-1] * k_cross[:, :, t-1]**mparam.theta # 384*50
            n = (mparam.nu * yterm / wage)**(1 / (1 - mparam.nu))
            y = yterm * n**mparam.nu
            k_cross_pre = k_cross[:, :, t-1]
            k_cross[:, :, t] = policy(k_cross[:, :, t - 1], ashock[:, t - 1])# 384*50
            a_tmp = torch.repeat_interleave(ashock[:, t:t+1], 50, dim=1)#samplerで作成したbatch_size, self.t_unrollのashockを使っている。
            a_tmp = torch.unsqueeze(a_tmp, 2)
            basic_s_tmp = torch.cat([torch.unsqueeze(k_cross[:,:,t], axis=-1), a_tmp], axis=-1)
            full_state_dict = {
                "basic_s": basic_s_tmp,
                "agt_s": self.init_ds.normalize_data(torch.unsqueeze(k_cross[:,:,t], axis=-1), key="agt_s", withtf=True)
            }
            e0 = -mparam.GAMY * price * k_cross + mparam.BETA * value(full_state_dict)
            basic_s_tmp_pre = torch.cat([torch.unsqueeze(k_cross_pre, axis=-1), a_tmp], axis=-1)
            full_state_dict_pre = {
                "basic_s": basic_s_tmp_pre,
                "agt_s": self.init_ds.normalize_data(torch.unsqueeze(k_cross_pre, axis=-1), key="agt_s", withtf=True)
            }
            e1 = mparam.p * (1 - mparam.delta) * k_cross_pre + mparam.BETA * value(full_state_dict_pre)
            xitemp = (e0 - e1)/(price * wage)
            xi = min(B, max(0, xitemp))
            alpha = xi / B
            inow = alpha * (mparam.GAMY * k_cross - (1 - mparam.delta) * k_cross_pre)
            ynow = ashock[:, t-1] * k_cross_pre**mparam.theta * n**mparam.nu
            nnow = n + xi**2/(2*B)
        
        Inow = torch.sum(inow, axis=1)
        Ynow = torch.sum(ynow, axis=1)
        Cnow = Ynow - Inow
        price1 = 1 / Cnow #n_sample, Tになってて欲しい。
        loss = torch.mean((price - price1)**2)

    return  loss