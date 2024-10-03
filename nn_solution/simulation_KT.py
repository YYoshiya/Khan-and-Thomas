import numpy as np
from scipy.interpolate import RectBivariateSpline
import torch

def simul_shocks(n_sample, T, mparam, state_init=None):
    n_agt = mparam.n_agt
    ashock = np.zeros([n_sample, T])
    if state_init:
        # convert productivity to 0/1 variable
        ashock[:, 0:1] = ((state_init["ashock"] - 1) / mparam.delta_a + 1) / 2
    else:
        ashock[:, 0] = np.random.binomial(1, 0.5, n_sample)  # stationary distribution of Z is (0.5, 0.5)
    
    for t in range(1, T):
        if_keep = np.random.binomial(1, 0.875, n_sample)  # prob for Z to stay the same is 0.875
        ashock[:, t] = if_keep * ashock[:, t - 1] + (1 - if_keep) * (1 - ashock[:, t - 1])
    
    ashock = (ashock * 2 - 1) * mparam.delta_a + 1  # convert 0/1 variable to productivity
    
    return ashock

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