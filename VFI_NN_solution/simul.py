import os
import numpy as np
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from datetime import datetime
import value_iter as vi
import pred_train as pred
from param import params
import random
import json


DTYPE = "float32"
if DTYPE == "float64":
    NP_DTYPE = np.float64
    TORCH_DTYPE = torch.float64  
elif DTYPE == "float32":
    NP_DTYPE = np.float32
    TORCH_DTYPE = torch.float32   
else:
    raise ValueError("Unknown dtype.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def simulation(params, nn, T, init=None, init_dist=None, last_dist=True):
    start_time = time.time()  # シミュレーション開始時刻を記録

    vi.move_models_to_device(nn, "cpu")
    i_size = params.ishock.size(0)
    G = params.grid_size

    dist_now = torch.full((G, i_size), 1.0 / (i_size * G), dtype=params.pi_i.dtype)
    if init_dist is not None:
        dist_now = nn.init_dist
    dist_now_k = torch.sum(dist_now, dim=1)

    k_now = params.k_grid  # (grid_size, nz)
    k_now_k = k_now[:, 0]  # Assuming aggregate shock is scalar for now
    ashock = generate_ashock(1, T, params.ashock, params.pi_a).squeeze(0)  # (T,)

    # Initialize histories
    dist_history = []
    k_history = []
    dist_k_history = []
    grid_k_history = []
    ashock_history = []
    mean_k_history = []
    price_diff_history = []
    price_history = []
    # Initialize lists to store statistics each period
    i_over_k_level_history = []
    i_over_k_std_history = []
    inaction_history = []
    positive_spike_history = []
    negative_spike_history = []
    positive_inv_history = []
    negative_inv_history = []

    grid = params.k_grid_tmp
    grid_1d = params.k_grid_tmp

    # tqdm を使用して進捗バーを表示
    with tqdm(total=T, desc="Simulation Progress", unit="period") as pbar:
        for t in range(T):
            basic_s = {
                "ashock": ashock[t],         
                "ishock": params.ishock.unsqueeze(0).expand(G, -1),  # Idiosyncratic shocks (G, I)
                "grid": params.k_grid,  #G,I
                "dist": dist_now,  #G,I
                "grid_k": params.k_grid_tmp,         
                "dist_k": dist_now_k,      
            }

            pnew, alpha, threshold, k_prime_adj = bisectp(nn, params, basic_s, max_expansions=5, max_bisect_iters=50, init=init)

            k_prime_non_adj = (1 - params.delta) * params.k_grid

            idx_adj_lower, idx_adj_upper, weight_adj = vi.map_to_grid(k_prime_adj, params.k_grid)
            idx_non_adj_lower, idx_non_adj_upper, weight_non_adj = vi.map_to_grid(k_prime_non_adj, params.k_grid)

            # Initialize new distribution
            dist_new = torch.zeros_like(dist_now)

            vi.update_distribution(
                dist_new, 
                dist_now, 
                alpha, 
                idx_adj_lower, 
                idx_adj_upper, 
                weight_adj, 
                params.pi_i, 
                adjusting=True
            )

            vi.update_distribution(
                dist_new, 
                dist_now, 
                alpha, 
                idx_non_adj_lower, 
                idx_non_adj_upper, 
                weight_non_adj, 
                params.pi_i, 
                adjusting=False
            )

            ##### Obtain statistics #####
            i_over_k = (k_prime_adj - k_prime_non_adj) / params.k_grid
            i_over_k_alpha = i_over_k * dist_now * alpha
            i_over_k_level = torch.sum(i_over_k_alpha, dim=(0, 1)).item()
            i_over_k_std = torch.sqrt(torch.sum(i_over_k**2 * dist_now * alpha, dim=(0, 1))).item()
            inaction = torch.sum(dist_now * (1 - alpha)).item()
            positive_spike = torch.where(i_over_k > 0.2, dist_now * alpha, torch.zeros_like(dist_now)).sum().item()
            negative_spike = torch.where(i_over_k < -0.2, dist_now * alpha, torch.zeros_like(dist_now)).sum().item()
            positive_inv = torch.where(i_over_k > 0, dist_now * alpha, torch.zeros_like(dist_now)).sum().item()
            negative_inv = torch.where(i_over_k < 0, dist_now * alpha, torch.zeros_like(dist_now)).sum().item()
            ##### Obtain statistics #####

            # Append statistics to their respective lists
            i_over_k_level_history.append(i_over_k_level)
            i_over_k_std_history.append(i_over_k_std)
            inaction_history.append(inaction)
            positive_spike_history.append(positive_spike)
            negative_spike_history.append(negative_spike)
            positive_inv_history.append(positive_inv)
            negative_inv_history.append(negative_inv)

            dist_sum = dist_new.sum()
            # Normalize distribution to prevent numerical errors
            dist_new /= dist_sum

            # Update aggregate capital distribution
            dist_new_k = dist_new.sum(dim=1)  # Sum over idiosyncratic shocks

            dist_history.append(dist_now.clone())
            dist_k_history.append(dist_now_k.clone())
            k_history.append(k_now.clone())
            grid_k_history.append(k_now_k.clone())
            ashock_history.append(ashock[t])  # Record scalar 'a'
            price_history.append(pnew)

            dist_now = dist_new
            dist_now_k = dist_new_k

            pbar.update(1)  # 進捗バーを更新

    ##### Calculate average statistics after period 500 #####
    start_period = 500#############Need to change to 500
    if T > start_period:
        i_over_k_level_mean = sum(i_over_k_level_history[start_period:]) / (T - start_period)
        i_over_k_std_mean = sum(i_over_k_std_history[start_period:]) / (T - start_period)
        inaction_mean = sum(inaction_history[start_period:]) / (T - start_period)
        positive_spike_mean = sum(positive_spike_history[start_period:]) / (T - start_period)
        negative_spike_mean = sum(negative_spike_history[start_period:]) / (T - start_period)
        positive_inv_mean = sum(positive_inv_history[start_period:]) / (T - start_period)
        negative_inv_mean = sum(negative_inv_history[start_period:]) / (T - start_period)

        # Compile average statistics into a dictionary
        mean_statistics = {
            "i_over_k_level_mean": i_over_k_level_mean,
            "i_over_k_std_mean": i_over_k_std_mean,
            "inaction_mean": inaction_mean,
            "positive_spike_mean": positive_spike_mean,
            "negative_spike_mean": negative_spike_mean,
            "positive_inv_mean": positive_inv_mean,
            "negative_inv_mean": negative_inv_mean,
        }

        # Directory structure setup
        results_dir = "results/simstats"
        current_datetime = datetime.now()
        date_str = current_datetime.strftime("%Y-%m-%d")
        time_str = current_datetime.strftime("%H_%M")

        # Path for the date-specific folder
        date_folder = os.path.join(results_dir, date_str)
        # Create the date folder if it does not exist
        os.makedirs(date_folder, exist_ok=True)

        # Generate the filename with current time
        filename = f"stats{time_str}.json"
        file_path = os.path.join(date_folder, filename)

        # Write the average statistics to the JSON file
        try:
            with open(file_path, "w") as f:
                json.dump(mean_statistics, f, indent=4)
            print(f"Average statistics have been saved to {file_path}.")
        except Exception as e:
            print(f"An error occurred while writing the file: {e}")

    vi.move_models_to_device(nn, "cuda")

    if last_dist:
        nn.init_dist = dist_now.clone()
        nn.init_dist_k = dist_now_k.clone()
    ##### Exclude average statistics from the return value #####
    end_time = time.time()  # シミュレーション終了時刻を記録
    elapsed_time = end_time - start_time  # 実行時間を計算
    print(f"Simulation completed in {elapsed_time:.2f} seconds.")

    return {
        "grid": k_history[100:],         # From the 100th period onwards
        "dist": dist_history[100:],      # From the 100th period onwards
        "dist_k": dist_k_history[100:],  # From the 100th period onwards
        "grid_k": grid_k_history[100:],  # From the 100th period onwards
        "ashock": ashock_history[100:],  # From the 100th period onwards
        "price": price_history[100:],    # From the 100th period onwards
        # Average statistics are excluded from the return value
    }

def golden_section_search_batch(
    train_data,
    price,
    nn,
    params,
    device,
    left_bound: float,
    right_bound: float,
    batch_size: int,
    max_iter: int = 20, 
    tol: float = 1e-5
):
    """
    golden_section_search_batchの高速化版。
    next_e0のうち、k以外は一度だけ計算し、
    ループ内では next_k 依存の部分だけ計算するようにする。
    """

    # next_e0 のうち k 以外をキャッシュし、k だけ差し替える関数を作る
    next_e0_partial = init_next_e0_sim(train_data, price, nn, params, device)

    # 黄金比
    phi = 0.618033988749895
    
    # 初期区間 [a, b]
    a = torch.full((batch_size,), left_bound, device=device, dtype=TORCH_DTYPE)
    b = torch.full((batch_size,), right_bound, device=device, dtype=TORCH_DTYPE)
    
    for _ in range(max_iter):
        dist_ = b - a
        c = b - phi * dist_
        d = a + phi * dist_

        fc = next_e0_partial(c)
        fd = next_e0_partial(d)

        mask = (fc > fd)
        b[mask] = d[mask]
        a[~mask] = c[~mask]

    x_star = 0.5 * (a + b)
    f_star = next_e0_partial(x_star)

    return x_star.expand(params.grid_size, -1), f_star.expand(params.grid_size, -1)

def bisectp(nn, params, data, max_expansions=5, max_bisect_iters=50, init=None):
    """
    Uses the bisection method to find the equilibrium price. If convergence is not achieved,
    the initial interval is expanded iteratively. Additionally, if the change in `diff` between
    consecutive iterations is less than or equal to 0.01, the interval is expanded.

    Args:
        nn: Object of the neural network class.
        params: Parameter object. Must include `critbp` for convergence criteria.
        data: Data dictionary. Must include 'grid', 'dist', 'ashock'.
        max_expansions (int): Maximum number of times to expand the initial interval.
        max_bisect_iters (int): Maximum number of iterations for the bisection method.
        init: Initial price guess. If None, computed from `price_fn`.

    Returns:
        p0: Converged price.
        alpha: Alpha value at convergence.
        threshold: Threshold value at convergence.

    Raises:
        ValueError: If the bisection method does not converge within the allowed expansions.
    """
    # Initialize the price based on the initial guess from the price function
    p_init = price_fn(data["grid"], data["dist"], data["ashock"], nn).squeeze(-1)
    if init is not None:
        p_init = torch.full_like(p_init, init)
    pL = p_init - 0.3  # Lower bound of the price interval
    pH = p_init + 0.3  # Upper bound of the price interval
    critbp = params.critbp  # Convergence criterion
    expansion_count = 0  # Counter for the number of expansions

    while expansion_count <= max_expansions:
        diff = float('inf')  # Initialize difference to infinity
        iter_count = 0  # Iteration counter for the bisection method
        prev_diff = None  # To store the previous difference

        # Bisection loop
        while diff > critbp and iter_count < max_bisect_iters:
            p0 = (pL + pH) / 2  # Midpoint of the current interval
            pnew, alpha, threshold, next_k = eq_price(nn, data, params, p0, device="cpu")  # Compute new price and distance
            B0 = p0 - pnew  # Difference between current price and new price

            if pnew < 0:
                pL = p0  # Adjust the lower bound if new price is negative
            else:
                if B0 < 0:
                    pL = p0  # Adjust the lower bound if B0 is negative
                else:
                    pH = p0  # Adjust the upper bound otherwise

            new_diff = abs(B0)  # Calculate the new difference

            # Check if the change in diff is small enough to trigger expansion
            #if prev_diff is not None:
                #if new_diff > 0.01 and abs(new_diff - prev_diff) <= 0.001:
                    #print(f"Change in diff ({abs(new_diff - prev_diff):.4f}) <= 0.01, triggering expansion.")
                    #break  # Exit the bisection loop to expand the interval

            prev_diff = new_diff  # Update previous difference
            diff = new_diff  # Update the current difference
            iter_count += 1  # Increment iteration counter

        if diff <= critbp:
            return p0, alpha, threshold, next_k  # Return the converged price and distance
        else:
            expansion_count += 1
            # Expand the initial interval if convergence was not achieved
            expansion_amount = expansion_count * 0.1
            pL = p_init - 0.3 - expansion_amount  # Expand lower bound
            pH = p_init + 0.3 + expansion_amount  # Expand upper bound
            #print(f"Expansion {expansion_count}: New interval [{pL.item():.4f}, {pH.item():.4f}]")

    # Raise an error if the maximum number of expansions is exceeded without convergence
    raise ValueError("Bisection method did not converge. Reached maximum number of expansions.")

def eq_price(nn, data, params, price, device="cpu"):
    i_size = params.ishock.size(0)
    max_cols = params.k_grid.size(0)
    ashock_2d = data["ashock"].view(1, 1).expand(max_cols, i_size)###What is this? This should come from data.
    ishock_2d = data["ishock"]
    price = price.view(-1, 1).expand(max_cols, i_size)
    wage = params.eta/price
    next_k, e0 = golden_section_search_batch(data, price, nn, params, "cpu", params.k_grid_min, params.k_grid_max, batch_size=params.nz)
    e1 = next_e1(data, price, nn, params, device)
    
    e0 = e0.view(params.grid_size, -1)
    e1 = e1.view(params.grid_size, -1)
    next_k = next_k.view(params.grid_size, -1)
    threshold = (e0 - e1) / params.eta
    xi = torch.min(torch.tensor(params.B, dtype=TORCH_DTYPE), torch.max(torch.tensor(0, dtype=TORCH_DTYPE), threshold))
    alpha = (xi / params.B).squeeze(-1)#G,I
    
    yterm = ashock_2d * ishock_2d  * data["grid"]**params.theta
    numerator = params.nu * yterm / (wage + 1e-6)
    numerator = torch.clamp(numerator, min=1e-6, max=1e8)  # 数値の範囲を制限
    nnow = torch.pow(numerator, 1 / (1 - params.nu))
    inow = alpha * (next_k - (1-params.delta) * data["grid"])
    ynow = ashock_2d*ishock_2d * data["grid"]**params.theta * nnow**params.nu
    Iagg = torch.sum(data["dist"] * inow)
    Yagg = torch.sum(data["dist"] * ynow)
    Cagg = Yagg - Iagg
    target = 1 / Cagg
    return target, alpha, threshold, next_k
                
def init_next_e0_sim(data, price, nn, params, device):
    """
    next_e0 のうち、呼び出すたびに変わらない部分を一度だけ計算してキャッシュし、
    k だけ変わる部分を内部関数で計算するようにする (シミュレーション用).

    Parameters
    ----------
    data : dict
        A dictionary containing 'grid', 'grid_k', 'dist', 'dist_k', 'ashock', etc.
    price : torch.Tensor
        Price tensor, shape assumed to be (1, i_size) so that price[0, :] => (i_size,)
    nn : network object
        Your neural network container (holding target_value, gm_model, etc.)
    params : object
        Parameter object with attributes like pi_a, pi_i, ashock, ishock, etc.
    device : torch.device
        The device on which tensors are placed (e.g., "cuda" or "cpu").

    Returns
    -------
    next_e0_partial : function
        A function that takes only k (shape: (dist_size,)) as input and returns e0.
    """

    with torch.no_grad():
        # G: number of grid points in 'grid'
        G = data["grid"].size(0)
        # i_size: number of idiosyncratic shock states
        i_size = params.ishock.size(0)

        # Compute next_gm only once (shape: (G, I))
        next_gm = dist_gm(data["grid_k"], data["dist_k"], data["ashock"], nn)

        # Reshape price if needed: original is (1, i_size) -> we take price[0, :] => (i_size,)
        price_ = price[0, :]  # (i_size,)

        # Determine ashock index from data["ashock"]
        ashock_idx = torch.where(params.ashock == data["ashock"])[0].item()

        # Construct probabilities for transitions
        # pi_a[ashock_idx]: shape (5,) for example -> repeat for i_size times -> (i_size, 5)
        ashock_exp = params.pi_a[ashock_idx].repeat(i_size, 1).unsqueeze(-1)  # => (i_size, 5, 1)
        ishock_exp = params.pi_i.unsqueeze(1)                                # => (i_size, 1, 5)
        # Multiply to get joint probabilities => (i_size, 5, 5) if we have 5 states each
        probabilities = ashock_exp * ishock_exp

        # Create normalized meshgrids for a_shock and i_shock
        a_mesh, i_mesh = torch.meshgrid(params.ashock, params.ishock, indexing='ij')
        a_mesh_norm = (a_mesh - params.ashock_min) / (params.ashock_max - params.ashock_min)
        i_mesh_norm = (i_mesh - params.ishock_min) / (params.ishock_max - params.ishock_min)

        # Flatten and expand to match the shape (i_size, a*i, 1)
        a_flat = a_mesh_norm.flatten().view(1, -1, 1).expand(i_size, -1, -1)
        i_flat = i_mesh_norm.flatten().view(1, -1, 1).expand(i_size, -1, -1)

        # Expand next_gm to match (i_size, a*i, 1)
        # next_gm: shape (G, I). The code does .view(-1,1,1) -> (G,1,1) => expand to (i_size, a*i, 1)
        next_gm_flat = next_gm.view(-1, 1, 1).expand(i_size, a_flat.size(1), -1)

    def next_e0_partial(k):
        """
        Internal function that takes k (shape: (dist_size,)) and computes e0 using
        cached expansions from init_next_e0_sim.
        """
        # Expand k to match the flattened mesh shape: (i_size, a*i, 1)
        k_expanded = k.view(-1, 1, 1).expand(-1, a_flat.size(1), -1)

        # Concatenate everything into data_e0 for the value function
        data_e0 = torch.cat([k_expanded, a_flat, i_flat, next_gm_flat], dim=2)  # => (i_size, a*i, 4)

        # Compute V0 => reshape => take expectation
        value0 = nn.target_value(data_e0).squeeze(-1)     # => shape (i_size*(a*i))
        value0 = value0.view(-1, len(params.ashock), len(params.ishock))  # => (i_size, nA, nI)

        expected_value0 = (value0 * probabilities).sum(dim=(1, 2))  # => shape (i_size,)

        # Finally e0 = -k * price_ + beta * E[V0]
        # Here k is broadcast with price_, both shape (i_size,) after some expansions
        e0 = -k * price_ + params.beta * expected_value0

        return e0.squeeze(-1)  # => shape (i_size,) or (dist_size,)

    return next_e0_partial



def next_e1(data, price, nn, params, device):
    G = data["grid"].size(0)
    i_size = params.ishock.size(0)
    next_gm = dist_gm(data["grid_k"], data["dist_k"], data["ashock"], nn)#G,I
    price = price.flatten()#dist_size
    ashock_idx = torch.where(params.ashock == data["ashock"])[0].item()
    ashock_exp = params.pi_a[ashock_idx].repeat(params.dist_size, 1).unsqueeze(-1)#dist_size, 5,1
    ishock_exp = params.pi_i.repeat(params.grid_size, 1).unsqueeze(1)#dist_size, 1, 5
    probabilities = ashock_exp * ishock_exp
    
    a_mesh, i_mesh = torch.meshgrid(params.ashock, params.ishock, indexing='ij')
    a_mesh_norm = (a_mesh - params.ashock_min) / (params.ashock_max - params.ashock_min)
    i_mesh_norm = (i_mesh - params.ishock_min) / (params.ishock_max - params.ishock_min)
    a_flat = a_mesh_norm.flatten().view(1, -1, 1).expand(params.dist_size, -1, -1) # shape: dist_size, i*a, 1
    i_flat = i_mesh_norm.flatten().view(1, -1, 1).expand(params.dist_size, -1, -1) # shape: dist_size, i*a, 1
    k_cross_flat = params.k_grid.flatten().view(-1, 1, 1).expand(-1, a_flat.size(1), -1)
    pre_k_flat = k_cross_flat * (1 - params.delta)
    next_gm_flat = next_gm.view(-1, 1, 1).expand(params.dist_size, a_flat.size(1), -1)#dist_size, i*a, 1
    
    data_e1 = torch.cat([pre_k_flat, a_flat, i_flat, next_gm_flat], dim=2)
    value1 = nn.target_value(data_e1).squeeze(-1)
    value1 = value1.view(-1, len(params.ashock), len(params.ishock))  # (batch_size, a, i)
    expected_value1 = (value1 * probabilities).sum(dim=(1, 2))  # (batch_size)
    e1 = -(1-params.delta) * params.k_grid.flatten() * price + params.beta * expected_value1
    return e1
    
    

def policy_fn_sim(ashock, ishock, grid_k, dist_k, price, nn):
    ashock_norm = (ashock - params.ashock_min) / (params.ashock_max - params.ashock_min)
    ishock_norm = (ishock - params.ishock_min) / (params.ishock_max - params.ishock_min)
    grid_norm = (grid_k - params.k_grid_min) / (params.k_grid_max - params.k_grid_min)
    gm_tmp = nn.gm_model_policy(grid_norm.unsqueeze(-1))#G,1
    gm = torch.sum(gm_tmp * dist_k.unsqueeze(-1), dim=-2).unsqueeze(1).expand(grid_k.size(0), ishock.size(1))#G, I
    price_norm = (price - params.price_min) / (params.price_max - params.price_min)
    state = torch.stack([ashock_norm, ishock_norm, gm, price_norm], dim=-1)
    output = nn.policy(state) * 8
    next_k = output
    return next_k#G,I,1

def price_fn(grid, dist, ashock, nn, mean=None):
    if mean is not None:
        mean = torch.sum(grid * dist, dim=-1)
        state = torch.stack([ashock, mean], dim=1)
    else:
        grid_norm = (grid - params.k_grid_min) / (params.k_grid_max - params.k_grid_min)
        ashock_norm = (ashock - params.ashock_min) / (params.ashock_max - params.ashock_min)
        gm_tmp = nn.gm_model_price(grid_norm).squeeze(-1)#grid_size, i_size
        gm_price = torch.sum(gm_tmp * dist, dim=-2)#i_size
        state = torch.cat([ashock_norm.unsqueeze(0), gm_price]).unsqueeze(0)#1,isize
        price = nn.price_model(state)#batch, 1
    return price

def dist_gm(grid, dist, ashock, nn):
    grid_norm = (grid - params.k_grid_min) / (params.k_grid_max - params.k_grid_min)
    ashock_norm = (ashock - params.ashock_min) / (params.ashock_max - params.ashock_min)
    gm_tmp = nn.target_gm_model(grid_norm.unsqueeze(-1))
    gm = torch.sum(gm_tmp * dist.unsqueeze(-1), dim=-2)
    state = torch.cat([ashock_norm.view(1), gm]).unsqueeze(0)
    next_gm = nn.next_gm_model(state)
    return next_gm

def generate_ashock(num_sample, T, shock, Pi):
    """
    Generates T ashock values using PyTorch.
    
    Parameters:
    - num_sample (int): Number of samples
    - T (int): Number of values to generate for each sample
    - shock (torch.Tensor): Ashock values corresponding to states (shape: (nz,))
    - Pi (torch.Tensor): Transition probability matrix (shape: (nz, nz))
    
    Returns:
    - torch.Tensor: Generated ashock values (shape: (num_sample, T))
    """
    # Ensure Pi does not contain any zero rows
    row_sums = Pi.sum(dim=1, keepdim=True)
    if torch.any(row_sums == 0):
        raise ValueError("There are rows in the Pi matrix where the sum of each row is zero.")
    
    # Normalize Pi
    Pi_normalized = Pi / row_sums
    
    # Re-normalize to ensure that the sum of probabilities is exactly 1 (fix numerical errors)
    Pi_normalized = Pi_normalized / Pi_normalized.sum(dim=1, keepdim=True)
    
    # Number of states
    nz = shock.size(0)
    
    # Randomly select initial states (uniform distribution)
    states = torch.randint(low=0, high=nz, size=(num_sample, T), device=Pi.device)
    
    # Set the initial state for each sample randomly
    states[:, 0] = torch.randint(low=0, high=nz, size=(num_sample,), device=Pi.device)
    
    for t in range(1, T):
        # Previous state
        prev_states = states[:, t - 1]  # Shape: (num_sample,)
        
        # Get the probability distribution corresponding to the previous state
        probs = Pi_normalized[prev_states]  # Shape: (num_sample, nz)
        
        # Sample the next state for each sample
        next_states = torch.multinomial(probs, num_samples=1).squeeze(1)  # Shape: (num_sample,)
        
        # Set the next state at the current time step
        states[:, t] = next_states
    
    # Map state indices to corresponding ashock values
    ashock_values = shock[states]  # Shape: (num_sample, T)
    
    return ashock_values
