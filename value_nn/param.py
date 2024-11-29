import numpy as np
import torch
class KTParam():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.beta = 0.9540
        self.theta = 0.256
        self.nu = 0.640
        self.delta = 0.069
        self.gamma = 1.0160
        self.eta = 2.40
        self.B = 0.5
        self.ashock = torch.tensor([0.9328, 0.9658, 1.0000, 1.0354, 1.0720])
        self.ashock_gpu = self.ashock.to(self.device)
        self.nz = 5
        self.pi_a = torch.tensor([
            [0.8537, 0.1377, 0.0083, 0.0002, 0.0000],
            [0.0344, 0.8579, 0.1035, 0.0042, 0.0001],
            [0.0014, 0.0690, 0.8593, 0.0690, 0.0014],
            [0.0001, 0.0042, 0.1035, 0.8579, 0.0344],
            [0.0000, 0.0002, 0.0083, 0.1377, 0.8537]
        ])
        self.pi_a_gpu = self.pi_a.to(self.device)
        
        self.ishock = torch.tensor([0.9328, 0.9658, 1.0000, 1.0354, 1.0720])
        self.ishock_gpu = self.ishock.to(self.device)

        # pi_i の定義
        self.pi_i = torch.tensor([
            [0.8537, 0.1377, 0.0083, 0.0002, 0.0000],
            [0.0344, 0.8579, 0.1035, 0.0042, 0.0001],
            [0.0014, 0.0690, 0.8593, 0.0690, 0.0014],
            [0.0001, 0.0042, 0.1035, 0.8579, 0.0344],
            [0.0000, 0.0002, 0.0083, 0.1377, 0.8537]
        ])
        self.pi_i_gpu = self.pi_i.to(self.device)
        # k_grid の定義
        linear_grid = torch.linspace(start=0.1, end=3, steps=30)
        log_grid = torch.logspace(start=torch.log10(torch.tensor(3.0)), end=1, steps=20)
        self.k_grid  = torch.cat((linear_grid, log_grid[1:]))
        self.k_grid_np = np.linspace(0.1, 3, 30)
        self.K_grid_np = np.linspace(1.0, 3, 10)
        ykSS = (self.gamma - self.beta * (1 - self.delta)) / self.beta / self.theta
        ckSS = ykSS + (1 - self.gamma - self.delta)
        ycSS = ykSS / ckSS
        nSS = self.nu / self.eta * ycSS
        self.kSS = (ykSS * nSS ** (-self.nu)) ** (1 / (self.theta - 1))
        self.grid_size = 50




