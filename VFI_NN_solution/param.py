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
        self.B = 0.02
        self.ashock = torch.tensor([0.9328, 0.9658, 1.0000, 1.0354, 1.0720])
        self.ashock_gpu = self.ashock.to(self.device)
        self.shock_max = self.ashock.max().item()
        self.shock_min = self.ashock.min().item()
        self.nz = 5
        pi_a = torch.tensor([
            [0.8537, 0.1377, 0.0083, 0.0002, 0.0000],
            [0.0344, 0.8579, 0.1035, 0.0042, 0.0001],
            [0.0014, 0.0690, 0.8593, 0.0690, 0.0014],
            [0.0001, 0.0042, 0.1035, 0.8579, 0.0344],
            [0.0000, 0.0002, 0.0083, 0.1377, 0.8537]
        ])
        self.pi_a = pi_a / pi_a.sum(dim=1, keepdim=True)
        self.pi_a_gpu = self.pi_a.to(self.device)
        
        self.ishock = torch.tensor([0.9328, 0.9658, 1.0000, 1.0354, 1.0720])
        self.ishock_gpu = self.ishock.to(self.device)

        # pi_i の定義
        pi_i = torch.tensor([
            [0.8537, 0.1377, 0.0083, 0.0002, 0.0000],
            [0.0344, 0.8579, 0.1035, 0.0042, 0.0001],
            [0.0014, 0.0690, 0.8593, 0.0690, 0.0014],
            [0.0001, 0.0042, 0.1035, 0.8579, 0.0344],
            [0.0000, 0.0002, 0.0083, 0.1377, 0.8537]
        ])
        self.pi_i = pi_i / pi_i.sum(dim=1, keepdim=True)
        self.pi_i_gpu = self.pi_i.to(self.device)
        self.grid_size = 50
        # 0.1から50までのlogspaceで50個のグリッドを生成
        start = torch.log10(torch.tensor(0.1))
        end = torch.log10(torch.tensor(8))
        self.k_grid_tmp = torch.logspace(start, end, steps=self.grid_size)
        #self.k_grid_tmp = torch.linspace(0.1, 4, self.grid_size)
        self.k_grid_max = self.k_grid_tmp.max().item()
        self.k_grid_min = self.k_grid_tmp.min().item()
        
        self.k_grid = self.k_grid_tmp.view(-1, 1).repeat(1, self.nz)
        self.k_grid_1d_gpu = self.k_grid_tmp.to(self.device)
        self.k_grid_gpu = self.k_grid.to(self.device)
        self.K_grid_np = np.linspace(0.1, 3, 10)

        
        ykSS = (self.gamma - self.beta * (1 - self.delta)) / self.beta / self.theta
        ckSS = ykSS + (1 - self.gamma - self.delta)
        ycSS = ykSS / ckSS
        nSS = self.nu / self.eta * ycSS
        self.kSS = (ykSS * nSS ** (-self.nu)) ** (1 / (self.theta - 1))
        
        
        
        self.critbp = 1e-4

params = KTParam()


