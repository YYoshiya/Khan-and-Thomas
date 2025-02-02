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
        self.B = 0.25
        self.ashock = torch.tensor([0.9467, 0.9730, 1.0000, 1.0277, 1.0562])
        self.ashock_gpu = self.ashock.to(self.device)
        self.ashock_max = self.ashock.max().item()
        self.ashock_min = self.ashock.min().item()
        self.nz = 5
        pi_a = data = torch.tensor([
    [0.66487382950848084, 0.32644680509665858, 8.6719853447329021E-3, 7.3798866820906639E-6, 1.6344559039538353E-10],
    [0.10528259034646785, 0.65313000929388065, 0.23761520120544521, 3.9701628500560249E-3, 2.0363041502990953E-6],
    [1.6957510522399625E-3, 0.16268559885118380, 0.67123730019315242, 0.16268559885118383, 1.6957510522399577E-3],
    [2.0363041503018918E-6, 3.9701628500560240E-3, 0.23761520120544521, 0.65313000929388065, 0.10528259034646781],
    [1.6344557373790284E-10, 7.3798866821097857E-6, 8.6719853447328483E-3, 0.32644680509665858, 0.66487382950848084]
])
        self.pi_a = pi_a / pi_a.sum(dim=1, keepdim=True)
        self.pi_a_gpu = self.pi_a.to(self.device)
        
        self.ishock = torch.tensor([0.9176, 0.9579, 1.0000, 1.0439, 1.0897])
        self.ishock_gpu = self.ishock.to(self.device)
        self.ishock_max = self.ishock.max().item()
        self.ishock_min = self.ishock.min().item()

        # pi_i の定義
        pi_i = torch.tensor([
    [0.66487382950848084, 0.32644680509665858, 8.6719853447329021E-3, 7.3798866820906639E-6, 1.6344559039538353E-10],
    [0.10528259034646785, 0.65313000929388065, 0.23761520120544521, 3.9701628500560249E-3, 2.0363041502990953E-6],
    [1.6957510522399625E-3, 0.16268559885118364, 0.67123730019315286, 0.16268559885118361, 1.6957510522399577E-3],
    [2.0363041503018841E-6, 3.9701628500560162E-3, 0.23761520120544521, 0.65313000929388065, 0.10528259034646781],
    [1.6344557373790165E-10, 7.3798866821097857E-6, 8.6719853447328483E-3, 0.32644680509665858, 0.66487382950848084]
])
        self.pi_i = pi_i / pi_i.sum(dim=1, keepdim=True)
        self.pi_i_gpu = self.pi_i.to(self.device)
        self.grid_size = 50
        # 0.1から50までのlogspaceで50個のグリッドを生成
        start = torch.log10(torch.tensor(0.1))
        end = torch.log10(torch.tensor(8))
        self.k_grid_tmp = torch.logspace(start, end, steps=self.grid_size)
        self.k_grid_tmp_lin = torch.linspace(0.1, 8, 150)
        self.k_grid_max = self.k_grid_tmp.max().item()
        self.k_grid_min = self.k_grid_tmp.min().item()
        
        self.k_grid = self.k_grid_tmp.view(-1, 1).repeat(1, self.nz)
        self.k_grid_1d_gpu = self.k_grid_tmp.to(self.device)
        self.k_grid_gpu = self.k_grid.to(self.device)
        self.K_grid_np = np.linspace(0.1, 3, 10)

        self.price_min = 1.5
        self.price_max = 3
        
        ykSS = (self.gamma - self.beta * (1 - self.delta)) / self.beta / self.theta
        ckSS = ykSS + (1 - self.gamma - self.delta)
        ycSS = ykSS / ckSS
        nSS = self.nu / self.eta * ycSS
        self.kSS = (ykSS * nSS ** (-self.nu)) ** (1 / (self.theta - 1))
        
        self.price_size = 3
        
        
        self.critbp = 1e-3

params = KTParam()


