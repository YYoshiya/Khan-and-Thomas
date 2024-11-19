import numpy as np
import torch
class KTParam():
    def __init__(self):
        self.beta = 0.9540
        self.theta = 0.3250
        self.nu = 0.580
        self.delta = 0.060
        self.gamma = 1.0160
        self.eta = 3.6142
        self.B = 0.002
        self.ashock = torch.tensor([0.9328, 0.9658, 1.0000, 1.0354, 1.0720])
        self.nz = 5
        self.pi_a = torch.tensor([
            [0.8537, 0.1377, 0.0083, 0.0002, 0.0000],
            [0.0344, 0.8579, 0.1035, 0.0042, 0.0001],
            [0.0014, 0.0690, 0.8593, 0.0690, 0.0014],
            [0.0001, 0.0042, 0.1035, 0.8579, 0.0344],
            [0.0000, 0.0002, 0.0083, 0.1377, 0.8537]
        ])
        
        self.ishock = torch.tensor([0.9328, 0.9658, 1.0000, 1.0354, 1.0720])

        # pi_i の定義
        self.pi_i = torch.tensor([
            [0.8537, 0.1377, 0.0083, 0.0002, 0.0000],
            [0.0344, 0.8579, 0.1035, 0.0042, 0.0001],
            [0.0014, 0.0690, 0.8593, 0.0690, 0.0014],
            [0.0001, 0.0042, 0.1035, 0.8579, 0.0344],
            [0.0000, 0.0002, 0.0083, 0.1377, 0.8537]
        ])

        # k_grid の定義
        self.k_grid = torch.linspace(0.1, 10, steps=30)
        ykSS = (self.gamma - self.beta * (1 - self.delta)) / self.beta / self.theta
        ckSS = ykSS + (1 - self.gamma - self.delta)
        ycSS = ykSS / ckSS
        nSS = self.nu / self.eta * ycSS
        self.kSS = (ykSS * nSS ** (-self.nu)) ** (1 / (self.theta - 1))




