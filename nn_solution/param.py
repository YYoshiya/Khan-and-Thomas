import numpy as np

class KTParam():
    def __init__(self, n_agt, beta, mats_path):
        self.mats_path = mats_path
        self.n_agt = n_agt
        self.beta = beta
        self.theta = 0.3250
        self.nu = 0.580
        self.delta = 0.060
        self.GAMY = 1.0160
        self.BETA = 0.9540
        self.eta = 3.6142
        self.B = 0.002
        self.Z = np.array([0.9328, 0.9658, 1.0000, 1.0354, 1.0720])
        self.nz = 5
        self.Pi = Pi = np.array([[0.8537, 0.1377, 0.0083, 0.0002, 0.0000],
                                [0.0344, 0.8579, 0.1035, 0.0042, 0.0001],
                                [0.0014, 0.0690, 0.8593, 0.0690, 0.0014],
                                [0.0001, 0.0042, 0.1035, 0.8579, 0.0344],
                                [0.0000, 0.0002, 0.0083, 0.1377, 0.8537]])
        # 基本となるlinspace配列を作成し、次元を追加
        k_ss_single = np.linspace(0.1, 3.0, n_agt).reshape(1, n_agt, 1)  # 形状: (1, n_agt, 1)
        self.k_ss = np.repeat(k_ss_single, 384, axis=0)  # 形状: (384, n_agt, 1)
        