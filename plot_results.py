# plot_results.py
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.filters.hp_filter import hpfilter


# 保存されたデータを読み込む
data = np.load('lumpyeqksresults.npz')
v = data['v']
BetaK = data['BetaK']
Betap = data['Betap']
Kvec = data['Kvec']
Kpvec = data['Kpvec']
Cvec = data['Cvec']
izvec = data['izvec']
Yvec = data['Yvec']
Ivec = data['Ivec']
Nvec = data['Nvec']
Wvec = data['Wvec']
Zvec = data['Zvec']

# 定数の設定
Z = np.array([0.9328, 0.9658, 1.0000, 1.0354, 1.0720])
GAMY  = 1.0160
BETA  = 0.9540
DELTA = 0.06

# プロットの作成
fig, axs = plt.subplots(3, 1, figsize=(12, 18))

# サブプロット311: Cのプロット
axs[0].plot(Cvec, 'b-', linewidth=2.0)
axs[0].set_xlim([0, len(Cvec)])
axs[0].set_ylabel('C', fontsize=14)
axs[0].set_xlabel('期間', fontsize=14)
axs[0].grid(True)
axs[0].tick_params(labelsize=14)
axs[0].set_facecolor('w')

# サブプロット312: K'のプロット
axs[1].plot(Kpvec, 'b-', linewidth=2.0)
axs[1].set_xlim([0, len(Kpvec)])
axs[1].set_ylabel("K'", fontsize=14)
axs[1].set_xlabel('期間', fontsize=14)
axs[1].grid(True)
axs[1].tick_params(labelsize=14)
axs[1].set_facecolor('w')

# サブプロット313: Aのプロット
A_series = Z[izvec]
axs[2].plot(A_series, 'r-', linewidth=2.0)
axs[2].set_xlim([0, len(A_series)])
axs[2].set_ylabel('A', fontsize=14)
axs[2].set_xlabel('期間', fontsize=14)
axs[2].grid(True)
axs[2].tick_params(labelsize=14)
axs[2].set_facecolor('w')

plt.tight_layout()
plt.savefig('KTsim.eps', format='eps')
plt.show()

# 追加変数の計算
Yvec_calc = Cvec + Kpvec - (1 - DELTA) * Kvec
Xvec = Kpvec - (1 - DELTA) * Kvec
Rvec = np.concatenate(([GAMY / BETA], Cvec[1:] / Cvec[:-1] * GAMY / BETA))

# HPフィルタの適用（ログ変換）
# インデックスは0から始まるため、501-1=500以降を使用
y_hp, _ = hpfilter(np.log(Yvec_calc[500:]), lamb=100)
c_hp, _ = hpfilter(np.log(Cvec[500:]), lamb=100)
x_hp, _ = hpfilter(np.log(Xvec[500:]), lamb=100)
n_hp, _ = hpfilter(np.log(Nvec[500:]), lamb=100)
w_hp, _ = hpfilter(np.log(Wvec[500:]), lamb=100)
r_hp, _ = hpfilter(np.log(Rvec[500:]), lamb=100)

# 標準偏差の計算
std_y = np.std(y_hp) * 100
std_c = np.std(c_hp) / np.std(y_hp)
std_x = np.std(x_hp) / np.std(y_hp)
std_n = np.std(n_hp) / np.std(y_hp)
std_w = np.std(w_hp) / np.std(y_hp)
std_r = np.std(r_hp) / np.std(y_hp)

print("標準偏差および比率:")
print([std_y, std_c, std_x, std_n, std_w, std_r])

# 相関係数の計算
corr_y_c = np.corrcoef(y_hp, c_hp)[0, 1]
corr_y_x = np.corrcoef(y_hp, x_hp)[0, 1]
corr_y_n = np.corrcoef(y_hp, n_hp)[0, 1]
corr_y_w = np.corrcoef(y_hp, w_hp)[0, 1]
corr_y_r = np.corrcoef(y_hp, r_hp)[0, 1]

print("相関係数:")
print([corr_y_c, corr_y_x, corr_y_n, corr_y_w, corr_y_r])
