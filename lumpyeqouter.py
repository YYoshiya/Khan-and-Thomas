# lumpyeqouter.py
import numpy as np
from scipy.interpolate import RectBivariateSpline
import time
import globals
from scipy.optimize import fminbound

GAMY = globals.GAMY
BETA = globals.BETA
DELTA = globals.DELTA
THETA = globals.THETA
NU = globals.NU
ETA = globals.ETA
B = globals.B
critbp = globals.critbp

def lumpyeqouter(v, BetaK, knotsk, knotsm, Z, Pi, izvec, kSS):

    nk = len(knotsk)
    nm = len(knotsm)
    nz = len(Z)

    print('  OUTER LOOP')
    start_time = time.time()

    simT = len(izvec)
    Zvec = np.zeros(simT)
    Kvec = np.zeros(simT)
    Kpvec = np.zeros(simT)
    Yvec = np.zeros(simT)
    Ivec = np.zeros(simT)
    Cvec = np.zeros(simT)
    Nvec = np.zeros(simT)
    Wvec = np.zeros(simT)

    Thetanow = np.ones(nm) / nm  # 正規化
    Kvecnow = np.full(nm, kSS)

    # Fit spline for each productivity shock level `iz`
    splines = []
    for iz in range(nz):
        vcond = np.zeros((nk, nm))
        for jz in range(nz):
            vcond += Pi[iz, jz] * v[:, :, jz]  # E[V(k,K,z')|z]

        # Use RectBivariateSpline to fit a 2D spline for each `iz`
        spline = RectBivariateSpline(knotsk, knotsm, vcond)
        splines.append(spline)

    for t in range(simT):
        mnow = np.dot(Thetanow, Kvecnow)
        iz = int(izvec[t])  # izvecの値が整数であることを確認

        X = np.array([1, np.log(mnow)])
        mp = np.exp(np.dot(BetaK[iz, :], X))

        znow = Z[iz]
        spline = splines[iz]

        # Evaluate spline at `klow` and `khigh` points
        klow = 0.5 * mp
        khigh = 1.5 * mp

        edv1 = spline(klow, mp, dx=1, grid=False)
        phigh = BETA * edv1 / GAMY

        edv2 = spline(khigh, mp, dx=1, grid=False)
        plow = BETA * edv2 / GAMY

        # 自前の二分法を使用
        pnew, Thetanew, Kvecnew, Yagg, Iagg, Cagg, Nagg, wnew = bisectp(
            plow, phigh, spline, knotsk, knotsm, mp, znow, Thetanow, Kvecnow)

        # Update distribution
        Thetanow = Thetanew.copy()
        Kvecnow = Kvecnew.copy()

        # Record aggregate variables
        Kagg = mnow
        Zagg = znow
        Kpagg = np.dot(Thetanow, Kvecnow)

        Zvec[t] = znow
        Kvec[t] = mnow
        Kpvec[t] = Kpagg
        Yvec[t] = Yagg
        Ivec[t] = Iagg
        Cvec[t] = Cagg
        Nvec[t] = Nagg
        Wvec[t] = wnew

        if t % 100 == 0:
            print(f'  time = {t}: pnow = {pnew:.4f}, pl = {plow:.4f}')

    print(f'  Elapsed time = {time.time() - start_time:.8f} seconds')

    return Yvec, Ivec, Cvec, Nvec, Wvec, Zvec, Kvec, Kpvec

def bisectp(pL, pH, spline, knotsk, knotsm, mp, znow, Theta, Kvec):
    diff = 1e+4
    iter_count = 0

    while diff > critbp:
        p0 = (pL + pH) / 2
        pnew, Thetanew, Kvecnew, Yagg, Iagg, Cagg, Nagg, wnew = pricemapmy(
            p0, spline, knotsk, knotsm, mp, znow, Theta, Kvec)
        B0 = p0 - pnew  # g(w) = w - f(w)

        if B0 < 0:
            pL = p0
        else:
            pH = p0

        diff = pH - pL
        iter_count += 1

        # デバッグ情報を表示したい場合は以下をコメント解除
        # print(f'  bisection {iter_count},  pH-pL = {diff:.10f}')

    return pnew, Thetanew, Kvecnew, Yagg, Iagg, Cagg, Nagg, wnew

def pricemapmy(p, spline, knotsk, knotsm, mp, znow, Thetanow, Kvecnow):
    w = ETA / p

    # Optimize kpnew using fminbound
    res = fminbound(
        lambda kp: vfuncsp2(kp, mp, p, spline),
        knotsk[0], knotsk[-1],
        xtol=1e-5,
        full_output=True
    )
    kpnew = res[0]
    e0 = -vfuncsp2(kpnew, mp, p, spline)

    nk = len(Kvecnow)
    alpha = np.zeros(nk)
    Ivec = np.zeros(nk)
    Yvec = np.zeros(nk)
    Nvec = np.zeros(nk)

    for ik in range(nk):
        know = Kvecnow[ik]
        # Solve for n
        yterm = znow * know ** THETA
        nnow = (NU * yterm / w) ** (1 / (1 - NU))
        # Solve for xi
        v1 = spline((1 - DELTA) * know / GAMY, mp, grid=False)
        e1 = -p * (1 - DELTA) * know + BETA * v1
        xitemp = (e0 - e1) / (p * w)
        xi = min(B, max(0, xitemp))
        # Adjusting probability
        alpha[ik] = xi / B

        inow = alpha[ik] * (GAMY * kpnew - (1 - DELTA) * know)
        ynow = znow * know ** THETA * nnow ** NU

        # Distribution
        Ivec[ik] = inow
        Yvec[ik] = ynow
        Nvec[ik] = nnow + xi ** 2 / (2 * B)

    # Next theta and kvec
    indices_alpha_lt_1 = np.where(alpha < 1)[0]
    if len(indices_alpha_lt_1) == 0:
        J = -1
    else:
        J = indices_alpha_lt_1.max()

    Thetanew = np.zeros(J + 2)
    Kvecnew = np.zeros(J + 2)
    Thetanew[0] = np.dot(alpha, Thetanow)
    Thetanew[1:J+2] = (1 - alpha[:J+1]) * Thetanow[:J+1]
    Kvecnew[0] = kpnew
    Kvecnew[1:J+2] = (1 - DELTA) / GAMY * Kvecnow[:J+1]

    Kpagg = np.dot(Thetanew, Kvecnew)
    Iagg = np.dot(Thetanow, Ivec)
    Yagg = np.dot(Thetanow, Yvec)
    Cagg = Yagg - Iagg
    Nagg = np.dot(Thetanow, Nvec)
    pnew = 1 / Cagg
    wnew = ETA * Cagg

    return pnew, Thetanew, Kvecnew, Yagg, Iagg, Cagg, Nagg, wnew

def vfuncsp2(kp, mp, p, spline):
    # スプライン補間を使って ev を計算
    ev = spline(kp, mp, grid=False)
    # vfuncsp2 の計算
    f = -GAMY * p * kp + BETA * ev
    return -f  # 最大化のために符号を反転
