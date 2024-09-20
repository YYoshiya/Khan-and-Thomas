import numpy as np
from scipy.interpolate import RectBivariateSpline
import time  # timeモジュールをインポート
import globals

GAMY = globals.GAMY
BETA = globals.BETA
DELTA = globals.DELTA
THETA = globals.THETA
NU = globals.NU
ETA = globals.ETA
B = globals.B
critin = globals.critin

def lumpyeqouter(v, BetaK, knotsk, knotsm, Z, Pi, izvec):

    nk = len(knotsk)
    nm = len(knotsm)
    nz = len(Z)

    print('  OUTER LOOP')
    start_time = time.time()  # 正しいtimeモジュールの呼び出し

    simT = len(izvec)
    Zvec = np.zeros(simT)
    Kvec = np.zeros(simT)
    Kpvec = np.zeros(simT)
    Yvec = np.zeros(simT)
    Ivec = np.zeros(simT)
    Cvec = np.zeros(simT)
    Nvec = np.zeros(simT)
    Wvec = np.zeros(simT)

    Thetanow = np.ones(nm)
    Kvecnow = kSS

    # Fit spline for each productivity shock level `iz`
    splines = []
    for iz in range(nz):
        vcond = np.zeros((nk, nm))
        for jz in range(nz):
            vcond += Pi[iz, jz] * v[:, :, jz]  # E[V(k,K,z')|z]
        
        # Use RectBivariateSpline to fit a 2D spline for each `iz`
        spline = RectBivariateSpline(knotsk, knotsm, vcond)
        splines.append(spline)

    for t in range(simT):  # timeをtに変更
        mnow = np.dot(Thetanow, Kvecnow)
        iz = izvec[t]  # timeをtに変更

        X = np.array([1, np.log(mnow)])
        mp = np.exp(np.dot(BetaK[iz, :], X))

        znow = Z[iz]
        spline = splines[iz]

        # Evaluate spline at `klow` and `khigh` points
        klow = 0.5 * mp
        khigh = 1.5 * mp

        ev1 = spline(klow, mp, grid=False)
        edv1 = spline(klow, mp, dx=1, grid=False)
        phigh = BETA * edv1 / GAMY

        ev2 = spline(khigh, mp, grid=False)
        edv2 = spline(khigh, mp, dx=1, grid=False)
        plow = BETA * edv2 / GAMY

        pnew, Thetanew, Kvecnew, Yagg, Iagg, Cagg, Nagg, wnew = bisectp(
            plow, phigh, spline, knotsk, knotsm, mp, znow, Thetanow, Kvecnow)

        # Update distribution
        Thetanow = Thetanew
        Kvecnow = Kvecnew

        # Record aggregate variables
        Kagg = mnow
        Zagg = znow
        Kpagg = np.dot(Thetanew, Kvecnew)

        Zvec[t] = znow  # timeをtに変更
        Kvec[t] = mnow  # timeをtに変更
        Kpvec[t] = Kpagg  # timeをtに変更
        Yvec[t] = Yagg  # timeをtに変更
        Ivec[t] = Iagg  # timeをtに変更
        Cvec[t] = Cagg  # timeをtに変更
        Nvec[t] = Nagg  # timeをtに変更
        Wvec[t] = wnew  # timeをtに変更

        if t % 100 == 0:  # timeをtに変更
            print(f'  time = {t}: pnow = {pnew:.4f}, pl = {plow:.4f}')

    print(f'  Elapsed time = {time.time() - start_time:.8f} seconds')

    return Yvec, Ivec, Cvec, Nvec, Wvec, Zvec, Kvec, Kpvec
