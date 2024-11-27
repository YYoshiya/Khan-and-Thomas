# lumpyeqinner.py
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import fminbound
import time
import globals

GAMY = globals.GAMY
BETA = globals.BETA
DELTA = globals.DELTA
THETA = globals.THETA
NU = globals.NU
ETA = globals.ETA
B = globals.B
critin = globals.critin

def lumpyeqinner(BetaK, Betap, knotsk, knotsm, Z, Pi):

    nk = len(knotsk)
    nm = len(knotsm)
    nz = len(Z)

    print('  INNER LOOP')
    start_time = time.time()

    # Part I. The Initial Value Function
    v0 = np.zeros((nk, nm, nz))
    nmat = np.zeros((nk, nm, nz))
    mpmat = np.zeros((nm, nz))
    pmat = np.zeros((nm, nz))
    wmat = np.zeros((nm, nz))

    for im in range(nm):
        for iz in range(nz):
            mnow = knotsm[im]
            znow = Z[iz]

            X = np.array([1, np.log(mnow)])
            mp = np.exp(np.dot(BetaK[iz, :], X))
            p0 = np.exp(np.dot(Betap[iz, :], X))

            mpmat[im, iz] = mp
            pmat[im, iz] = p0
            w0 = ETA / p0
            wmat[im, iz] = w0

            for ik in range(nk):
                know = knotsk[ik]
                yterm = znow * know**THETA
                n = (NU * yterm / w0)**(1 / (1 - NU))
                nmat[ik, im, iz] = n
                y = yterm * n**NU
                v0temp = y - w0 * n + (1 - DELTA) * know
                v12 = v0temp * p0
                v0[ik, im, iz] = v12

    v = v0
    kp = np.zeros((nm, nz))

    # Part II. Iteration on Contraction Mapping
    vnew = np.zeros((nk, nm, nz))
    kpnew = np.zeros((nm, nz))
    e0 = np.zeros((nm, nz))
    e1 = np.zeros((nk, nm, nz))
    xi = np.zeros((nk, nm, nz))

    diff = 1e4
    iter = 0
    s1 = 0

    # Creating a list of splines for each z (Productivity shocks)
    while diff > critin:
        splines = []
        for iz in range(nz):
            vcond = np.zeros((nk, nm))
            for jz in range(nz):
                vcond += Pi[iz, jz] * v[:, :, jz]  # E[V(k,K,z')|z]

            # Fit 2D spline with RectBivariateSpline
            spline = RectBivariateSpline(knotsk, knotsm, vcond)
            splines.append(spline)

            # target k
            if s1 == 0:
                for im in range(nm):
                    mp = mpmat[im, iz]
                    p = pmat[im, iz]
                    spline_func = lambda kp: vfuncsp2(kp, mp, p, spline)
                    kpnew[im, iz], _, _, _ = fminbound(spline_func, knotsk[0], knotsk[-1], full_output=True)

            # solve for xi(k,K,z)
            for im in range(nm):
                mp = mpmat[im, iz]
                p = pmat[im, iz]
                w = wmat[im, iz]
                e0[im, iz] = -vfuncsp2(kpnew[im, iz], mp, p, spline)

                for ik in range(nk):
                    know = knotsk[ik]
                    v1 = spline((1 - DELTA) / GAMY * know, mp, grid=False)
                    e1[ik, im, iz] = -p * (1 - DELTA) * know + BETA * v1

                    xitemp = (e0[im, iz] - e1[ik, im, iz]) / (p * w)
                    xi[ik, im, iz] = min(B, max(0, xitemp))

                    vnew[ik, im, iz] = (v0[ik, im, iz]
                        - p * w * xi[ik, im, iz]**2 / (2 * B) 
                        + xi[ik, im, iz] / B * e0[im, iz]
                        + (1 - xi[ik, im, iz] / B) * e1[ik, im, iz])

        diffkp = np.max(np.abs(kpnew - kp))
        diffv = np.max(np.abs(vnew - v))
        diff = diffv
        iter += 1
        print(f"Iteration {iter}: ||Tkp-kp|| = {diffkp:.8f}, ||Tv-v|| = {diffv:.8f}")

        if diffkp < 1e-4 and s1 == 0:
            s1 = 1
        elif s1 > 0 and s1 < 20:
            s1 += 1
        elif s1 >= 20:
            s1 = 0

        kp = kpnew.copy()
        v = vnew.copy()

    print(f"  Elapsed time = {time.time() - start_time:.8f} seconds")

    return v

def vfuncsp2(kp, mp, p, spline):
    # スプライン補間を使って ev を計算
    ev = spline(kp, mp, grid=False)
    # vfuncsp2 の計算
    f = -GAMY * p * kp + BETA * ev
    return -f  # 最大化のために符号を反転
