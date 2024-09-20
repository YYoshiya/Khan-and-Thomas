import numpy as np
from scipy.interpolate import RectBivariateSpline

def lumpyeqouter(v, BetaK, knotsk, knotsm, Z, Pi, izvec):
    global GAMY, BETA, kSS

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

    for time in range(simT):
        mnow = np.dot(Thetanow, Kvecnow)
        iz = izvec[time]

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

        Zvec[time] = znow
        Kvec[time] = mnow
        Kpvec[time] = Kpagg
        Yvec[time] = Yagg
        Ivec[time] = Iagg
        Cvec[time] = Cagg
        Nvec[time] = Nagg
        Wvec[time] = wnew

        if time % 100 == 0:
            print(f'  time = {time}: pnow = {pnew:.4f}, pl = {plow:.4f}')

    print(f'  Elapsed time = {time.time() - start_time:.8f} seconds')

    return Yvec, Ivec, Cvec, Nvec, Wvec, Zvec, Kvec, Kpvec
