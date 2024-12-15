# main.py
import numpy as np
import time
from CEpolicylin import CEpolicylin
from lumpyeqinner import lumpyeqinner
from lumpyeqouter import lumpyeqouter
import scipy.io
import globals

# Start timer
start_time = time.time()

# Convergence criteria
critout = 1e-4  # for outer loop
critin  = 1e-5  # for inner loop

GAMY = globals.GAMY
BETA = globals.BETA
DELTA = globals.DELTA
THETA = globals.THETA
NU = globals.NU
ETA = globals.ETA
B = globals.B
critin = globals.critin

ykSS = (GAMY - BETA * (1 - DELTA)) / BETA / THETA
ckSS = ykSS + (1 - GAMY - DELTA)
ycSS = ykSS / ckSS
nSS = NU / ETA * ycSS
kSS = (ykSS * nSS ** (-NU)) ** (1 / (THETA - 1))

# Aggregate shock process
Z = np.array([0.9328, 0.9658, 1.0000, 1.0354, 1.0720])
nz = 5
Pi = np.array([[0.8537, 0.1377, 0.0083, 0.0002, 0.0000],
               [0.0344, 0.8579, 0.1035, 0.0042, 0.0001],
               [0.0014, 0.0690, 0.8593, 0.0690, 0.0014],
               [0.0001, 0.0042, 0.1035, 0.8579, 0.0344],
               [0.0000, 0.0002, 0.0083, 0.1377, 0.8537]])

kbounds = [0.1, 3.0]
nk = 25
knotsk = np.logspace(np.log10(kbounds[0] + -1.0 * kbounds[0] + 1.0), 
                     np.log10(kbounds[1] + -1.0 * kbounds[0] + 1.0), nk)
knotsk += (kbounds[0] - 1.0)
rk = nk - 2

mbounds = [0.8, 1.2]
nm = 5
knotsm = np.linspace(mbounds[0], mbounds[1], nm)
rm = nm - 2

# Load the PlannerSim data from files
data = scipy.io.loadmat('PlannerSim.mat')

# Extract the variables from the loaded .mat file
Kvec = data['Kvec'].flatten()
Kpvec = data['Kpvec'].flatten()
Cvec = data['Cvec'].flatten()
izvec = data['izvec'].flatten().astype(int) - 1  # インデックスとして使用するため1を引いて整数型に変換
simT = len(Kvec)

# Get regression coefficients
BetaK, Betap = CEpolicylin(izvec, Kvec, Kpvec, 1.0/Cvec, nz, simT)

diff = 1e4
iter = 0

while diff > critout:
    iter_start_time = time.time()

    # Calculate value function
    v = lumpyeqinner(BetaK, Betap, knotsk, knotsm, Z, Pi)

    # Perform simulation
    Yvec, Ivec, Cvec, Nvec, Wvec, Zvec, Kvec, Kpvec = lumpyeqouter(v, BetaK, knotsk, knotsm, Z, Pi, izvec, kSS)

    BetaKnew = np.zeros((nz, 2))
    Betapnew = np.zeros((nz, 2))

    # Update coefficients
    for iz in range(nz):
        x = Kvec[izvec == iz]
        y = Kpvec[izvec == iz]
        p = 1.0 / Cvec[izvec == iz]

        X = np.column_stack((np.ones(len(x)), np.log(x)))
        y_vals = np.column_stack((np.log(y), np.log(p)))
        Beta = np.linalg.lstsq(X, y_vals, rcond=None)[0]
        BetaKnew[iz, :] = Beta[:, 0]
        Betapnew[iz, :] = Beta[:, 1]

    diffmp = np.max(np.abs(BetaKnew - BetaK))
    diffp = np.max(np.abs(Betapnew - Betap))
    diff = max(diffmp, diffp)

    iter += 1
    print(f"Iteration {iter}: ||Tmp-mp|| = {diffmp:.4f}, ||Tp-p|| = {diffp:.4f}, Elapsed time = {time.time() - iter_start_time:.4f}")

    BetaK = BetaKnew.copy()
    Betap = Betapnew.copy()

# Save the results
np.savez('lumpyeqksresults.npz',
         v=v,
         BetaK=BetaK,
         Betap=Betap,
         Kvec=Kvec,
         Kpvec=Kpvec,
         Cvec=Cvec,
         izvec=izvec,
         Yvec=Yvec,
         Ivec=Ivec,
         Nvec=Nvec,
         Wvec=Wvec,
         Zvec=Zvec)

print(f"Total elapsed time: {time.time() - start_time:.4f} seconds")
