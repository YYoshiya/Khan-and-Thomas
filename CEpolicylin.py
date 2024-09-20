# CEpolicylin.py
import numpy as np

def CEpolicylin(ivec, kvec, kprime, p, znum, simLength):
    """
    CEpolicylin function: Perform linear regressions for the Krusell-Smith model.
    """
    BetaK = np.zeros((znum, 2))
    Betap = np.zeros((znum, 2))

    for i in range(znum):
        zloc = np.where(ivec[:simLength] == i)[0]
        zobs = len(zloc)
        
        if zobs == 0:
            continue
        
        kz = kvec[zloc]
        kzprime = kprime[zloc]
        pz = p[zloc]
        
        klog = np.log(kz)
        plog = np.log(pz)
        kprimelog = np.log(kzprime)
        unit = np.ones(zobs)
        
        # Independent and dependent variables for linear regression in logs
        Y1 = np.column_stack((kprimelog, plog))
        X2 = np.column_stack((unit, klog))
        
        # Multivariate linear regression
        Beta2 = np.linalg.lstsq(X2, Y1, rcond=None)[0]
        
        # Estimates
        estimates2 = X2 @ Beta2.T
        
        N = zobs - 2
        
        # Residuals and statistics for capital
        residkt = estimates2[:, 0] - kprimelog
        ssrkt = np.sum(residkt ** 2) / N
        ssdkt = np.sum((kprimelog - np.mean(kprimelog)) ** 2) / N
        R2kt = 1 - ssrkt / ssdkt
        ssrkt = np.sqrt(ssrkt)
        
        # Residuals and statistics for price
        residpt = estimates2[:, 1] - plog
        ssrpt = np.sum(residpt ** 2) / N
        ssdpt = np.sum((plog - np.mean(plog)) ** 2) / N
        R2pt = 1 - ssrpt / ssdpt
        ssrpt = np.sqrt(ssrpt)
        
        # Debugging information
        print("\n")
        print(" -------------------------------------------------------------------------------- ")
        print(f" znum={i+1}:  Regression Coefficients based on log(x) = Beta0 + Beta1*log(k)")
        print(f" kf    {Beta2[0, 0]:+10.4f}  {Beta2[1, 0]:+10.4f}")
        print(f"  p    {Beta2[0, 1]:+10.4f}  {Beta2[1, 1]:+10.4f}")
        print("\n")
        print(f" znum={i+1}:  Regression Statistics based on {zobs} observations ")
        print(f" kf     {np.min(np.abs(residkt)):8.4e}   {np.max(np.abs(residkt)):8.4e}   {ssrkt:8.4e}    {R2kt:12.10f}")
        print(f"  p     {np.min(np.abs(residpt)):8.4e}   {np.max(np.abs(residpt)):8.4e}   {ssrpt:8.4e}    {R2pt:12.10f}")
        print(" -------------------------------------------------------------------------------- ")
        
        # Store the coefficients
        BetaK[i, :] = Beta2[:, 0]
        Betap[i, :] = Beta2[:, 1]
    
    return BetaK, Betap
