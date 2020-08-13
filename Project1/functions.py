import numpy as np
#import matplotlib.pyplot as plt
from scipy.stats import norm

class Gaussian_Process_Regression():
    def __init__(self):
        self.K = None
        self.kernel_name1 = 'RBF'
        self.a1_1 = 200.0
        self.a2_1 = 20.0
        self.a3_1 = 0.0
        

    def xx2K(self,xn,xm):
        if (len(xn) == 1):
            self.K = np.zeros([1,len(xm)])
        else :
            self.K = 0.0*np.outer(xn,xm)
        for i in range(len(xn)):
            self.K[i,:] = self.a1_1*np.exp(-(xn[i] - xm[:])**2/self.a2_1) + self.a3_1
        return self.K
    
    def xsample2meanvariance(self,_xsample, _ysample, _x, eps = 1.0e-8):
        self.K = self.xx2K(_xsample,_xsample) + eps*np.eye(len(_xsample))
        L = np.linalg.cholesky(self.K)
        #plt.matshow(K)
        #plt.matshow(L)
        kast = self.xx2K(_xsample,_x)
        kastast = self.xx2K(_x,_x)
        w = np.linalg.solve(L, _ysample)
        z = np.linalg.solve(L.T, w)
        mean = np.dot(kast.T, z)
        W = np.linalg.solve(L, kast)
        Z = np.linalg.solve(L.T, W)
        fvariance = kastast - np.dot(kast.T, Z)
        fvariance = np.diag(fvariance)
        std = np.sqrt(fvariance)
        return mean, std
