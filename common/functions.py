import numpy as np
#import matplotlib.pyplot as plt
from scipy.stats import norm

class Gaussian_Process_Regression():
    def __init__(self):
        self.K = None
#        self.kernel_name1 = 'RBF'
        self.a1_RBF = 1.0
        self.a2_RBF = 1.0
        self.a1_exp = 1.0
        self.a2_exp = 1.0
        #self.a1_linear = 1.0
        self.a1_const = 1.0
        

    def xx2K(self,xn,xm):
        if (len(xn) == 1):
            self.K = np.zeros([1,len(xm)])
        else :
            self.K = 0.0*np.outer(xn,xm)
        for i in range(len(xn)):
            self.K[i,:] = \
            self.a1_RBF*np.exp(-(xn[i] - xm[:])**2/self.a2_RBF) \
            +\
            self.a1_exp*np.exp(-np.abs(xn[i] - xm[:])/self.a2_exp)\
            +\
            self.a1_const
        return self.K
    
    def xsample2meanvariance(self,_xsample, _ysample, _x, eps = 1.0e-8, normalization = True):
        if (normalization):
            xmin = np.amin(_x)
            xmax = np.amax(_x)
            xmid = 0.5*(xmin + xmax)
            _x = (_x - xmid)/(xmax - xmin)
            _xsample = (_xsample - xmid)/(xmax - xmin)
            ymin = np.amin(_ysample)
            ymax = np.amax(_ysample)
            ymid = 0.5*(ymin + ymax)
            _ysample = (_ysample - ymid)/(ymax - ymin)            
        self.K = self.xx2K(_xsample,_xsample) + eps*np.eye(len(_xsample))
        L = np.linalg.cholesky(self.K)
        #plt.matshow(K)
        #plt.matshow(L)
        kast = self.xx2K(_xsample,_x)
        kastast = self.xx2K(_x,_x) + eps*np.eye(len(_x))
        w = np.linalg.solve(L, _ysample)
        z = np.linalg.solve(L.T, w)
        mean = np.dot(kast.T, z)
        W = np.linalg.solve(L, kast)
        Z = np.linalg.solve(L.T, W)
        fvariance = kastast - np.dot(kast.T, Z)
        fvariance = np.diag(fvariance)
        std = np.sqrt(np.abs(fvariance))
        if (normalization):
            mean = mean*(ymax - ymin) + ymid
            std = std*(ymax - ymin)
        return mean, std
#
def PF(_xsample, _ysample, _x, _deg, normalization = True):
    if (normalization):
        xmin = np.amin(_x)
        xmax = np.amax(_x)
        xmid = 0.5*(xmin + xmax)
        _x = (_x - xmid)/(xmax - xmin)
        _xsample = (_xsample - xmid)/(xmax - xmin)
        ymin = np.amin(_ysample)
        ymax = np.amax(_ysample)
        ymid = 0.5*(ymin + ymax)
        _ysample = (_ysample - ymid)/(ymax - ymin)
    Nsample = len(_xsample)
    polyinfo = np.polyfit(_xsample, _ysample, _deg)
    p = np.poly1d(polyinfo)
    mean = p(_x)
    if (normalization):
        mean = mean*(ymax - ymin) + ymid
    return mean