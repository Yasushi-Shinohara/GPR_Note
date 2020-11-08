import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels

class Gaussian_Process_Regression(GaussianProcessRegressor):
    def xsample2meanstd(self, _xsample, _ysample, _x, eps = 1.0e-8, normalization = True): #eps is ignored in this function and given in the parent class
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
        _Xsample = _xsample.reshape(-1,1)
        self.fit(_Xsample, _ysample)
        _X = _x.reshape(-1,1)
        mean, std = self.predict(_X, return_std=True) #Get the maen and standard error for the first trial set
        if (normalization):
            mean = mean*(ymax - ymin) + ymid
            std = std*(ymax -ymin)
        return mean, std

def Regression1D(GPR_skl,_xsample, _ysample, _x, normalization = True):
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
    _Xsample = _xsample.reshape(-1,1)
    GPR_skl.fit(_Xsample, _ysample)
    _X = _x.reshape(-1,1)
    mean, std = GPR_skl.predict(_X, return_std=True) #Get the maen and standard error for the first trial set
    if (normalization):
        mean = mean*(ymax - ymin) + ymid
        std = std*(ymax -ymin)
    return mean, std