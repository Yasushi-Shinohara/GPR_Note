import numpy as np
from scipy.stats import norm

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
