import numpy as np
from scipy.ndimage import convolve1d
from datetime import datetime
import scipy

def trank(m, s, tol):
    ss = s * s
    t = (1-tol) * np.sum(ss)
    r = 0
    sm = 0
    while sm < t:
        sm += ss[r]
        r += 1
    return r


def convolve2d(x, m, tol):
    mx, nx = x.shape
    mm, nm = m.shape
    u,s,v = np.linalg.svd(m)
    rank = trank(m, s, tol)   
    t = u
    u = v
    v = t
    vp = v.transpose()         
    y = convolve1d(convolve1d(x, u[:,0] * s[0]), vp[0,:])
    for r in range (1, rank):
        y = y + convolve1d(convolve1d(x, u[:,r] * s[r]), vp[r,:])
    return y

noise_mc = np.load("/home/dalitcohen/Documents/projects/test/numpy/before_construct_klt_templates/noise_mc.npy")
c = np.ones((39, 39)).astype(np.float64)

start = datetime.now()
for i in range(20):
    res = convolve2d(noise_mc, c, 0)
print(datetime.now()-start)



start = datetime.now()
for i in range(20):
    g = scipy.signal.fftconvolve(noise_mc, c ,'valid')
print(datetime.now()-start)

print("donnnn")