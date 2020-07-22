import cupy as cp
import numpy as np
from scipy.signal import fftconvolve
from datetime import datetime
image = np.load("/home/dalitcohen/Documents/projects/test/numpy/before_construct_klt_templates/noise_mc.npy")
kernel = np.ones((43,43))

start = datetime.now()
for i in range(20):
    res_scipy = fftconvolve(image[:-1,:-1], kernel,'valid')
print(datetime.now()-start)

def fftconvolve2d(x, y):
    x_shape = np.array(x.shape)
    y_shape = np.array(y.shape)
    z_shape = x_shape + y_shape - 1
    z = np.fft.ifft2(np.fft.fft2(x, tuple(z_shape)) * np.fft.fft2(y, tuple(z_shape))).real
    valid_shape = x_shape - y_shape + 1
    start = (z_shape - valid_shape) // 2
    end = start + valid_shape
    z = z[start[0]:end[0], start[1]:end[1]]
    return z

start = datetime.now()
for i in range(20):
    res = fftconvolve2d(image[:-1,:-1], kernel)

print(datetime.now()-start)

image = cp.load("/home/dalitcohen/Documents/projects/test/numpy/before_construct_klt_templates/noise_mc.npy")
kernel = cp.ones((43,43))

def fftconvolve2d_cp(x, y):
    xn, xm = x.shape
    yn, ym = y.shape
    zn = xn + yn -1
    zm = xm + ym -1
    z = cp.fft.ifft2(cp.fft.fft2(x, s=(zn, zm)) * cp.fft.fft2(y, s=(zn, zm))).real
    valid_n = xn - yn + 1
    valid_m = xm - ym + 1
    start_n = (zn - valid_n) // 2
    start_m = (zm - valid_m) // 2
    end_n = start_n + valid_n
    end_m = start_m + valid_m
    z = z[start_n:end_n, start_m:end_m]
    return z

start = datetime.now()
for i in range(20):
    res_cp = fftconvolve2d_cp(image[:-1,:-1], kernel)


print(datetime.now()-start)