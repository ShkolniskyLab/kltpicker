# comparison between:
#   scipy.signal.fftconvolve
#   with and without pyfftw
# for the detect_particles function.

import numpy as np
from datetime import datetime
import pyfftw
from scipy.signal import fftconvolve as fc
import scipy


noise_mc = np.load("/home/dalitcohen/Documents/projects/test/numpy/before_construct_klt_templates/noise_mc.npy")
c = np.ones((39, 39)).astype("float32")

start = datetime.now()
for i in range(20):
    g = fc(noise_mc, c ,mode='valid')
print(datetime.now()-start)


def fftconvolve(A, B):
    MK =  B.shape[0]
    NK = B.shape[1]
    M = A.shape[0]
    N = A.shape[1]
    
    Y = M - MK + 1
    X = N - NK + 1
    M = M + MK - 1
    N = N + NK - 1

    a = np.pad(A, ((0, MK - 1), (0, NK - 1)), mode='constant')
    b = np.pad(B, ((0, M - 1), (0, N - 1)), mode='constant')

    fft_A_obj = pyfftw.builders.rfft2(a, s=(M, N), threads=8)
    fft_B_obj = pyfftw.builders.rfft2(b, s=(M, N), threads=8)
    ifft_obj = pyfftw.builders.irfft2(fft_A_obj.output_array, s=(M, N), threads=8)
    
    offset_Y = int(np.floor((M - Y)/2))
    offset_X = int(np.floor((N - X)/2))

    return ifft_obj(fft_A_obj(a) * fft_B_obj(b))[offset_Y:offset_Y + Y, offset_X:offset_X + X]

start = datetime.now()
for i in range(20):
    res = fftconvolve(noise_mc, c)
print(datetime.now()-start)


start = datetime.now()
for i in range(20):
    res = scipy.ndimage.convolve(noise_mc, c)
print(datetime.now()-start)


def fftconvolve_np(A, B):
    MK =  B.shape[0]
    NK = B.shape[1]
    M = A.shape[0]
    N = A.shape[1]
    
    Y = M - MK + 1
    X = N - NK + 1
    M = M + MK - 1
    N = N + NK - 1

    a = np.pad(A, ((0, MK - 1), (0, NK - 1)), mode='constant')
    b = np.pad(B, ((0, M - 1), (0, N - 1)), mode='constant')

    offset_Y = int(np.floor((M - Y)/2))
    offset_X = int(np.floor((N - X)/2))
    print("here i am")

    return np.fft.ifft(np.fft.fft(a) * np.fft.fft(b))[offset_Y:offset_Y + Y, offset_X:offset_X + X]

#noise_mc_np = np.asarray(noise_mc)
#c_np = np.asarray(c)

noise_mc = np.load("/home/dalitcohen/Documents/projects/test/numpy/before_construct_klt_templates/noise_mc.npy")
c = np.ones((39, 39)).astype(np.float64)


start = datetime.now()
for i in range(1):
    res = fftconvolve_np(noise_mc, c)
print(datetime.now()-start)



print("done")
