import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter
from scipy.fftpack import fftshift
try:
    import cupy as cp
except:
    pass

def fftcorrelate(image, filt):
    """
    Cross-correlate an image and given filter using FFT.
    
    Parameters
    ----------
    image : numpy.ndarray
        First input.
    filt : numpy.ndarray
        Second input.
    
    Returns
    -------
    result : numpy.ndarray
        A 2-dimensional array containing a subset of the discrete linear 
        cross-correlation of image with filt.
    """  
    filt = np.rot90(filt, 2)
    pad_shift = 1 - np.mod(np.array(filt.shape), 2)
    filt_center = np.floor((np.array(filt.shape) + 1) / 2).astype("int")
    pad = np.array(filt.shape) - filt_center
    padded_image = np.zeros((image.shape[0] + 2 * pad[0], image.shape[1] + 2 * pad[1]))
    padded_image[pad[0]: pad[0] + image.shape[0], pad[1]: pad[1] + image.shape[1]] = image
    if np.any(pad_shift):
        padded_image = padded_image[pad_shift[0] - 1: -1, pad_shift[1] - 1: -1]
    result = signal.fftconvolve(padded_image, filt, 'valid')
    return result

def f_trans_2(b):
    """
    2-D FIR filter using frequency transformation.

    Produces the 2-D FIR filter h that corresponds to the 1-D FIR filter b 
    using the McClellan transform.
    """
    # McClellan transformation:
    t = np.array([[1, 2, 1], [2, -4, 2], [1, 2, 1]]) / 8
    n = int((b.size - 1) / 2)
    b = np.flip(b, 0)
    b = fftshift(b)
    b = np.flip(b, 0)
    a = 2 * b[0:n + 1]
    a[0] = a[0] / 2
    # Use Chebyshev polynomials to compute h:
    p0 = 1
    p1 = t
    h = a[1] * p1
    rows = 1
    cols = 1
    h[rows, cols] = h[rows, cols] + a[0] * p0
    p2 = 2 * signal.convolve2d(t, p1)
    p2[2, 2] = p2[2, 2] - p0
    for i in range(2, n + 1):
        rows = p1.shape[0] + 1
        cols = p1.shape[1] + 1
        hh = h
        h = a[i] * p2
        h[1:rows, 1:cols] = h[1:rows, 1:cols] + hh
        p0 = p1
        p1 = p2
        rows += 1
        cols += 1
        p2 = 2 * signal.convolve2d(t, p1)
        p2[2:rows, 2:cols] = p2[2:rows, 2:cols] - p0
    h = np.rot90(h, k=2)
    return h

def radial_avg(z, m, bins):
    """
    Radially average 2-D square matrix z into m bins.

    Computes the average along the radius of a unit circle inscribed in the 
    square matrix z. The average is computed in m bins. 
    The radial average is not computed beyond the unit circle, in the corners
    of the matrix z. The radial average is returned in zr.
    
    Parameters
    ----------
    z : numpy.ndarray
        2-D square matrix.
    m : int
        Number of bins.
    bins : numpy.ndarray
        The bins.

    Returns
    -------
    zr : numpy.ndarray
        Radial average of z.
    R : numpy.ndarray
        Mid-points of the bins.
    """
    zr = np.zeros(m)
    for j in range(m):
        n = bins[j][0].size
        if n:
            zr[j] += np.sum(z[bins[j]])/n
        else:
            zr[j] = np.nan
    return zr

def stdfilter(a, nhood):
    "Local standard deviation of image."
    c1 = uniform_filter(a, nhood, mode='reflect')
    c2 = uniform_filter(a * a, nhood, mode='reflect')
    return np.sqrt(c2 - c1 * c1) * np.sqrt(nhood ** 2. / (nhood ** 2 - 1))

def trig_interpolation_mat(x, xq):
    """
    Matrix for trigonometric interpolation.
    When calculating an interpolant many times with the same nodes and 
    evaluation points, but with different interpolation values at the nodes,
    it is computationally more efficient to construct the following matrix
    once and calculate the trigonometric interpolant by p = dot(mat, y), where
    y is the vector of interpolation values at the nodes x.

    Parameters
    ----------
    x : numpy.ndarray
        interpolation nodes (vector).
    xq : numpy.ndarray
        evaluation points for the interpolant (vector).

    Returns
    -------
    mat : numpy.ndarray
        p = dot(mat, y) where p is the trigonometric interpolant, and y
        are the interpolation values at x.
    """
    n = x.size
    scale = n * (x[1] - x[0]) / 2
    xs = (x / scale) * np.pi / 2
    xi = (xq / scale) * np.pi / 2
    mat = np.zeros((xq.size, x.size))
    if n % 2:
        for k in range(n):
            a = np.sin(n * (xi - xs[k])) / (n * np.sin(xi - xs[k]))
            a[(xi - xs[k]) == 0] = 1
            mat[:, k] = a
    else:
        for k in range(n):
            a = np.sin(n * (xi - xs[k])) / (n * np.tan(xi - xs[k]))
            a[(xi - xs[k]) == 0] = 1
            mat[:, k] = a
    return mat

def trig_interpolation(x, y, xq):
    """   
    Trigonometric interpolation.
    
    Parameters
    ----------
    x : numpy.ndarray
        Interpolation nodes (vector).
    y : numpy.ndarray
        Interpolation values at nodes (vector).
    xq : numpy.ndarray
        Evaluation points for the interpolant (vector).
        
    Returns
    -------
    p : numpy.ndarray
        Values of the trigonometric interpolant (vector).
    """
    n = x.size
    h = 2 / n
    scale = (x[1] - x[0]) / h
    xs = (x / scale) * np.pi / 2
    xi = (xq / scale) * np.pi / 2
    p = np.zeros(xi.size)
    if n % 2:
        for k in range(n):
            a = np.sin(n * (xi - xs[k])) / (n * np.sin(xi - xs[k]))
            a[(xi - xs[k]) == 0] = 1
            p = p + y[k] * a
    else:
        for k in range(n):
            a = np.sin(n * (xi - xs[k])) / (n * np.tan(xi - xs[k]))
            a[(xi - xs[k]) == 0] = 1
            p = p + y[k] * a
    return p

def fftconvolve2d_gpu(x, y):
    """
    Convolve two 2-dimensional arrays using FFT, utilizing CuPy.
    The convolution is in valid mode, meaning that the output consists only of
    those elements that do not rely on the zero-padding.
    
    Parameters
    ----------
    x : numpy.ndarray
        First input.
    y : numpy.ndarray
        Second input.
    
    Returns
    -------
    z : numpy.ndarray
        A 2-dimensional array containing a subset of the discrete linear 
        convolution of x with y.
    """
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
