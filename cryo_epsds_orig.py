import numpy as np
from pyfftw.interfaces.numpy_fft import fft2, ifft2
from pyfftw import FFTW
from numba import jit
from datetime import datetime

def fast_cfft2(x, axes=(-1, -2)):
    if len(x.shape) == 2:
        return np.fft.fftshift(np.transpose(fft2(np.transpose(np.fft.ifftshift(x)))))
    elif len(x.shape) == 3:
        y = np.fft.ifftshift(x, axes=axes)
        y = fft2(y, axes=axes)
        y = np.fft.fftshift(y, axes=axes)
        return y

def cryo_epsds(imstack, samples_idx, max_d):
    p = imstack.shape[0]
    if max_d >= p:
        max_d = p - 1

    r, x, _ = cryo_epsdr(imstack, samples_idx, max_d)

    r2 = np.zeros((2 * p - 1, 2 * p - 1))
    dsquare = np.square(x)
    for i in range(-max_d, max_d + 1):
        for j in range(-max_d, max_d + 1):
            d = i ** 2 + j ** 2
            if d <= max_d ** 2:
                idx, _ = bsearch(dsquare, d * (1 - 1e-13), d * (1 + 1e-13))
                r2[i + p - 1, j + p - 1] = r[idx - 1]

    w = gwindow(p, max_d)
    p2 = fast_cfft2(r2 * w)

    p2 = p2.real

    e = 0
    for i in range(imstack.shape[2]):
        im = imstack[:, :, i]
        e += np.sum(np.square(im[samples_idx] - np.mean(im[samples_idx])))

    mean_e = e / (len(samples_idx[0]) * imstack.shape[2])
    p2 = (p2 / p2.sum()) * mean_e * p2.size
    neg_idx = np.where(p2 < 0)
    p2[neg_idx] = 0
    return p2, r, r2, x


def cryo_epsdr(vol, samples_idx, max_d):
    p = vol.shape[0]
    k = vol.shape[2]
    i = np.array([[x for x in range(max_d + 1)] for x in range(max_d + 1)])
    j = i.transpose()
    dists = i ** 2 + j ** 2
    dsquare = np.sort(np.unique(dists[np.where(dists <= max_d ** 2)]))
    # dsquare = []
    # for d in dists:
    #     if d not in dsquare and d <= max_d ** 2:
    #         dsquare.append(d)
    # dsquare = np.array(dsquare)
    # dsquare.sort()
    
    corrs = np.zeros(len(dsquare))
    corr_count = np.zeros(len(dsquare))
    x = dsquare ** 0.5

    dist_map = np.zeros(dists.shape)
    for i in range(max_d + 1):
        for j in range(max_d + 1):
            d = i ** 2 + j ** 2
            if d <= max_d ** 2:
                idx, _ = bsearch(dsquare, d - 1e-13, d + 1e-13)
                dist_map[i, j] = idx

    dist_map = dist_map.astype('int') - 1
    valid_dists = np.where(dist_map != -1)

    mask = np.zeros((p, p))
    mask[samples_idx] = 1
    tmp = np.zeros((2 * p + 1, 2 * p + 1))
    tmp[:p, :p] = mask
    ftmp = fft2(tmp)
    c = ifft2(ftmp * np.conj(ftmp))
    c = c[:max_d + 1, :max_d + 1]
    c = np.round(c.real).astype('int')

    r = np.zeros(len(corrs))

    # optimized version
    vol = vol.transpose((2, 0, 1)).copy()
    input_fft2 = np.zeros((2 * p + 1, 2 * p + 1), dtype='complex128')
    output_fft2 = np.zeros((2 * p + 1, 2 * p + 1), dtype='complex128')
    input_ifft2 = np.zeros((2 * p + 1, 2 * p + 1), dtype='complex128')
    output_ifft2 = np.zeros((2 * p + 1, 2 * p + 1), dtype='complex128')
    flags = ('FFTW_MEASURE', 'FFTW_UNALIGNED')
    a_fft2 = FFTW(input_fft2, output_fft2, axes=(0, 1), direction='FFTW_FORWARD', flags=flags)
    a_ifft2 = FFTW(input_ifft2, output_ifft2, axes=(0, 1), direction='FFTW_BACKWARD', flags=flags)
    sum_s = np.zeros(output_ifft2.shape, output_ifft2.dtype)
    sum_c = c * vol.shape[0]
    for i in range(k):
        proj = vol[i]

        input_fft2[samples_idx] = proj[samples_idx]
        a_fft2()
        np.multiply(output_fft2, np.conj(output_fft2), out=input_ifft2)
        a_ifft2()
        sum_s += output_ifft2

    for curr_dist in zip(valid_dists[0], valid_dists[1]):
        dmidx = dist_map[curr_dist]
        corrs[dmidx] += sum_s[curr_dist].real
        corr_count[dmidx] += sum_c[curr_dist]

    idx = np.where(corr_count != 0)[0]
    r[idx] += corrs[idx] / corr_count[idx]
    cnt = corr_count[idx]

    idx = np.where(corr_count == 0)[0]
    r[idx] = 0
    x[idx] = 0
    return r, x, cnt

@jit(nopython=True)
def bsearch(x, lower_bound, upper_bound):
    """
    Binary search in a sorted vector.
    
    Binary O(log2(N)) search of the range of indices of all elements of x 
    between LowerBound and UpperBound. If no elements between LowerBound and
    Upperbound are found, the returned lower_index and upper_index are empty.
    The array x is assumed to be sorted from low to high, and is NOT verified
    for such sorting. 
    Based on code from 
    http://stackoverflow.com/questions/20166847/faster-version-of-find-for-sorted-vectors-matlab

    Parameters
    ----------
    x : numpy.ndarray
        A vector of sorted values from low to high.
    lower_bound : float
        Lower boundary on the values of x in the search.
    upper_bound : flo
        Upper boundary on the values of x in the search.

    Returns
    -------
    lower_idx: int
        The smallest index such that LowerBound<=x(index)<=UpperBound.
    upper_idx: int
        The largest index such that LowerBound<=x(index)<=UpperBound.

    """
    if lower_bound > x[-1] or upper_bound < x[0] or upper_bound < lower_bound:
        return None, None
    lower_idx_a = 1
    lower_idx_b = len(x)
    upper_idx_a = 1
    upper_idx_b = len(x)

    while lower_idx_a + 1 < lower_idx_b or upper_idx_a + 1 < upper_idx_b:
        lw = int(np.floor((lower_idx_a + lower_idx_b) / 2))
        if x[lw - 1] >= lower_bound:
            lower_idx_b = lw
        else:
            lower_idx_a = lw
            if upper_idx_a < lw < upper_idx_b:
                upper_idx_a = lw

        up = int(np.ceil((upper_idx_a + upper_idx_b) / 2))
        if x[up - 1] <= upper_bound:
            upper_idx_a = up
        else:
            upper_idx_b = up
            if lower_idx_a < up < lower_idx_b:
                lower_idx_b = up

    if x[lower_idx_a - 1] >= lower_bound:
        lower_idx = lower_idx_a
    else:
        lower_idx = lower_idx_b
    if x[upper_idx_b - 1] <= upper_bound:
        upper_idx = upper_idx_b
    else:
        upper_idx = upper_idx_a

    if upper_idx < lower_idx:
        return None, None

    return lower_idx, upper_idx

@jit(nopython=True)
def gwindow(p, max_d):
    """
    Create 2D Gaussian window for spectral estimation. Return a (2p-1)x(2p-1) 
    Gaussian window to be used for 2D power spectrum estimation. 

    Parameters
    ----------
    p : int
        Size of the returned mask.
    max_d : int
        Width of the Gaussian.

    Returns
    -------
    w : numpy.ndarray
        (2p-1)x(2p-1) array..

    """
    l = 2 * p - 1
    y = np.array([[x for x in range(l)] for x in range(l)])
    x = y - p + 1
    alpha = float(3)
    # Reciprocal of the standard deviation of the Gaussian window. 1/alpha is
    # the width of the Fourier transform of the window.See Harris 78 for more
    # details. 
    w = np.exp(-alpha * (x ** 2 + x.transpose() ** 2) / (2 * max_d ** 2))
    return w

@jit(nopython=True)
def accumelate_corrs(dsquare_len, valid_dists, dist_map, sum_c, sum_s):
    corr_count = np.zeros(dsquare_len)
    corrs = np.zeros(dsquare_len)
    for curr_dist in zip(valid_dists[0], valid_dists[1]):
        dmidx = dist_map[curr_dist]
        corrs[dmidx] += sum_s[curr_dist].real
        corr_count[dmidx] += sum_c[curr_dist]
    return corr_count, corrs

@jit(nopython=True)        
def autocorr_2d(max_d, x, r, p):
    """
    Use the 1D autocorrelation r, x to populate an array r2 of the 2D
    isotropic autocorrelation.

    Parameters
    ----------
    max_d : int
        The 1D autocorrelations were computed up to a maximal distance of 
        max_d pixels.
    x : numpy.ndarray
        Distaces at which the samples of the autocorrelation function are
        given. A vector of the same length as r.
    r : numpy.ndarray
        1D vector with samples of the isotropic autocorrelation function.
    p : int
        The dimension of the square image that is being processed.

    Returns
    -------
    r2 : numpy.ndarray
        2D isotropic autocorrelation.

    """
    dsquare = x ** 2
    r2 = np.zeros((int(2 * p - 1), int(2 * p - 1)), dtype=np.int64)
    for i in range(-max_d, max_d + 1):
        for j in range(-max_d, max_d + 1):
            d = i ** 2 + j ** 2
            if d <= max_d ** 2:
                idx, _ = bsearch(dsquare, d * (1 - 1e-13), d * (1 + 1e-13))
                r2[i + p - 1, j + p - 1] = r[int(idx - 1)]        
    return r2

@jit(nopython=True)
def distmap(max_d, dsquare, dists_shape):
    """
    Create a distance map whose value at the index (i,j) is the index in the
    array dsquare whose value is i^2+j^2. Pairs (i,j) that correspond to
    distances that are larger than max_d are indicated by (-1).  
    """
    dist_map = np.zeros(dists_shape)
    for i in range(max_d + 1):
        for j in range(max_d + 1):
            d = i ** 2 + j ** 2
            if d <= max_d ** 2:
                idx, _ = bsearch(dsquare, d - 1e-13, d + 1e-13)
                dist_map[i, j] = idx
    dist_map = dist_map.astype(np.int32) - 1
    return dist_map

patch_size = 79

block = np.load("/home/dalitcohen/Documents/tmp.npy")
start = datetime.now()
for i in range(1000):
    psd_block1 = cryo_epsds(block[:,:,np.newaxis],
                       np.where(np.zeros((int(patch_size), int(patch_size))) == 0),
                       int(np.floor(0.3 * patch_size)))
    
print(datetime.now()-start)
print(psd_block1[0])