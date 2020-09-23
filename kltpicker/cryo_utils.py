import numpy as np
from pyfftw.interfaces.numpy_fft import fft2, ifft2
from pyfftw import FFTW
from numpy.polynomial.legendre import leggauss
from numba import jit
try:
    import cupy as cp
except:
    pass

def downsample_gpu(image, size_out):
    """ 
    Use Fourier methods to change the sample interval and/or aspect ratio
    of any dimensions of the input image. Uses CuPy.
    """
    x = cp.fft.fftshift(cp.fft.fft2(image))   
    # crop x:
    nx, ny = x.shape   
    nsx = int(cp.floor(nx/2) - cp.floor(size_out[1]/2))
    nsy = int(cp.floor(ny/2) - cp.floor(size_out[0]/2))
    fx = x[nsx : nsx + size_out[1], nsy : nsy + size_out[0]]
    output = cp.fft.ifft2(cp.fft.ifftshift(fx)) * (cp.prod(cp.array(size_out)) / cp.prod(cp.array(image.shape)))  
    return cp.asnumpy(output.real)

def downsample(image, size_out):
    """ 
    Use Fourier methods to change the sample interval and/or aspect ratio
    of any dimensions of the input image. 
    """
    x = np.fft.fftshift(np.fft.fft2(image))   
    # crop x:
    nx, ny = x.shape   
    nsx = int(np.floor(nx/2) - np.floor(size_out[1]/2))
    nsy = int(np.floor(ny/2) - np.floor(size_out[0]/2))
    fx = x[nsx : nsx + size_out[1], nsy : nsy + size_out[0]]
    output = np.fft.ifft2(np.fft.ifftshift(fx)) * (np.prod(size_out) / np.prod(image.shape))  
    return output.real

def lgwt(n, a, b):
    """
    Get n leggauss points in interval [a, b]
    Parameters
    ----------
    n : int
        Number of points.
    a : float
        Interval starting point.
    b : float
        Interval end point.
    Returns
    -------
    x : numpy.ndarray
        Sample points.
    w : numpy.ndarray
        Weights.
    """
    x1, w = leggauss(n)
    m = (b - a) / 2
    c = (a + b) / 2
    x = m * x1 + c
    w = m * w
    x = np.flipud(x)
    return x, w

def cryo_epsds(image, samples_idx, max_d):
    """
    Estimate the 2D isotropic power spectrum of a given image using only the 
    pixels given in samples_idx. Typically, samples_idx will correspond to the
    pixles in the image that are outside a certain radius (where there is no
    particle). The power spectrum is estimated using the correlogram method.

    Parameters
    ----------
    image : numpy.ndarray
        Projection. Must be square and can have odd or even dimensions.
    samples_idx : tuple
        Pixel indices to use for autocorrelation estimation.
    max_d : int
        The autocorrelation is estimated for distances up to max_d. The value
        of max_d should be much less than the number of data samples, N. A 
        typical value in the literature is N/1, but this may result in a 
        spectrum that is too smooth.

    Returns
    -------
    p2 : numpy.ndarray
        2D power spectrum function. If each image is of size pxp, then P2 is 
        of size (2p-1)x(2p-1). P2 is always real.
    r : numpy.ndarray
        1D isotropic autocorrelation function.
    r2 : numpy.ndarray
        2D isotropic autocorrelation function.
    x : numpy.ndarray
        Distances at which the autocorrelction R was estimated.
    """
    p = image.shape[0]
    if max_d >= p:
        max_d = p - 1
    # Estimate the 1D isotropic autocorrelation function.
    r, x, _ = cryo_epsdr(image, samples_idx, max_d)

    # Use the 1D autocorrelation estimated above to populate an array of the 2D
    # isotropic autocorrelction.
    r2 = autocorr_2d(max_d, x, r, p)
    
    # Window the 2D autocorrelation and Fourier transform it to get the power
    # spectrum. Always use the Gaussian window, as it has positive Fourier
    # transform.     
    w = gwindow(p, max_d)
    
    p2 = np.fft.fftshift(fft2(np.fft.ifftshift(r2 * w))).real
    
    # Normalize the power spectrum P2. The power spectrum is normalized such
    # that its energy is equal to the average energy of the noise samples used
    # to estimate it.
    e = np.sum(np.square(image[samples_idx] - np.mean(image[samples_idx])))
    mean_e = e / len(samples_idx[0])
    
    # Normalize p2 such that its mean energy is preserved and is equal to
    # mean_e. That way the mean energy does not go down if the number of 
    # pixels is artificially changed (say by upsampling, downsampling, or 
    # cropping). Note that p2 is already in units of energy, and so the total
    # energy is given by sum(p2) and not by norm(p2).
    p2 = (p2 / p2.sum()) * mean_e * p2.size
    
    # Due to the truncation of the Gaussian window, we get small negative
    # values in p2, so we ignore them.
    p2 = np.where(p2 < 0, 0, p2)
    return p2

def cryo_epsdr(image, samples_idx, max_d):
    """
    Estimate the 1D isotropic autocorrelation of an image. The samples to use
    are given in samples_idx. The correlation is computed up to a maximal
    distance of max_d.

    Parameters
    ----------
    image : numpy.ndarray
        square pxp image.
    samples_idx : tuple
        pixel indices to use for autocorrelation estimation.
    max_d : int
        Correlations are computed up to a maximal distance of max_d pixels.
        Default p-1.

    Returns
    -------
    r : numpy.ndarray
        1D vector with samples of the isotropic autocorrelation function.
    x : numpy.ndarray
        Distaces at which the samples of the autocorrelation function are
        given. A vector of the same length as R.
    cnt : numpy.ndarray
        Number of autocorrelation samples available for each distance.

    """
    p = image.shape[0]
    
    # Generate all possible squared distances. For a vertical shift i and 
    # horizontal shift j, dists(i,j) contains the corresponding isotropic 
    # correlation distance i^2+j^2. dsquare is then the set of all possible 
    # squared distances, that is, all distances that can be generated by 
    # integer steps in the horizontal and vertical directions.
    i = np.array([[x for x in range(max_d + 1)] for x in range(max_d + 1)])
    dists = i ** 2 + i.transpose() ** 2
    dsquare = np.sort(np.unique(dists[np.where(dists <= max_d ** 2)]))
    x = dsquare ** 0.5 # Distances at which the correlations are computed.
    
    # Create a distance map whose value at the index (i,j) is the index in the
    # array dsquare whose value is i^2+j^2. Pairs (i,j) that correspond to
    # distances that are larger than max_d are indicated by (-1).   
    dist_map = distmap(max_d, dsquare, dists.shape)
    valid_dists = np.where(dist_map != -1)

    # Compute the number of terms in the expression sum_{j}x(j)x(j+d) for each
    # distance d. As the correlation is two-dimensioanl, we compute for each
    # sum of the form  R(k1,k2)=sum_{i,j} X_{i,j} X_{i+k1,j+k2}, how many 
    # summands are in the in it for each (k1,k2). This is done by setting the
    # participating image samples to 1 and computing autocorrelation again.
    mask = np.zeros((p, p))
    mask[samples_idx] = 1
    tmp = np.zeros((2 * p + 1, 2 * p + 1))
    tmp[:p, :p] = mask
    ftmp = fft2(tmp)
    c = ifft2(ftmp * np.conj(ftmp))
    c = c[:max_d + 1, :max_d + 1]
    c = np.round(c.real).astype('int') 

    r = np.zeros(len(dsquare)) # r(i) is the value of the ACF at distance x(i)
    
    # Compute non-periodic autocorrelation of masked image with itself (mask
    # all pixels that are not used to autocorrelation estimation).
    input_fft2 = np.zeros((2 * p + 1, 2 * p + 1), dtype='complex128')
    output_fft2 = np.zeros((2 * p + 1, 2 * p + 1), dtype='complex128')
    input_ifft2 = np.zeros((2 * p + 1, 2 * p + 1), dtype='complex128')
    output_ifft2 = np.zeros((2 * p + 1, 2 * p + 1), dtype='complex128')
    flags = ('FFTW_MEASURE', 'FFTW_UNALIGNED')
    a_fft2 = FFTW(input_fft2, output_fft2, axes=(0, 1), direction='FFTW_FORWARD', flags=flags)
    a_ifft2 = FFTW(input_ifft2, output_ifft2, axes=(0, 1), direction='FFTW_BACKWARD', flags=flags)
    sum_c = c
    
    input_fft2[samples_idx] = image[samples_idx]
    a_fft2()
    np.multiply(output_fft2, np.conj(output_fft2), out=input_ifft2)
    a_ifft2()
    sum_s = output_ifft2
    
    # Accumulate all autocorrelation values R(k1,k2) such that k1^2+k2^2=const 
    # (all autocorrelations of a certain distance).
    # corrs(i) contains the sum of all products of the form x(j)x(j+d), where
    # d=sqrt(dsquare(i)).
    # corr_count is the number of pairs x(j)x(j+d) for each d.
    corr_count, corrs = accumelate_corrs(len(dsquare), valid_dists, dist_map, sum_c, sum_s)

    # Remove zero correlation sums (distances for which we had no samples at 
    # that distance)
    idx = np.where(corr_count != 0)[0]
    r[idx] += corrs[idx] / corr_count[idx]
    cnt = corr_count[idx]
    idx = np.where(corr_count == 0)[0]
    r[idx] = 0
    x[idx] = 0
    return r, x, cnt

@jit(nopython=True)
def accumelate_corrs(dsquare_len, valid_dists, dist_map, sum_c, sum_s):
    """
    Accumulate all autocorrelation values R(k1,k2) such that k1^2+k2^2=const 
    (all autocorrelations of a certain distance).
    corrs(i) contains the sum of all products of the form x(j)x(j+d), where
    d=sqrt(dsquare(i)).
    corr_count is the number of pairs x(j)x(j+d) for each d.
    """
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
    r2 = np.zeros((int(2 * p - 1), int(2 * p - 1)), dtype=np.float64)
    for i in range(-max_d, max_d + 1):
        for j in range(-max_d, max_d + 1):
            d = i ** 2 + j ** 2
            if d <= max_d ** 2:
                idx, _ = bsearch(dsquare, d * (1 - 1e-13), d * (1 + 1e-13))
                r2[i + p - 1, j + p - 1] = r[int(idx) - 1]        
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
        (2p-1)x(2p-1) array.
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

    return lower_idx, upper_idx

def cryo_prewhiten(image, noise_response):
    """
    Pre-whiten a projection using the power spectrum of the noise.

    Parameters
    ----------
    image : numpy.ndarray
        image/projection.
    noise_response : numpy.ndarray
        2d image with the power spectrum of the noise.

    Returns
    -------
    p2 : Pre-whitened image.
    """
    delta = np.finfo(image.dtype).eps

    L1, L2 = image.shape
    l1 = L1 // 2
    l2 = L2 // 2
    K1, K2 = noise_response.shape
    k1 = int(np.ceil(K1 / 2))
    k2 = int(np.ceil(K2 / 2))

    # The whitening filter is the sqrt of of the power spectrum of the noise.
    # Also, normalize the energy of the filter to one.
    filter_var = np.sqrt(noise_response)
    filter_var /= np.linalg.norm(filter_var)

    # The filter should be circularly symmetric. In particular, it is up-down
    # and left-right symmetric. Get rid of any tiny asymmetries in the filter:
    filter_var = (filter_var + np.flipud(filter_var)) / 2
    filter_var = (filter_var + np.fliplr(filter_var)) / 2

    # The filter may have very small values or even zeros. We don't want to
    # process these so make a list of all large entries.
    nzidx = np.where(filter_var > 100 * delta)

    fnz = filter_var[nzidx]
    one_over_fnz = 1 / fnz

    # matrix with 1/fnz in nzidx, 0 elsewhere. Later we multiply the Fourier
    # transform of the padded image with this.
    one_over_fnz_as_mat = np.zeros((noise_response.shape[0], noise_response.shape[1]))
    one_over_fnz_as_mat[nzidx] += one_over_fnz
    
    # Pad the input projection.
    pp = np.zeros((noise_response.shape[0], noise_response.shape[1]))
    p2 = np.zeros((L1, L2), dtype='complex128')
    row_start_idx = k1 - l1 - 1
    row_end_idx = k1 + l1
    col_start_idx = k2 - l2 - 1
    col_end_idx = k2 + l2
    if L1 % 2 == 0 and L2 % 2 == 0:
        row_end_idx -= 1
        col_end_idx -= 1
    pp[row_start_idx:row_end_idx, col_start_idx:col_end_idx] = image.copy()
    
    # Take the Fourier transform of the padded image.
    fp = np.fft.fftshift(np.transpose(fft2(np.transpose(np.fft.ifftshift(pp)))))   
    # Divide the image by the whitening filter, but only in places where the 
    # filter is large. In frequecies where the filter is tiny  we cannot 
    # pre-whiten so we just put zero.
    fp *= one_over_fnz_as_mat
    # pp2 for padded p2.
    pp2 = np.fft.fftshift(np.transpose(ifft2(np.transpose(np.fft.ifftshift(fp)))))
    # The resulting image should be real.
    p2 = np.real(pp2[row_start_idx:row_end_idx, col_start_idx:col_end_idx]).copy()
    return p2

def cryo_prewhiten_gpu(image, noise_response):
    """
    Pre-whiten a projection using the power spectrum of the noise. We accelerate
    the Fourier transform by using GPU (CuPy).

    Parameters
    ----------
    image : numpy.ndarray
        image/projection.
    noise_response : numpy.ndarray
        2d image with the power spectrum of the noise.

    Returns
    -------
    p2 : Pre-whitened image.
    """
    delta = np.finfo(image.dtype).eps

    L1, L2 = image.shape
    l1 = L1 // 2
    l2 = L2 // 2
    K1, K2 = noise_response.shape
    k1 = int(np.ceil(K1 / 2))
    k2 = int(np.ceil(K2 / 2))

    # The whitening filter is the sqrt of of the power spectrum of the noise.
    # Also, normalize the energy of the filter to one.
    filter_var = np.sqrt(noise_response)
    filter_var /= np.linalg.norm(filter_var)
    
    # The filter should be circularly symmetric. In particular, it is up-down
    # and left-right symmetric. Get rid of any tiny asymmetries in the filter:
    filter_var = (filter_var + np.flipud(filter_var)) / 2
    filter_var = (filter_var + np.fliplr(filter_var)) / 2
    
    # The filter may have very small values or even zeros. We don't want to
    # process these so make a list of all large entries.
    nzidx = np.where(filter_var > 100 * delta)

    fnz = filter_var[nzidx]
    one_over_fnz = 1 / fnz

    # matrix with 1/fnz in nzidx, 0 elsewhere, later we multiply the Fourier
    # transform of the padded image with this.
    one_over_fnz_as_mat = np.zeros((noise_response.shape[0], noise_response.shape[1]))
    one_over_fnz_as_mat[nzidx] += one_over_fnz
    
    # Pad the input projection.
    pp = cp.zeros((noise_response.shape[0], noise_response.shape[1]))
    p2 = np.zeros((L1, L2), dtype='complex128')
    row_start_idx = k1 - l1 - 1
    row_end_idx = k1 + l1
    col_start_idx = k2 - l2 - 1
    col_end_idx = k2 + l2
    if L1 % 2 == 0 and L2 % 2 == 0:
        row_end_idx -= 1
        col_end_idx -= 1
    pp[row_start_idx:row_end_idx, col_start_idx:col_end_idx] = cp.asarray(image)
    
    # Take the Fourier transform of the padded image.
    fp = cp.fft.fftshift(cp.transpose(cp.fft.fft2(cp.transpose(cp.fft.ifftshift(pp)))))
    # Divide the image by the whitening filter, but only in places where the 
    # filter is large. In frequecies where the filter is tiny  we cannot 
    # pre-whiten so we just put zero.
    fp = fp * cp.asarray(one_over_fnz_as_mat)
    # pp2 for padded p2.
    pp2 = cp.fft.fftshift(cp.transpose(cp.fft.ifft2(cp.transpose(cp.fft.ifftshift(fp)))))
    # The resulting image should be real.
    p2 = cp.real(pp2[row_start_idx:row_end_idx, col_start_idx:col_end_idx])
    return cp.asnumpy(p2)

def als_find_min(sreal, eps, max_iter):
    """
    ALS method for RPSD factorization.

    Approximate Clean and Noise PSD and the particle location vector alpha.
    Parameters
    ----------
    sreal : numpy.ndarray
        PSD matrix to be factorized
    eps : float
        Convergence term
    max_iter : int
        Maximum iterations.
    
    Returns
    -------
    p2 : numpy.ndarray 
        Pre-whitened image.
    approx_clean_psd : numpy.ndarray 
        Approximated clean PSD
    approx_noise_psd : numpy.ndarray
        Approximated noise PSD
    alpha_approx : numpy.ndarray
        Particle location vector alpha.
    stop_par : int
        Stop algorithm if an error occurred.
    """
    sreal = sreal.transpose()
    sz = sreal.shape
    patch_num = sz[1]
    One = np.ones(patch_num)
    s_norm_inf = np.apply_along_axis(lambda x: max(np.abs(x)), 0, sreal)
    max_col = np.argmax(s_norm_inf)
    min_col = np.argmin(s_norm_inf)
    clean_sig_tmp = np.abs(sreal[:, max_col] - sreal[:, min_col])
    s_norm_1 = np.apply_along_axis(lambda x: sum(np.abs(x)), 0, sreal)
    min_col = np.argmin(s_norm_1)
    noise_sig_tmp = np.abs(sreal[:, min_col])
    s = sreal - np.outer(noise_sig_tmp, One)
    alpha_tmp = (np.dot(clean_sig_tmp, s)) / np.sum(clean_sig_tmp ** 2)
    alpha_tmp = alpha_tmp.clip(min=0, max=1)
    stop_par = 0
    cnt = 1
    while stop_par == 0:
        if not np.linalg.norm(alpha_tmp, 1):
            alpha_tmp = np.random.random(alpha_tmp.size)
        approx_clean_psd = np.dot(s, alpha_tmp) / sum(alpha_tmp ** 2)
        approx_clean_psd = approx_clean_psd.clip(min=0, max=None)
        s = sreal - np.outer(approx_clean_psd, alpha_tmp)
        approx_noise_psd = np.dot(s, np.ones(patch_num)) / patch_num
        approx_noise_psd = approx_noise_psd.clip(min=0, max=None)
        s = sreal - np.outer(approx_noise_psd, One)
        if not np.linalg.norm(approx_clean_psd, 1):
            approx_clean_psd = np.random.random(approx_clean_psd.size)
        alpha_approx = np.dot(approx_clean_psd, s) / sum(approx_clean_psd ** 2)
        alpha_approx = alpha_approx.clip(min=0, max=1)
        if np.linalg.norm(noise_sig_tmp - approx_noise_psd) / np.linalg.norm(approx_noise_psd) < eps:
            if np.linalg.norm(clean_sig_tmp - approx_clean_psd) / np.linalg.norm(approx_clean_psd) < eps:
                if np.linalg.norm(alpha_approx - alpha_tmp) / np.linalg.norm(alpha_approx) < eps:
                    break
        noise_sig_tmp = approx_noise_psd
        alpha_tmp = alpha_approx
        clean_sig_tmp = approx_clean_psd
        cnt += 1
        if cnt > max_iter:
            stop_par = 1
            break
    return approx_clean_psd, approx_noise_psd, alpha_approx, stop_par

def picking_from_scoring_mat(log_test_n, mrc_name, kltpicker, mg_big_size):
    idx_row = np.arange(log_test_n.shape[0])
    idx_col = np.arange(log_test_n.shape[1])
    [col_idx, row_idx] = np.meshgrid(idx_col, idx_row)
    r_del = np.floor(kltpicker.patch_size_pick_box)
    shape = (log_test_n.shape[1], log_test_n.shape[0])
    log_max = np.max(log_test_n)
   
    # preparing particle output files:
    box_path = kltpicker.output_particles / 'box'
    star_path = kltpicker.output_particles / 'star'
    if not kltpicker.output_particles.exists():
        kltpicker.output_particles.mkdir()
    if not box_path.exists():
        box_path.mkdir()
    if not star_path.exists():
        star_path.mkdir()
    box_file = open(box_path / mrc_name.replace('.mrc', '.box'), 'w')
    star_file = open(star_path / mrc_name.replace('.mrc', '.star'), 'w')
    star_file.write('data_\n\nloop_\n_rlnCoordinateX #1\n_rlnCoordinateY #2\n_rlnAutopickFigureOfMerit #3\n')

    # picking particles: 
    scoring_mat = log_test_n.copy()
    if kltpicker.num_of_particles == -1:  # pick all particles.
        num_picked = 0     
        p_max = kltpicker.threshold + 1
        while num_picked <= kltpicker.max_iter and p_max > kltpicker.threshold:
            max_index = np.argmax(scoring_mat.transpose().flatten())
            p_max = scoring_mat.transpose().flatten()[max_index]
            if not p_max > kltpicker.threshold:
                break
            else:
                [index_col, index_row] = np.unravel_index(max_index, shape)
                ind_row_patch = (index_row - 1) + kltpicker.patch_size_func
                ind_col_patch = (index_col - 1) + kltpicker.patch_size_func
                row_idx_b = row_idx - index_row
                col_idx_b = col_idx - index_col
                rsquare = row_idx_b ** 2 + col_idx_b ** 2
                scoring_mat[rsquare <= (r_del ** 2)] = -np.inf
                box_file.write(
                    '%i\t%i\t%i\t%i\n' % (np.round((1 / kltpicker.mgscale) * (ind_col_patch + 1 - np.floor(kltpicker.patch_size_pick_box / 2))),
                                          np.round((mg_big_size[1] + 1) - (1 / kltpicker.mgscale) * (
                                                      ind_row_patch + 1 + np.floor(kltpicker.patch_size_pick_box / 2))),
                                          np.round((1 / kltpicker.mgscale) * kltpicker.patch_size_pick_box), np.round((1 / kltpicker.mgscale) * kltpicker.patch_size_pick_box)))           
                star_file.write('%i\t%i\t%f\n' % (
                np.round((1 / kltpicker.mgscale) * (ind_col_patch + 1)), np.round((mg_big_size[1] + 1) - ((1 / kltpicker.mgscale) * (ind_row_patch + 1))),
                p_max / log_max))
                num_picked += 1
        star_file.close()
        box_file.close()
        num_picked_particles = num_picked

    else:  # pick only the number of particles specified by user.
        num_picked = 0
        p_max = kltpicker.threshold + 1
        while num_picked <= kltpicker.max_iter and num_picked <= kltpicker.num_of_particles-1 and p_max > kltpicker.threshold:
            max_index = np.argmax(scoring_mat.transpose().flatten())
            p_max = scoring_mat.transpose().flatten()[max_index]
            if not p_max > kltpicker.threshold:
                break
            else:
                [index_col, index_row] = np.unravel_index(max_index, shape)
                ind_row_patch = (index_row - 1) + kltpicker.patch_size_func
                ind_col_patch = (index_col - 1) + kltpicker.patch_size_func
                row_idx_b = row_idx - index_row
                col_idx_b = col_idx - index_col
                rsquare = row_idx_b ** 2 + col_idx_b ** 2
                scoring_mat[rsquare <= (r_del ** 2)] = -np.inf
                box_file.write(
                    '%i\t%i\t%i\t%i\n' % (np.round((1 / kltpicker.mgscale) * (ind_col_patch + 1 - np.floor(kltpicker.patch_size_pick_box / 2))),
                                          np.round((mg_big_size[1] + 1) - (1 / kltpicker.mgscale) * (
                                                      ind_row_patch + 1 + np.floor(kltpicker.patch_size_pick_box / 2))),
                                          np.round((1 / kltpicker.mgscale) * kltpicker.patch_size_pick_box), np.round((1 / kltpicker.mgscale) * kltpicker.patch_size_pick_box)))
                star_file.write('%i\t%i\t%f\n' % (
                np.round((1 / kltpicker.mgscale) * (ind_col_patch + 1)), np.round((mg_big_size[1] + 1) - ((1 / kltpicker.mgscale) * (ind_row_patch + 1))),
                p_max / log_max))
                num_picked += 1
        star_file.close()
        box_file.close()
        num_picked_particles = num_picked
    
    # pick noise:    
    if kltpicker.num_of_noise_images != 0:
        scoring_mat = log_test_n.copy()
        p_min = kltpicker.threshold - 1

        # preparing noise output files:
        box_path = kltpicker.output_noise / 'box'
        star_path = kltpicker.output_noise / 'star'
        if not kltpicker.output_noise.exists():
            kltpicker.output_noise.mkdir()
        if not box_path.exists():
            box_path.mkdir()
        if not star_path.exists():
            star_path.mkdir()
        box_file = open(box_path / mrc_name.replace('.mrc', '.box'), 'w')
        star_file = open(star_path / mrc_name.replace('.mrc', '.star'), 'w')
        star_file.write('data_\n\nloop_\n_rlnCoordinateX #1\n_rlnCoordinateY #2\n')
        
        num_picked = 0
        while num_picked <= kltpicker.num_of_noise_images-1 and p_min < kltpicker.threshold:
            min_index = np.argmin(scoring_mat.transpose().flatten())
            p_min = scoring_mat.transpose().flatten()[min_index]
            if not p_min < kltpicker.threshold:
                break
            else:
                [index_col, index_row] = np.unravel_index(min_index, shape)
                ind_row_patch = (index_row - 1) + kltpicker.patch_size_func
                ind_col_patch = (index_col - 1) + kltpicker.patch_size_func
                row_idx_b = row_idx - index_row
                col_idx_b = col_idx - index_col
                rsquare = row_idx_b ** 2 + col_idx_b ** 2
                scoring_mat[rsquare <= (r_del ** 2)] = np.inf
                box_file.write(
                    '%i\t%i\t%i\t%i\n' % (np.round((1 / kltpicker.mgscale) * (ind_col_patch + 1 - np.floor(kltpicker.patch_size_pick_box / 2))),
                                          np.round((mg_big_size[1] + 1) - (1 / kltpicker.mgscale) * (
                                                      ind_row_patch + 1 + np.floor(kltpicker.patch_size_pick_box / 2))),
                                          np.round((1 / kltpicker.mgscale) * kltpicker.patch_size_pick_box), np.round((1 / kltpicker.mgscale) * kltpicker.patch_size_pick_box)))
                star_file.write('%i\t%i\n' % (np.round((1 / kltpicker.mgscale) * (ind_col_patch + 1)), np.round((mg_big_size[1] + 1) - ((1 / kltpicker.mgscale) * (ind_row_patch + 1)))))
                num_picked += 1
        star_file.close()
        box_file.close()
        num_picked_noise = num_picked
           
    else:
        num_picked_noise = 0
            
    return num_picked_particles, num_picked_noise


