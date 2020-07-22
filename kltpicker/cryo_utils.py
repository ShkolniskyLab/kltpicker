import numpy as np
import cupy as cp
from pyfftw.interfaces.numpy_fft import fft2, ifft2
from pyfftw import FFTW
from numpy.polynomial.legendre import leggauss
import operator as op
from numba import jit


def crop(x, out_shape):
    """

    :param x: ndarray of size (N_1,...N_k)
    :param out_shape: iterable of integers of length k. The value in position i is the size we want to cut from the
        center of x in dimension i. If the value is <= 0 then the dimension is left as is
    :return: out: The center of x with size outshape.
    """
    in_shape = np.array(x.shape)
    out_shape = np.array([s if s > 0 else in_shape[i] for i, s in enumerate(out_shape)])
    start_indices = in_shape // 2 - out_shape // 2
    end_indices = start_indices + out_shape
    indexer = tuple([slice(i, j) for (i, j) in zip(start_indices, end_indices)])
    out = x[indexer]
    return out


def downsample(stack, n, mask=None, stack_in_fourier=False):
    """ Use Fourier methods to change the sample interval and/or aspect ratio
        of any dimensions of the input image 'img'. If the optional argument
        stack is set to True, then the *first* dimension of 'img' is interpreted as the index of
        each image in the stack. The size argument side is an integer, the size of the
        output images.  Let the size of a stack
        of 2D images 'img' be n1 x n1 x k.  The size of the output will be side x side x k.

        If the optional mask argument is given, this is used as the
        zero-centered Fourier mask for the re-sampling. The size of mask should
        be the same as the output image size. For example for downsampling an
        n0 x n0 image with a 0.9 x nyquist filter, do the following:
        msk = fuzzymask(n,2,.45*n,.05*n)
        out = downsample(img, n, 0, msk)
        The size of the mask must be the size of output. The optional fx output
        argument is the padded or cropped, masked, FT of in, with zero
        frequency at the origin.
    """

    size_in = np.square(stack.shape[1])
    size_out = np.square(n)
    mask = 1 if mask is None else mask
    num_images = stack.shape[0]
    output = np.zeros((num_images, n, n), dtype='float64')
    images_batches = np.array_split(np.arange(num_images), 500)
    for batch in images_batches:
        if batch.size:
            curr_batch = np.array(stack[batch])
            curr_batch = curr_batch if stack_in_fourier else fft2(curr_batch)
            fx = crop(np.fft.fftshift(curr_batch, axes=(-2, -1)), (-1, n, n)) * mask
            output[batch] = ifft2(np.fft.ifftshift(fx, axes=(-2, -1))) * (size_out / size_in)
    return output


def cfft2(x, axes=(-1, -2)):
    if len(x.shape) == 2:
        return np.fft.fftshift(np.transpose(np.fft.fft2(np.transpose(np.fft.ifftshift(x)))))
    elif len(x.shape) == 3:
        y = np.fft.ifftshift(x, axes=axes)
        y = np.fft.ifft2(y, axes=axes)
        y = np.fft.fftshift(y, axes=axes)
        return y
    else:
        raise ValueError("x must be 2D or 3D")


def icfft2(x, axes=(-1, -2)):
    if len(x.shape) == 2:
        return np.fft.fftshift(np.transpose(np.fft.ifft2(np.transpose(np.fft.ifftshift(x)))))
    elif len(x.shape) == 3:
        y = np.fft.ifftshift(x, axes=axes)
        y = ifft2(y, axes=axes)
        y = np.fft.fftshift(y, axes=axes)
        return y
    else:
        raise ValueError("x must be 2D or 3D")


def icfft(x, axis=0):
    return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(x, axis), axis=axis), axis)


def fast_cfft2(x, axes=(-1, -2)):
    if len(x.shape) == 2:
        return np.fft.fftshift(np.transpose(fft2(np.transpose(np.fft.ifftshift(x)))))
    elif len(x.shape) == 3:
        y = np.fft.ifftshift(x, axes=axes)
        y = fft2(y, axes=axes)
        y = np.fft.fftshift(y, axes=axes)
        return y
    else:
        raise ValueError("x must be 2D or 3D")


def fast_cfft2_cp(x, axes=(-1, -2)):
    if len(x.shape) == 2:
        return cp.fft.fftshift(cp.transpose(cp.fft.fft2(cp.transpose(cp.fft.ifftshift(x)))))
    elif len(x.shape) == 3:
        y = cp.fft.ifftshift(x, axes=axes)
        y = cp.fft.fft2(y, axes=axes)
        y = cp.fft.fftshift(y, axes=axes)
        return y
    else:
        raise ValueError("x must be 2D or 3D")


def fast_icfft2(x, axes=(-1, -2)):
    if len(x.shape) == 2:
        return np.fft.fftshift(np.transpose(ifft2(np.transpose(np.fft.ifftshift(x)))))

    elif len(x.shape) == 3:
        y = np.fft.ifftshift(x, axes=axes)
        y = ifft2(y, axes=axes)
        y = np.fft.fftshift(y, axes=axes)
        return y

    else:
        raise ValueError("x must be 2D or 3D")

def fast_icfft2_cp(x, axes=(-1, -2)):
    if len(x.shape) == 2:
        return cp.fft.fftshift(cp.transpose(cp.fft.ifft2(cp.transpose(cp.fft.ifftshift(x)))))

    elif len(x.shape) == 3:
        y = cp.fft.ifftshift(x, axes=axes)
        y = cp.fft.ifft2(y, axes=axes)
        y = cp.fft.fftshift(y, axes=axes)
        return y

    else:
        raise ValueError("x must be 2D or 3D")


def lgwt(n, a, b):
    """
    Get n leggauss points in interval [a, b]

    :param n: number of points
    :param a: interval starting point
    :param b: interval end point
    :returns SamplePoints(x, w): sample points, weight
    """

    x1, w = leggauss(n)
    m = (b - a) / 2
    c = (a + b) / 2
    x = m * x1 + c
    w = m * w
    x = np.flipud(x)
    return x, w


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

def cryo_epsds_cp(imstack, samples_idx, max_d):
    p = imstack.shape[0]
    if max_d >= p:
        max_d = p - 1

    r, x, _ = cryo_epsdr_cp(imstack, samples_idx, max_d)

    r2 = cp.zeros((2 * p - 1, 2 * p - 1))
    dsquare = cp.square(x)
    for i in range(-max_d, max_d + 1):
        for j in range(-max_d, max_d + 1):
            d = i ** 2 + j ** 2
            if d <= max_d ** 2:
                idx, _ = bsearch_cp(dsquare, d * (1 - 1e-13), d * (1 + 1e-13))
                r2[i + p - 1, j + p - 1] = r[idx - 1]

    w = gwindow_cp(p, max_d)
    p2 = fast_cfft2_cp(r2 * w)

    p2 = p2.real

    e = 0
    for i in range(imstack.shape[2]):
        im = imstack[:, :, i]
        e += cp.sum(cp.square(im[samples_idx] - cp.mean(im[samples_idx])))

    mean_e = e / (len(samples_idx[0]) * imstack.shape[2])
    p2 = (p2 / p2.sum()) * mean_e * p2.size
    neg_idx = cp.where(p2 < 0)
    p2[neg_idx] = 0
    return p2, r, r2, x

def cryo_epsdr(vol, samples_idx, max_d):
    p = vol.shape[0]
    k = vol.shape[2]
    i, j = np.meshgrid(np.arange(max_d + 1), np.arange(max_d + 1))
    dists = np.square(i) + np.square(j)
    dsquare = np.sort(np.unique(dists[np.where(dists <= max_d ** 2)]))

    corrs = np.zeros(len(dsquare))
    corr_count = np.zeros(len(dsquare))
    x = np.sqrt(dsquare)

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

def cryo_epsdr_cp(vol, samples_idx, max_d):
    p = vol.shape[0]
    k = vol.shape[2]
    i, j = cp.meshgrid(cp.arange(max_d + 1), cp.arange(max_d + 1))
    dists = cp.square(i) + cp.square(j)
    dsquare = cp.sort(cp.unique(dists[cp.where(dists <= max_d ** 2)]))

    corrs = cp.zeros(len(dsquare))
    corr_count = cp.zeros(len(dsquare))
    x = cp.sqrt(dsquare)

    dist_map = cp.zeros(dists.shape)
    for i in range(max_d + 1):
        for j in range(max_d + 1):
            d = i ** 2 + j ** 2
            if d <= max_d ** 2:
                idx, _ = bsearch_cp(dsquare, d - 1e-13, d + 1e-13)
                dist_map[i, j] = idx

    dist_map = dist_map.astype('int') - 1
    valid_dists = cp.where(dist_map != -1)

    mask = cp.zeros((p, p))
    mask[samples_idx] = 1
    tmp = cp.zeros((2 * p + 1, 2 * p + 1))
    tmp[:p, :p] = mask
    ftmp = cp.fft.fft2(tmp)
    c = cp.fft.ifft2(ftmp * cp.conj(ftmp))
    c = c[:max_d + 1, :max_d + 1]
    c = cp.around(c.real).astype('int')

    r = cp.zeros(len(corrs))

    vol = vol.transpose((2, 0, 1)).copy()
    input_fft2 = cp.zeros((2 * p + 1, 2 * p + 1), dtype='complex128')
    output_fft2 = cp.zeros((2 * p + 1, 2 * p + 1), dtype='complex128')
    input_ifft2 = cp.zeros((2 * p + 1, 2 * p + 1), dtype='complex128')
    output_ifft2 = cp.zeros((2 * p + 1, 2 * p + 1), dtype='complex128')
    
    sum_s = cp.zeros(output_ifft2.shape, output_ifft2.dtype)
    sum_c = c * vol.shape[0]
    for i in range(k):
        proj = vol[i]

        input_fft2[samples_idx] = proj[samples_idx]
        output_fft2 = cp.fft.fft2(input_fft2)
        cp.multiply(output_fft2, cp.conj(output_fft2), out=input_ifft2)
        output_ifft2 = cp.fft.ifft2(input_ifft2)
        sum_s += output_ifft2

    for curr_dist in zip(valid_dists[0], valid_dists[1]):
        dmidx = dist_map[curr_dist]
        corrs[dmidx] += sum_s[curr_dist].real
        corr_count[dmidx] += sum_c[curr_dist]

    idx = cp.where(corr_count != 0)[0]
    r[idx] += corrs[idx] / corr_count[idx]
    cnt = corr_count[idx]

    idx = cp.where(corr_count == 0)[0]
    r[idx] = 0
    x[idx] = 0
    return r, x, cnt


def gwindow(p, max_d):
    x, y = np.meshgrid(np.arange(-(p - 1), p), np.arange(-(p - 1), p))
    alpha = 3.0
    w = np.exp(-alpha * (np.square(x) + np.square(y)) / (2 * max_d ** 2))
    return w

def gwindow_cp(p, max_d):
    x, y = cp.meshgrid(cp.arange(-(p - 1), p), cp.arange(-(p - 1), p))
    alpha = 3.0
    w = cp.exp(-alpha * (cp.square(x) + cp.square(y)) / (2 * max_d ** 2))
    return w


@jit(nopython=True)
def bsearch(x, lower_bound, upper_bound):
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

def bsearch_cp(x, lower_bound, upper_bound):
    if lower_bound > x[-1] or upper_bound < x[0] or upper_bound < lower_bound:
        return None, None
    lower_idx_a = 1
    lower_idx_b = len(x)
    upper_idx_a = 1
    upper_idx_b = len(x)

    while lower_idx_a + 1 < lower_idx_b or upper_idx_a + 1 < upper_idx_b:
        lw = int(cp.floor((lower_idx_a + lower_idx_b) / 2))
        if x[lw - 1] >= lower_bound:
            lower_idx_b = lw
        else:
            lower_idx_a = lw
            if upper_idx_a < lw < upper_idx_b:
                upper_idx_a = lw

        up = int(cp.ceil((upper_idx_a + upper_idx_b) / 2))
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


def cryo_prewhiten(proj, noise_response, rel_threshold=None):
    """
    Pre-whiten a stack of projections using the power spectrum of the noise.


    :param proj: stack of images/projections
    :param noise_response: 2d image with the power spectrum of the noise. If all
                           images are to be whitened with respect to the same power spectrum,
                           this is a single image. If each image is to be whitened with respect
                           to a different power spectrum, this is a three-dimensional array with
                           the same number of 2d slices as the stack of images.

    :param rel_threshold: The relative threshold used to determine which frequencies
                          to whiten and which to set to zero. If empty (the default)
                          all filter values less than 100*eps(class(proj)) are
                          zeroed out, while otherwise, all filter values less than
                          threshold times the maximum filter value for each filter
                          is set to zero.

    :return: Pre-whitened stack of images.
    """

    delta = np.finfo(proj.dtype).eps

    L1, L2, num_images = proj.shape
    l1 = L1 // 2
    l2 = L2 // 2
    K1, K2 = noise_response.shape
    k1 = int(np.ceil(K1 / 2))
    k2 = int(np.ceil(K2 / 2))

    filter_var = np.sqrt(noise_response)
    filter_var /= np.linalg.norm(filter_var)

    filter_var = (filter_var + np.flipud(filter_var)) / 2
    filter_var = (filter_var + np.fliplr(filter_var)) / 2

    if rel_threshold is None:
        nzidx = np.where(filter_var > 100 * delta)
    else:
        raise NotImplementedError('not implemented for rel_threshold != None')

    fnz = filter_var[nzidx]
    one_over_fnz = 1 / fnz

    # matrix with 1/fnz in nzidx, 0 elsewhere
    one_over_fnz_as_mat = np.zeros((noise_response.shape[0], noise_response.shape[1]))
    one_over_fnz_as_mat[nzidx] += one_over_fnz
    pp = np.zeros((noise_response.shape[0], noise_response.shape[1]))
    p2 = np.zeros((num_images, L1, L2), dtype='complex128')
    proj = proj.transpose((2, 0, 1)).copy()

    row_start_idx = k1 - l1 - 1
    row_end_idx = k1 + l1
    col_start_idx = k2 - l2 - 1
    col_end_idx = k2 + l2

    if L1 % 2 == 0 and L2 % 2 == 0:
        row_end_idx -= 1
        col_end_idx -= 1

    for i in range(num_images):
        pp[row_start_idx:row_end_idx, col_start_idx:col_end_idx] = proj[i]
        fp = fast_cfft2(pp)
        fp *= one_over_fnz_as_mat
        pp2 = fast_icfft2(fp)
        p2[i] = np.real(pp2[row_start_idx:row_end_idx, col_start_idx:col_end_idx])

    # change back to x,y,z convention
    proj = p2.real.transpose((1, 2, 0)).copy()
    return proj, filter_var, nzidx


def als_find_min(sreal, eps, max_iter):
    """
    ALS method for RPSD factorization.

    Approximate Clean and Noise PSD and the particle location vector alpha.
    :param sreal: PSD matrix to be factorized
    :param eps: Convergence term
    :param max_iter: Maximum iterations
    :return approx_clean_psd: Approximated clean PSD
    :return approx_noise_psd: Approximated noise PSD
    :return alpha_approx: Particle location vector alpha.
    :return stop_par: Stop algorithm if an error occurred.
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
    shape = log_test_n.shape
    scoring_mat = log_test_n
    if kltpicker.num_of_particles == -1:
        num_picked_particles = write_output_files(scoring_mat, shape, r_del, np.iinfo(np.int32(10)).max, op.gt,
                                                  kltpicker.threshold + 1, kltpicker.threshold,
                                                  kltpicker.patch_size_func,
                                                  row_idx, col_idx, kltpicker.output_particles, mrc_name,
                                                  kltpicker.mgscale, mg_big_size, -np.inf,
                                                  kltpicker.patch_size_pick_box)
    else:
        num_picked_particles = write_output_files(scoring_mat, shape, r_del, kltpicker.num_of_particles, op.gt,
                                                  kltpicker.threshold + 1, kltpicker.threshold,
                                                  kltpicker.patch_size_func,
                                                  row_idx, col_idx, kltpicker.output_particles, mrc_name,
                                                  kltpicker.mgscale, mg_big_size, -np.inf,
                                                  kltpicker.patch_size_pick_box)
    if kltpicker.num_of_noise_images != 0:
        num_picked_noise = write_output_files(scoring_mat, shape, r_del, kltpicker.num_of_noise_images, op.lt,
                                              kltpicker.threshold - 1, kltpicker.threshold, kltpicker.patch_size_func,
                                              row_idx, col_idx, kltpicker.output_noise, mrc_name,
                                              kltpicker.mgscale, mg_big_size, np.inf, kltpicker.patch_size_pick_box)
    else:
        num_picked_noise = 0
    return num_picked_particles, num_picked_noise


def write_output_files(scoring_mat, shape, r_del, max_iter, oper, oper_param, threshold, patch_size_func, row_idx,
                       col_idx, output_path, mrc_name, mgscale, mg_big_size, replace_param, patch_size_pick_box):
    num_picked = 0
    box_path = output_path / 'box'
    star_path = output_path / 'star'
    if not output_path.exists():
        output_path.mkdir()
    if not box_path.exists():
        box_path.mkdir()
    if not star_path.exists():
        star_path.mkdir()
    box_file = open(box_path / mrc_name.replace('.mrc', '.box'), 'w')
    star_file = open(star_path / mrc_name.replace('.mrc', '.star'), 'w')
    star_file.write('data_\n\nloop_\n_rlnCoordinateX #1\n_rlnCoordinateY #2\n')
    iter_pick = 0
    log_max = np.max(scoring_mat)
    while iter_pick <= max_iter and oper(oper_param, threshold):
        max_index = np.argmax(scoring_mat.transpose().flatten())
        oper_param = scoring_mat.transpose().flatten()[max_index]
        if not oper(oper_param, threshold):
            break
        else:
            [index_col, index_row] = np.unravel_index(max_index, shape)
            ind_row_patch = (index_row - 1) + patch_size_func
            ind_col_patch = (index_col - 1) + patch_size_func
            row_idx_b = row_idx - index_row
            col_idx_b = col_idx - index_col
            rsquare = row_idx_b ** 2 + col_idx_b ** 2
            scoring_mat[rsquare <= (r_del ** 2)] = replace_param
            box_file.write(
                '%i\t%i\t%i\t%i\n' % ((1 / mgscale) * (ind_col_patch + 1 - np.floor(patch_size_pick_box / 2)),
                                      (mg_big_size[0] + 1) - (1 / mgscale) * (
                                                  ind_row_patch + 1 + np.floor(patch_size_pick_box / 2)),
                                      (1 / mgscale) * patch_size_pick_box, (1 / mgscale) * patch_size_pick_box))
            star_file.write('%i\t%i\t%f\n' % (
            (1 / mgscale) * (ind_col_patch + 1), (mg_big_size[0] + 1) - ((1 / mgscale) * (ind_row_patch + 1)),
            oper_param / log_max))
            iter_pick += 1
            num_picked += 1
    star_file.close()
    box_file.close()
    return num_picked

