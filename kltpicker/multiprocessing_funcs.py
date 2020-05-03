from scipy.signal import fftconvolve
import numpy as np
from .cryo_utils import cryo_epsds
from .util import radial_avg, trig_interpolation
from scipy.linalg import eigh

def estimate_rpsd_pool(k, m, noise_mc, patch_size, L, rho_samp, num_quads, idx, nodes):
    row = np.ceil((k + 1) / m).astype(int)
    col = (k + 1 - (row - 1) * m).astype(int)
    noisemc_block = noise_mc[(row - 1) * patch_size.astype(int):row * patch_size.astype(int),
                    (col - 1) * patch_size.astype(int): col * patch_size.astype(int)]
    noisemc_block = noisemc_block - np.mean(noisemc_block)
    psd_block = cryo_epsds(noisemc_block[:, :, np.newaxis],
                           np.where(np.zeros((int(patch_size), int(patch_size))) == 0),
                           np.floor(0.3 * patch_size).astype(int))
    psd_block = psd_block[0]
    [r_block, r] = radial_avg(psd_block, L)
    block_var = np.var(noisemc_block, ddof=1)
    psd_rad = np.abs(trig_interpolation(r * np.pi, r_block, rho_samp))
    psd_mat = np.reshape(psd_rad[idx], [num_quads, num_quads])
    var_psd = (1 / (2 * np.pi) ** 2) * np.linalg.multi_dot([nodes, psd_mat, nodes.transpose()])
    scaling_psd = block_var / var_psd
    r_block = scaling_psd * r_block
    return r_block

def construct_klt_templates_pool(sqrt_rr, j_r_rho_n, d_rho_psd_quad_ker, sqrt_diag_quad_nys):
    h_nodes = sqrt_rr * np.linalg.multi_dot([j_r_rho_n, d_rho_psd_quad_ker, j_r_rho_n.transpose()])
    tmp = np.linalg.multi_dot([sqrt_diag_quad_nys, h_nodes, sqrt_diag_quad_nys.transpose()])
    eig_vals, eig_funcs = eigh(tmp)
    eig_vals = np.real(eig_vals)
    eig_vals = eig_vals[::-1]  # Descending.
    eig_funcs = eig_funcs[:, ::-1]
    eig_vals = np.where(np.abs(eig_vals) < np.spacing(1), 0, eig_vals)
    eig_funcs[:, eig_vals == 0] = 0
    return eig_funcs, eig_vals

def mp_v(tup):
    qi, patch_size_func, noise_mc = tup
    q_tmp = np.reshape(qi, (patch_size_func, patch_size_func)).transpose()
    q_tmp = q_tmp - np.mean(q_tmp)
    q_tmp = np.flip(q_tmp, 1)
    v_tmp = fftconvolve(noise_mc, q_tmp, 'valid')
    return v_tmp

def detect_particles_part2(vj, t_mat):
    v = np.sum((vj @ t_mat) * vj, 1)
    return v