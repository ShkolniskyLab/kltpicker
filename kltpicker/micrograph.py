import numpy as np
import cupy as cp
from .util import f_trans_2, stdfilter, trig_interpolation, trig_interpolation_gpu, radial_avg, radial_avg_gpu, fftcorrelate
from scipy import signal
from .cryo_utils import lgwt, cryo_epsds, cryo_epsds_gpu, cryo_prewhiten, picking_from_scoring_mat, als_find_min
from numpy.linalg import eigh

# Globals:
EPS = 10 ** (-2)  # Convergence term for ALS.
PERCENT_EIG_FUNC = 0.99
NUM_QUAD_NYS = 2 ** 7
NUM_QUAD_KER = 2 ** 7
MAX_FUN = 400


class Micrograph:
    """
    Object that contains all the variables and methods needed for the particle picking.

    ...
    Attributes
    ----------
    micrograph : np.ndarray
        Micrograph after downsampling.
    micrograph_pic : np.ndarray
        Original micrograph data.
    mc_size : tuple
        Size of micrograph after downsampling.
    mg_big_size : tuple
        Size of original micrograph.
    noise_mc : np.ndarray
    approx_clean_psd : np.ndarray
        Approximated clean PSD.
    approx_noise_psd : np.ndarray
        Approximated noise PSD.
    approx_noise_var : float
        Approximate noise variance.
    r : np.ndarray
    stop_par : int
        Flag to stop algorithm if maximal number of iterations in PSD approximation was exceeded.
    psd : np.ndarray
    eig_func : np.ndarray
        Eigenfunctions.
    eig_val : np.ndarray
        Eigenvalues.
    num_of_func : int
        Number of eigenfunctions.
    mrc_name : str
        Name of .mrc file.

    Methods
    -------
    cutoff_filter(self, patch_size)
        Radial bandpass filter.
    estimate_rpsd(self, patch_size, max_iter)
    prewhiten_micrograph(self)
    construct_klt_templates(self, kltpicker)
        Constructing the KLTpicker templates as the eigenfunctions of a given kernel.
    detect_particles(self, kltpicker)
        Construct the scoring matrix and then use the picking_from_scoring_mat function to pick particles and noise
        images.
    picking_from_scoring_mat(log_test_n, mrc_name, kltpicker, mg_big_size)
        Pick particles and noise images from the scoring matrix.
    """

    def __init__(self, micrograph, mc_size, mrc_name, mg_big_size):
        self.micrograph = micrograph
        self.mc_size = mc_size
        self.mg_big_size = mg_big_size
        self.noise_mc = 0
        self.approx_clean_psd = 0
        self.approx_noise_psd = 0
        self.approx_noise_var = 0
        self.r = 0
        self.stop_par = 0
        self.psd = 0
        self.eig_func = 0
        self.eig_val = 0
        self.num_of_func = 0
        self.mrc_name = mrc_name

    def cutoff_filter(self, patch_size):
        """Radial bandpass filter."""
        bandpass1d = signal.firwin(int(patch_size), np.array([0.05, 0.95]), pass_zero=False)
        bandpass2d = f_trans_2(bandpass1d)
        micrograph = fftcorrelate(self.micrograph, bandpass2d)
        self.noise_mc = micrograph

    def estimate_rpsd(self, patch_size, max_iter):
        """Approximate clean and noise RPSD per micrograph."""
        micro_size = self.noise_mc.shape[0]
        m = np.floor(micro_size / patch_size)
        M = (m ** 2).astype(int)
        L = int(patch_size)
        s = np.zeros((M, L))
        num_quads = 2 ** 7
        quad, nodes = lgwt(num_quads, -np.pi, np.pi)
        x = np.tile(quad, (num_quads, 1))
        y = x.transpose()
        rho_mat = np.sqrt(x ** 2 + y ** 2)
        rho_mat = np.where(rho_mat > np.pi, 0, rho_mat)
        rho_samp, idx = np.unique(rho_mat, return_inverse=True)
        for k in range(M):
            row = np.ceil((k + 1) / m).astype(int)
            col = (k + 1 - (row - 1) * m).astype(int)
            noisemc_block = self.noise_mc[(row - 1) * patch_size.astype(int):row * patch_size.astype(int),
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
            s[k, :] = r_block
        noisemc_block = self.noise_mc[0: patch_size.astype(int), patch_size.astype(int): 2 * patch_size.astype(int)]
        noisemc_block = noisemc_block - np.mean(noisemc_block)
        psd_block = cryo_epsds(noisemc_block[:, :, np.newaxis],
                               np.where(np.zeros((int(patch_size), int(patch_size))) == 0),
                               np.floor(0.3 * patch_size).astype(int))
        psd_block = psd_block[0]
        _, r = radial_avg(psd_block, L)

        # find min arg using ALS:
        approx_clean_psd, approx_noise_psd, alpha, stop_par = als_find_min(s, EPS, max_iter)
        std_mat = stdfilter(self.noise_mc, patch_size)
        var_mat = std_mat ** 2
        cut = int((patch_size - 1) / 2 + 1)
        var_mat = var_mat[cut - 1:-cut, cut - 1:-cut]
        var_vec = var_mat.transpose().flatten()
        var_vec.sort()
        j = np.floor(0.25 * var_vec.size).astype('int')
        noise_var_approx = np.mean(var_vec[0:j])
        num_of_quad = 2 ** 7
        quad, nodes = lgwt(num_of_quad, -np.pi, np.pi)
        y = np.tile(quad, (num_of_quad, 1))
        x = y.transpose()
        rho_mat = np.sqrt(x ** 2 + y ** 2)
        rho_mat = np.where(rho_mat > np.pi, 0, rho_mat)
        rho_samp, idx = np.unique(rho_mat, return_inverse=True)
        clean_psd_nodes = np.abs(trig_interpolation(r * np.pi, approx_clean_psd, rho_samp))
        noise_psd_nodes = np.abs(trig_interpolation(r * np.pi, approx_noise_psd, rho_samp))
        clean_psd_mat = np.reshape(clean_psd_nodes[idx], (num_of_quad, num_of_quad))
        noise_psd_mat = np.reshape(noise_psd_nodes[idx], (num_of_quad, num_of_quad))
        scaling_psd_approx = (np.linalg.multi_dot([nodes, noise_psd_mat, nodes.transpose()]) - (
                4 * np.pi ** 2) * noise_var_approx) / np.linalg.multi_dot([nodes, clean_psd_mat, nodes.transpose()])
        noise_psd_approx_sigma = approx_noise_psd - scaling_psd_approx * approx_clean_psd
        noise_psd_approx_sigma = noise_psd_approx_sigma.clip(min=0, max=None)
        s_mean = np.mean(s, 0)
        s_mean_psd_nodes = np.abs(trig_interpolation(r * np.pi, s_mean, rho_samp))
        s_mean_psd_mat = np.reshape(s_mean_psd_nodes[idx], (num_of_quad, num_of_quad))
        s_mean_var_psd = (1 / (2 * np.pi) ** 2) * np.linalg.multi_dot([nodes, s_mean_psd_mat, nodes.transpose()])
        clean_var_psd = (1 / (2 * np.pi) ** 2) * np.linalg.multi_dot([nodes, clean_psd_mat, nodes.transpose()])
        clean_var = s_mean_var_psd - noise_var_approx
        approx_scaling = clean_var / clean_var_psd
        self.approx_clean_psd = approx_scaling * approx_clean_psd
        self.approx_noise_psd = noise_psd_approx_sigma
        self.approx_noise_var = noise_var_approx
        self.r = r
        self.stop_par = stop_par

    def estimate_rpsd_gpu(self, patch_size, max_iter):
        """Approximate clean and noise RPSD per micrograph, using CUDA."""
        micro_size = self.noise_mc.shape[0]
        m = np.floor(micro_size / patch_size)
        M = (m ** 2).astype(int)
        L = int(patch_size)
        s = cp.zeros((M, L))
        num_quads = 2 ** 7
        quad, nodes = lgwt(num_quads, -np.pi, np.pi)
        quad = cp.array(quad)
        nodes = cp.array(nodes)
        x = cp.tile(cp.array(quad), (num_quads, 1))
        y = x.transpose()
        rho_mat = cp.sqrt(x ** 2 + y ** 2)
        rho_mat = cp.where(rho_mat > cp.pi, 0, rho_mat)
        rho_samp, idx = cp.unique(rho_mat, return_inverse=True)
        noise_mc = cp.array(self.noise_mc)
        for k in range(M):
            print(k)
            row = cp.ceil((k + 1) / m).astype(int)
            col = (k + 1 - (row - 1) * m).astype(int)
            noisemc_block = noise_mc[(row - 1) * patch_size.astype(int):row * patch_size.astype(int),
                            (col - 1) * patch_size.astype(int): col * patch_size.astype(int)]
            noisemc_block = noisemc_block - cp.mean(noisemc_block)
            psd_block = cryo_epsds_gpu(cp.expand_dims(noisemc_block, 2),
                                   cp.where(cp.zeros((int(patch_size), int(patch_size))) == 0),
                                   int(cp.floor(0.3 * patch_size)))
            psd_block = psd_block[0]
            [r_block, r] = radial_avg_gpu(psd_block, L)
            block_var = cp.var(noisemc_block, ddof=1)
            psd_rad = cp.abs(trig_interpolation_gpu(r * cp.pi, r_block, rho_samp))
            psd_mat = cp.reshape(psd_rad[idx], [num_quads, num_quads])
            var_psd = (1 / (2 * cp.pi) ** 2) * cp.dot(nodes, cp.dot(psd_mat, nodes.transpose()))
            scaling_psd = block_var / var_psd
            r_block = scaling_psd * r_block
            s[k, :] = r_block
        noisemc_block = noise_mc[0: patch_size.astype(int), patch_size.astype(int): 2 * patch_size.astype(int)]
        noisemc_block = noisemc_block - cp.mean(noisemc_block)
        psd_block = cryo_epsds_gpu(cp.expand_dims(noisemc_block, 2),
                               cp.where(cp.zeros((int(patch_size), int(patch_size))) == 0),
                               cp.floor(0.3 * patch_size).astype(int))
        psd_block = psd_block[0]
        _, r = radial_avg_gpu(psd_block, L)
        
        s = cp.asnumpy(s)

        # find min arg using ALS:
        approx_clean_psd, approx_noise_psd, alpha, stop_par = als_find_min(s, EPS, max_iter)
        std_mat = stdfilter(self.noise_mc, patch_size)
        var_mat = std_mat ** 2
        cut = int((patch_size - 1) / 2 + 1)
        var_mat = var_mat[cut - 1:-cut, cut - 1:-cut]
        var_vec = var_mat.transpose().flatten()
        var_vec.sort()
        j = np.floor(0.25 * var_vec.size).astype('int')
        noise_var_approx = np.mean(var_vec[0:j])
        num_of_quad = 2 ** 7
        quad, nodes = lgwt(num_of_quad, -np.pi, np.pi)
        y = np.tile(quad, (num_of_quad, 1))
        x = y.transpose()
        rho_mat = np.sqrt(x ** 2 + y ** 2)
        rho_mat = np.where(rho_mat > np.pi, 0, rho_mat)
        rho_samp, idx = np.unique(rho_mat, return_inverse=True)
        clean_psd_nodes = np.abs(trig_interpolation(r * np.pi, approx_clean_psd, rho_samp))
        noise_psd_nodes = np.abs(trig_interpolation(r * np.pi, approx_noise_psd, rho_samp))
        clean_psd_mat = np.reshape(clean_psd_nodes[idx], (num_of_quad, num_of_quad))
        noise_psd_mat = np.reshape(noise_psd_nodes[idx], (num_of_quad, num_of_quad))
        scaling_psd_approx = (np.linalg.multi_dot([nodes, noise_psd_mat, nodes.transpose()]) - (
                4 * np.pi ** 2) * noise_var_approx) / np.linalg.multi_dot([nodes, clean_psd_mat, nodes.transpose()])
        noise_psd_approx_sigma = approx_noise_psd - scaling_psd_approx * approx_clean_psd
        noise_psd_approx_sigma = noise_psd_approx_sigma.clip(min=0, max=None)
        s_mean = np.mean(s, 0)
        s_mean_psd_nodes = np.abs(trig_interpolation(r * np.pi, s_mean, rho_samp))
        s_mean_psd_mat = np.reshape(s_mean_psd_nodes[idx], (num_of_quad, num_of_quad))
        s_mean_var_psd = (1 / (2 * np.pi) ** 2) * np.linalg.multi_dot([nodes, s_mean_psd_mat, nodes.transpose()])
        clean_var_psd = (1 / (2 * np.pi) ** 2) * np.linalg.multi_dot([nodes, clean_psd_mat, nodes.transpose()])
        clean_var = s_mean_var_psd - noise_var_approx
        approx_scaling = clean_var / clean_var_psd
        self.approx_clean_psd = approx_scaling * approx_clean_psd
        self.approx_noise_psd = noise_psd_approx_sigma
        self.approx_noise_var = noise_var_approx
        self.r = r
        self.stop_par = stop_par


    def prewhiten_micrograph(self):
        r = np.floor((self.mc_size[1] - 1) / 2).astype('int')
        c = np.floor((self.micrograph.shape[0] - 1) / 2).astype('int')
        col = np.arange(-c, c + 1) * np.pi / c
        row = np.arange(-r, r + 1) * np.pi / r
        Row, Col = np.meshgrid(row, col)
        rad_mat = np.sqrt(Col ** 2 + Row ** 2)
        rad_samp, idx = np.unique(rad_mat, return_inverse=True)
        rad_samp_tmp = rad_samp[rad_samp < np.max(self.r * np.pi)]
        noise_psd_nodes = np.abs(trig_interpolation(self.r * np.pi, self.approx_noise_psd, rad_samp_tmp))
        noise_psd_nodes = np.pad(noise_psd_nodes, (0, rad_samp.size - noise_psd_nodes.size), 'constant',
                                 constant_values=noise_psd_nodes[-1])
        noise_psd_mat = np.reshape(noise_psd_nodes[idx], [col.size, row.size])
        noise_mc_prewhite = cryo_prewhiten(self.noise_mc[:, :, np.newaxis], noise_psd_mat)
        noise_mc_prewhite = noise_mc_prewhite[0][:, :, 0]
        noise_mc_prewhite = noise_mc_prewhite - np.mean(noise_mc_prewhite)
        noise_mc_prewhite = noise_mc_prewhite / np.linalg.norm(noise_mc_prewhite, 'fro')
        self.noise_mc = noise_mc_prewhite

    def construct_klt_templates(self, kltpicker):
        """Constructing the KLTpicker templates as the eigenfunctions of a given kernel."""
        eig_func_tot = np.zeros((kltpicker.max_order, NUM_QUAD_NYS, NUM_QUAD_NYS))
        eig_val_tot = np.zeros((kltpicker.max_order, NUM_QUAD_NYS))
        sqrt_rr = np.sqrt(kltpicker.r_r)
        d_rho_psd_quad_ker = np.diag(kltpicker.rho) * np.diag(self.psd) * np.diag(kltpicker.quad_ker)
        sqrt_diag_quad_nys = np.sqrt(np.diag(kltpicker.quad_nys))
        for n in range(kltpicker.max_order):
            h_nodes = sqrt_rr * np.linalg.multi_dot([kltpicker.j_r_rho[n, :, :], d_rho_psd_quad_ker,
                                                     kltpicker.j_r_rho[n, :, :].transpose()])
            tmp = np.linalg.multi_dot([sqrt_diag_quad_nys, h_nodes, sqrt_diag_quad_nys.transpose()])
            eig_vals, eig_funcs = eigh(tmp)
            eig_vals = np.real(eig_vals)
            eig_vals = eig_vals[::-1]  # Descending.
            eig_funcs = eig_funcs[:, ::-1]
            eig_vals = np.where(np.abs(eig_vals) < np.spacing(1), 0, eig_vals)
            eig_funcs[:, eig_vals == 0] = 0
            eig_func_tot[n, :, :] = eig_funcs
            eig_val_tot[n, :] = eig_vals
        r_idx = np.arange(0, NUM_QUAD_NYS)
        c_idx = np.arange(0, kltpicker.max_order)
        r_idx = np.tile(r_idx, kltpicker.max_order)
        c_idx = np.tile(c_idx, (NUM_QUAD_NYS, 1))
        eig_val_tot = eig_val_tot.flatten()
        r_idx = r_idx.transpose().flatten()
        c_idx = c_idx.transpose().flatten()
        sort_idx = np.argsort(eig_val_tot)
        sort_idx = sort_idx[::-1]
        eig_val_tot = eig_val_tot[sort_idx]
        r_idx = r_idx[sort_idx]
        c_idx = c_idx[sort_idx]
        sum_of_eig = np.sum(eig_val_tot)
        cum_sum_eig_val = np.cumsum(eig_val_tot / sum_of_eig)
        last_eig_idx = (cum_sum_eig_val > PERCENT_EIG_FUNC).argmax() + 1
        eig_val = np.zeros((2 * last_eig_idx, 1))
        eig_func = np.zeros((2 * last_eig_idx, kltpicker.cosine.shape[1]))
        count = 0
        for i in range(last_eig_idx):
            order = c_idx[i]
            idx_of_eig = r_idx[i]
            h_samp = np.sqrt(kltpicker.rsamp_r) * np.linalg.multi_dot([kltpicker.j_samp[order, :, :],
                                                                       np.diag(
                                                                           kltpicker.rho * self.psd * kltpicker.quad_ker),
                                                                       kltpicker.j_r_rho[order, :, :].transpose()])
            v_correct = (1 / np.sqrt(kltpicker.quad_nys)) * eig_func_tot[order, :, idx_of_eig]
            v_nys = np.dot(h_samp, (kltpicker.quad_nys * v_correct)) / eig_val_tot[i]
            v_nys = cp.reshape(v_nys[kltpicker.idx_rsamp], kltpicker.idx_rsamp.shape)
            if not order:
                eig_func[count, :] = (1 / np.sqrt(2 * np.pi)) * v_nys
                eig_val[count, 0] = eig_val_tot[i]
                count += 1
            else:
                eig_func[count, :] = np.sqrt((1 / np.pi)) * v_nys * kltpicker.cosine[order, :]
                eig_val[count, 0] = eig_val_tot[i]
                count += 1
                eig_func[count, :] = np.sqrt((1 / np.pi)) * v_nys * kltpicker.sine[order, :]
                eig_val[count, 0] = eig_val_tot[i]
                count += 1
        eig_val = eig_val[eig_val > 0]
        eig_func = eig_func[0:len(eig_val), :]
        if eig_func.shape[0] < MAX_FUN:
            num_of_fun = eig_func.shape[0]
        else:
            num_of_fun = MAX_FUN
        self.eig_func = eig_func
        self.eig_val = eig_val
        self.num_of_func = num_of_fun

    def construct_klt_templates_gpu(self, kltpicker):
        """Constructing the KLTpicker templates as the eigenfunctions of a given kernel."""
        eig_func_tot = cp.zeros((kltpicker.max_order, NUM_QUAD_NYS, NUM_QUAD_NYS))
        eig_val_tot = cp.zeros((kltpicker.max_order, NUM_QUAD_NYS))
        sqrt_rr = cp.sqrt(cp.array(kltpicker.r_r))
        d_rho_psd_quad_ker = cp.diag(cp.array(kltpicker.rho)) * cp.diag(cp.array(self.psd)) * cp.diag(cp.array(kltpicker.quad_ker))
        sqrt_diag_quad_nys = cp.sqrt(cp.diag(cp.array(kltpicker.quad_nys)))
        j_r_rho = cp.array(kltpicker.j_r_rho)
        for n in range(kltpicker.max_order):
            h_nodes = sqrt_rr * cp.dot(j_r_rho[n, :, :], cp.dot(d_rho_psd_quad_ker,
                                                     j_r_rho[n, :, :].transpose()))
            tmp = cp.dot(sqrt_diag_quad_nys, cp.dot(h_nodes, sqrt_diag_quad_nys.transpose()))
            eig_vals, eig_funcs = cp.linalg.eigh(tmp)
            eig_vals = cp.real(eig_vals)
            eig_vals = eig_vals[::-1]  # Descending.
            eig_funcs = eig_funcs[:, ::-1]
            eig_vals = cp.where(cp.abs(eig_vals) < np.spacing(1), 0, eig_vals)
            eig_funcs[:, eig_vals == 0] = 0
            eig_func_tot[n, :, :] = eig_funcs
            eig_val_tot[n, :] = eig_vals
            
        c_idx = cp.tile(cp.arange(0, kltpicker.max_order), (NUM_QUAD_NYS,1)).transpose().flatten()
        r_idx = cp.tile(cp.arange(0, NUM_QUAD_NYS), kltpicker.max_order)
        eig_val_tot = eig_val_tot.flatten()
        sort_idx = cp.argsort(eig_val_tot)
        sort_idx = sort_idx[::-1]
        eig_val_tot = eig_val_tot[sort_idx]
        r_idx = r_idx[sort_idx]
        c_idx = c_idx[sort_idx]
        sum_of_eig = cp.sum(eig_val_tot)
        cum_sum_eig_val = cp.cumsum(eig_val_tot / sum_of_eig)
        last_eig_idx = (cum_sum_eig_val > PERCENT_EIG_FUNC).argmax() + 1
        last_eig_idx = int(last_eig_idx)
        eig_val = cp.zeros((2 * last_eig_idx, 1))
        eig_func = cp.zeros((2 * last_eig_idx, kltpicker.cosine.shape[1]))
        count = 0
        rsamp_r = cp.array(kltpicker.rsamp_r)
        rho = cp.array(kltpicker.rho)
        psd = cp.array(self.psd)
        quad_ker = cp.array(kltpicker.quad_ker)
        quad_nys = cp.array(kltpicker.quad_nys)
        cosine = cp.array(kltpicker.cosine)
        sine = cp.array(kltpicker.sine)
        j_samp = cp.array(kltpicker.j_samp)
        for i in range(last_eig_idx):
            order = c_idx[i]
            idx_of_eig = r_idx[i]
            h_samp = cp.sqrt(rsamp_r) * cp.dot(j_samp[order, :, :], cp.dot(cp.diag(rho * psd * quad_ker),
                                                                       j_r_rho[order, :, :].transpose()))
            v_correct = (1 / cp.sqrt(quad_nys)) * eig_func_tot[order, :, idx_of_eig]
            v_nys = cp.dot(h_samp, (quad_nys * v_correct)) / eig_val_tot[i]
            v_nys = cp.reshape(v_nys[kltpicker.idx_rsamp], kltpicker.idx_rsamp.shape)
            if not order:
                eig_func[count, :] = (1 / cp.sqrt(2 * np.pi)) * v_nys
                eig_val[count, 0] = eig_val_tot[i]
                count += 1
            else:
                eig_func[count, :] = cp.sqrt((1 / cp.pi)) * v_nys * cosine[order, :]
                eig_val[count, 0] = eig_val_tot[i]
                count += 1
                eig_func[count, :] = cp.sqrt((1 / np.pi)) * v_nys * sine[order, :]
                eig_val[count, 0] = eig_val_tot[i]
                count += 1
        eig_val = eig_val[eig_val > 0]
        eig_func = eig_func[0:len(eig_val), :]
        if eig_func.shape[0] < MAX_FUN:
            num_of_fun = eig_func.shape[0]
        else:
            num_of_fun = MAX_FUN
        self.eig_func = eig_func
        self.eig_val = eig_val
        self.num_of_func = num_of_fun


    def detect_particles(self, kltpicker):
        """
        Construct the scoring matrix and then use the picking_from_scoring_mat function to pick particles and noise
        images.
        """
        eig_func_stat = self.eig_func[0:self.num_of_func, :]
        eig_val_stat = self.eig_val[0:self.num_of_func]
        for i in range(self.num_of_func):
            tmp_func = np.reshape(eig_func_stat[i, :], (kltpicker.patch_size_func, kltpicker.patch_size_func))
            tmp_func[kltpicker.rad_mat > np.floor((kltpicker.patch_size_func - 1) / 2)] = 0
            eig_func_stat[i, :] = tmp_func.flatten()
        [q, r] = np.linalg.qr(eig_func_stat.transpose(), 'complete')
        r = r[0:self.num_of_func, 0:self.num_of_func]
        kappa = np.linalg.multi_dot([r, np.diag(eig_val_stat), r.transpose()]) + (
                self.approx_noise_var * np.eye(self.num_of_func))
        kappa_inv = np.linalg.inv(kappa)
        t_mat = (1 / self.approx_noise_var) * np.eye(self.num_of_func) - kappa_inv
        
        [D, P] = np.linalg.eigh(t_mat)
        D = D[::-1]  # Descending.
        P = P[:, ::-1]
        p_full = np.zeros(q.shape)
        p_full[0:t_mat.shape[0], 0:t_mat.shape[1]] = P
        
        mu = np.linalg.slogdet((1 / self.approx_noise_var) * kappa)[1]
        last_block_row = self.mc_size[0] - kltpicker.patch_size_func + 1
        last_block_col = self.mc_size[1] - kltpicker.patch_size_func + 1
        num_of_patch_row = last_block_row
        num_of_patch_col = last_block_col
        
        qp = q @ p_full
        qp = qp.transpose().copy()
        
        log_test_mat = np.zeros((num_of_patch_row, num_of_patch_col))
        for i in range(self.num_of_func):
            qp_tmp = np.reshape(qp[i, :], (kltpicker.patch_size_func, kltpicker.patch_size_func)).transpose()
            
            qp_tmp = np.flip(np.flip(qp_tmp, 0), 1)
            
            scoreTmp = signal.fftconvolve(self.noise_mc, qp_tmp,'valid')
            log_test_mat = log_test_mat + D[i] * abs(scoreTmp ** 2)
        
        log_test_mat = log_test_mat.transpose() - mu
        neigh = np.ones((kltpicker.patch_size_func, kltpicker.patch_size_func))
        log_test_n = signal.fftconvolve(log_test_mat, neigh, 'valid')
        [num_picked_particles, num_picked_noise] = picking_from_scoring_mat(log_test_n.transpose(), self.mrc_name, kltpicker,
                                                                            self.mg_big_size)
        return num_picked_particles, num_picked_noise
