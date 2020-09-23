from pathlib import Path
import numpy as np
import scipy.special as ssp
from .cryo_utils import lgwt
from .kltpicker_input import get_start_time
from multiprocessing import Pool, cpu_count
import time

# Globals:
EPS = 10 ** (-2)  # Convergence term for ALS.
PERCENT_EIG_FUNC = 0.99
NUM_QUAD_NYS = 2 ** 7
NUM_QUAD_KER = 2 ** 7
MAX_FUN = 400
MAX_ITER = 6 * (10 ** 4)
MAX_ORDER = 100
THRESHOLD = 0

class KLTPicker:
    """
    KLTpicker object that holds all variables that are used in the computations.

    ...
    Attributes
    ----------
    particle_size : float
        Size of particles to look for in micrographs.
    input_dir : str
        Directory from which to read .mrc files.
    output_dir : str
        Output directory in which to write results.
    no_gpu : bool
        Optional - whether to use GPU or not.
    mgscale : float
        Scaling parameter.
    max_order : int
        Maximal order of eigenfunctions.
    micrographs : np.ndarray
        Array of 2-D micrographs.
    patch_size_pick_box : int
        Particle box size to use.
    num_of_particles : int
        Number of particles to pick per micrograph.
    num_of_noise_images : int
        Number of noise images.
    threshold : float
        Threshold for the picking.
    patch_size : int
        Approximate size of particle after downsampling.
    patch_size_func : int
        Size of disc for computing the eigenfunctions.
    max_iter : int
        Maximal number of iterations for PSD approximation.
    rsamp_length : int
    rad_mat : np.ndarray
    quad_ker : np.ndarray
    quad_nys : np.ndarray
    rho : np.ndarray
    j_r_rho : np.ndarray
    j_samp : np.ndarray
    cosine : np.ndarray
    sine : np.ndarray
    rsamp_r : np.ndarray
    r_r : np.ndarray


    Methods
    -------
    preprocess()
        Initializes parameters needed for the computation.
    get_micrographs()
        Reads .mrc files, downsamples them and adds them to the KLTpicker object.
    """

    def __init__(self, args):
        self.particle_size = args.particle_size
        self.input_dir = Path(args.input_dir)
        self.output_dir = Path(args.output_dir)
        self.output_noise = self.output_dir / ('pickedNoiseParticleSize%d' % args.particle_size)
        self.output_particles = self.output_dir / ('pickedParticlesParticleSize%d' % args.particle_size)
        self.no_gpu = args.no_gpu
        self.mgscale = 100 / args.particle_size
        self.max_order = MAX_ORDER
        self.quad_ker = 0
        self.quad_nys = 0
        self.rho = 0
        self.j_r_rho = 0
        self.j_samp = 0
        self.cosine = 0
        self.sine = 0
        self.rsamp_r = 0
        self.r_r = 0
        self.patch_size_pick_box = np.floor(self.mgscale * args.particle_size)
        self.num_of_particles = args.num_particles
        self.num_of_noise_images = args.num_noise
        self.threshold = THRESHOLD
        patch_size = np.floor(0.8 * self.mgscale * args.particle_size)
        if np.mod(patch_size, 2) == 0:
            patch_size -= 1
        self.patch_size = patch_size
        patch_size_function = np.floor(0.4 * self.mgscale * args.particle_size)
        if np.mod(patch_size_function, 2) == 0:
            patch_size_function -= 1
        self.patch_size_func = int(patch_size_function)
        self.max_iter = MAX_ITER
        self.rsamp_length = 0
        self.rad_mat = 0
        self.verbose = args.verbose
        self.num_mrcs = 0
        self.start_time = get_start_time(self.output_dir)


    def preprocess(self):
        """Initializes parameters."""
        radmax = np.floor((self.patch_size_func - 1) / 2)
        x = np.arange(-radmax, radmax + 1, 1).astype('float64')
        X, Y = np.meshgrid(x, x)
        rad_mat = np.sqrt(np.square(X) + np.square(Y))
        rsamp, idx_rsamp = np.unique(rad_mat.transpose().flatten(), return_inverse=True)
        theta = np.arctan2(Y, X).transpose().flatten()
        rho, quad_ker = lgwt(NUM_QUAD_KER, 0, np.pi)
        rho = np.flipud(rho.astype('float64'))
        quad_ker = np.flipud(quad_ker.astype('float64'))
        r, quad_nys = lgwt(NUM_QUAD_NYS, 0, radmax)
        r = np.flipud(r.astype('float64'))
        quad_nys = np.flipud(quad_nys.astype('float64'))
        r_r = np.outer(r, r)
        r_rho = np.outer(r, rho)
        rsamp_r = np.outer(np.ones(len(rsamp)), r)
        rsamp_rho = np.outer(rsamp, rho)
        pool = Pool(max(cpu_count()-2, 1))
        res_j_r_rho = pool.starmap(ssp.jv, [(n, r_rho) for n in range(self.max_order)])
        res_j_samp = pool.starmap(ssp.jv, [(n, rsamp_rho) for n in range(self.max_order)])
        res_cosine = pool.map(np.cos, [n * theta for n in range(self.max_order)])
        res_sine = pool.map(np.sin, [n * theta for n in range(self.max_order)])
        pool.close()
        pool.join()
        j_r_rho = np.squeeze(res_j_r_rho)
        j_samp = np.squeeze(res_j_samp)
        cosine = np.squeeze(res_cosine)
        sine = np.squeeze(res_sine)
        cosine[0, :] = 0
        self.quad_ker = quad_ker
        self.quad_nys = quad_nys
        self.rho = rho
        self.j_r_rho = j_r_rho
        self.j_samp = j_samp
        self.cosine = cosine
        self.sine = sine
        self.rsamp_r = rsamp_r
        self.r_r = r_r
        self.rad_mat = rad_mat
        self.idx_rsamp = idx_rsamp

