import numpy as np
from kltpicker import KLTPicker
import argparse
from kltpicker import Micrograph
import mrcfile
from kltpicker.cryo_utils import downsample
from pathlib import Path

TEST_DATA = "/home/dalitcohen/Documents/projects/test/numpy/"
TEST_DATA_RPSD = "/home/dalitcohen/Documents/projects/test/numpy/before_2nd_estimate_rpsd/"

parser = argparse.ArgumentParser()
args = parser.parse_args()
args.particle_size = 300
args.input_dir = "/home/dalitcohen/Documents/mrcs"
args.output_dir = "/home/dalitcohen/Documents/out"
args.gpu_use = 0
args.max_order = 100
args.num_of_particles = -1
args.num_of_noise_images =0
args.threshold = 0
args.max_iter = 6 * (10 ** 4)


def get_micrograph(mrc_file, mgscale):
    """Reads .mrc files and downsamples them."""
    mrc = mrcfile.open(mrc_file)
    mrc_data = mrc.data.astype('float64').transpose()
    mrc.close()
    mrc_size = mrc_data.shape
    mrc_data = np.rot90(mrc_data)
    mrc_data = downsample(mrc_data[np.newaxis, :, :], int(np.floor(mgscale * mrc_size[0])))[0]
    if np.mod(mrc_data.shape[0], 2) == 0:  # Odd size is needed.
        mrc_data = mrc_data[0:-1, :]
    if np.mod(mrc_data.shape[1], 2) == 0:  # Odd size is needed.
        mrc_data = mrc_data[:, 0:-1]
    mrc_data = mrc_data - np.mean(mrc_data.transpose().flatten())
    mrc_data = mrc_data / np.linalg.norm(mrc_data, 'fro')
    mc_size = mrc_data.shape
    micrograph = Micrograph(mrc_data, mc_size, mrc_file.name, mrc_size)
    return micrograph

if __name__ == "__main__":
    picker = KLTPicker(args)
    # get the data:
    picker.quad_ker = np.load(TEST_DATA + "quad_ker.npy")
    picker.quad_nys = np.load(TEST_DATA + "quad_nys.npy")
    picker.rho = np.load(TEST_DATA + "rho.npy")
    picker.j_r_rho = np.load(TEST_DATA + "j_r_rho.npy")
    picker.j_samp = np.load(TEST_DATA + "j_samp.npy")
    picker.cosine = np.load(TEST_DATA + "cosine.npy")
    picker.sine = np.load(TEST_DATA + "sine.npy")
    picker.rsamp_r = np.load(TEST_DATA + "rsamp_r.npy")
    picker.r_r = np.load(TEST_DATA + "r_r.npy")
    picker.rad_mat = np.load(TEST_DATA + "rad_mat.npy")
    picker.rsamp_length = 1521
    picker.input_dir = Path("/home/dalitcohen/Documents/mrcs")
    picker.mgscale = 100 / 300
    mrc_files = picker.input_dir.glob("*.mrc")
    micrograph = get_micrograph(list(mrc_files)[0], picker.mgscale)
    micrograph.psd = np.load(TEST_DATA_RPSD + "psd.npy")
    micrograph.approx_noise_psd = np.load(TEST_DATA_RPSD + "approx_noise_psd.npy")
    micrograph.approx_clean_psd = np.load(TEST_DATA_RPSD + "approx_clean_psd.npy")
    micrograph.approx_noise_var = np.load(TEST_DATA_RPSD + "approx_noise_var.npy")
    micrograph.noise_mc = np.load(TEST_DATA_RPSD+ "noise_mc.npy")
    micrograph.r = np.load(TEST_DATA_RPSD + "r.npy")
    #run:
    micrograph.estimate_rpsd(picker.patch_size, picker.max_iter)
    print("done")