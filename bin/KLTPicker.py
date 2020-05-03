#!/usr/bin/env python

from pathlib import Path
import warnings
from sys import exit
import argparse
import numpy as np
from kltpicker.kltpicker import KLTPicker
from kltpicker.util import trig_interpolation
from kltpicker.kltpicker_input import get_args
from tqdm import tqdm
import mrcfile
from kltpicker.micrograph import Micrograph
from kltpicker.cryo_utils import downsample

warnings.filterwarnings("ignore")

# Globals:
PERCENT_EIG_FUNC = 0.99
EPS = 10 ** (-2)  # Convergence term for ALS.
NUM_QUAD_NYS = 2 ** 10
NUM_QUAD_KER = 2 ** 10
MAX_FUN = 400


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help='Input directory.')
    parser.add_argument('--output_dir', help='Output directory.')
    parser.add_argument('-s', '--particle_size', help='Expected size of particles in pixels.', default=300, type=int)
    parser.add_argument('--num_of_particles',
                        help='Number of particles to pick per micrograph. If set to -1 will pick all particles.',
                        default=-1, type=int)
    parser.add_argument('--num_of_noise_images', help='Number of noise images to pick per micrograph.',
                        default=0, type=int)
    parser.add_argument('--max_iter', help='Maximum number of iterations.', default=6 * (10 ** 4), type=int)
    parser.add_argument('--gpu_use', action='store_true', default=False)
    parser.add_argument('--max_order', help='Maximum order of eigenfunction.', default=100, type=int)
    parser.add_argument('--percent_eigen_func', help='', default=0.99, type=float)
    parser.add_argument('--max_functions', help='', default=400, type=int)
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose.', default=False)
    parser.add_argument('--threshold', help='Threshold for the picking', default=0, type=float)
    parser.add_argument('--show_figures', action='store_true', help='Show figures', default=False)
    parser.add_argument('--preprocess', action='store_false', help='Do not run preprocessing.', default=True)
    args = parser.parse_args()
    return args


def process_micrograph(micrograph, picker):
    micrograph.cutoff_filter(picker.patch_size)
    micrograph.estimate_rpsd(picker.patch_size, picker.max_iter)
    micrograph.approx_noise_psd = micrograph.approx_noise_psd + np.median(micrograph.approx_noise_psd) / 10
    micrograph.prewhiten_micrograph()
    micrograph.estimate_rpsd(picker.patch_size, picker.max_iter)
    micrograph.psd = np.abs(trig_interpolation(np.pi * micrograph.r.astype('float64'), micrograph.approx_clean_psd,
                                               picker.rho.astype('float64')))
    micrograph.construct_klt_templates(picker)
    num_picked_particles, num_picked_noise = micrograph.detect_particles(picker)
    return num_picked_particles, num_picked_noise

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

def main():
    args = parse_args()
    if args.output_dir is None or args.input_dir is None:
        args.input_dir, args.output_dir, args.particle_size, args.num_of_particles, args.num_of_noise_images = get_args()
    num_files = len(list(Path(args.input_dir).glob("*.mrc")))
    if num_files > 0:
        print("Running on %i files." % len(list(Path(args.input_dir).glob("*.mrc"))))
    else:
        print("Could not find any .mrc files in %s. \nExiting..." % args.input_dir)
        exit(0)
    picker = KLTPicker(args)
    if args.preprocess:
        print("Preprocessing...")
        picker.preprocess()
        print("Preprocess finished.")
    else:
        print("Skipping preprocessing.")
    mrc_files = picker.input_dir.glob("*.mrc")
    for mrc_file in tqdm(list(mrc_files), desc='Picking particles from micrographs...'):
        micrograph = get_micrograph(mrc_file, picker.mgscale)
        process_micrograph(micrograph, picker)
    print("Finished successfully!")

if __name__ == "__main__":
    main()

