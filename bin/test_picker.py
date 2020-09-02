from pathlib import Path
import warnings
from sys import exit
import numpy as np
from kltpicker.kltpicker import KLTPicker
from kltpicker.util import trig_interpolation
from kltpicker.kltpicker_input import parse_args, get_args
import mrcfile
from kltpicker.micrograph import Micrograph
from kltpicker.cryo_utils import downsample, downsample_cp
import cupy as cp

warnings.filterwarnings("ignore")


def process_micrograph(micrograph, picker):
    micrograph.cutoff_filter(picker.patch_size)
    micrograph.estimate_rpsd(picker.patch_size, picker.max_iter, picker.no_gpu)
    micrograph.approx_noise_psd = micrograph.approx_noise_psd + np.median(micrograph.approx_noise_psd) / 10
    micrograph.prewhiten_micrograph(picker.no_gpu)
    micrograph.estimate_rpsd(picker.patch_size, picker.max_iter, picker.no_gpu)   
    micrograph.psd = np.abs(trig_interpolation(np.pi * micrograph.r.astype('float64'), micrograph.approx_clean_psd,
                                               picker.rho.astype('float64')))
    micrograph.construct_klt_templates(picker)
    num_picked_particles, num_picked_noise = micrograph.detect_particles(picker)
    return [micrograph.mrc_name, num_picked_particles, num_picked_noise]

def get_micrograph(mrc_file, mgscale, no_gpu):
    """Reads an mrc file and downsamples it."""
    mrc = mrcfile.open(mrc_file, permissive=True)
    mrc_data = mrc.data.astype('float64').transpose()
    mrc.close()
    mrc_size = mrc_data.shape
    mrc_data = np.rot90(mrc_data)
    if no_gpu:
        mrc_data = downsample(mrc_data, (np.floor(np.multiply(mgscale, mrc_size))).astype(int))
    else:
        mrc_data = downsample_cp(cp.asarray(mrc_data), int(np.floor(mgscale * mrc_size[0])))
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
    args = parse_args(1)
    if args.output_dir is None or args.input_dir is None:
        args.input_dir, args.output_dir, args.particle_size, args.num_of_particles, args.num_of_noise_images, args.no_gpu = get_args(1)
    num_files = len(list(Path(args.input_dir).glob("*.mrc")))
    if num_files > 0:
        print("Running on %i files." % len(list(Path(args.input_dir).glob("*.mrc"))))
    else:
        print("Could not find any .mrc files in %s. \nExiting..." % args.input_dir)
        exit(0)
    picker = KLTPicker(args)
    mrc_files = picker.input_dir.glob("*.mrc")   
    print("Preprocessing (usually takes up to 1 minute)...")
    picker.preprocess()
    if not picker.output_dir.exists():
        Path.mkdir(picker.output_dir)
    for mrc_file in mrc_files:
        micrograph = get_micrograph(mrc_file, picker.mgscale, picker.no_gpu)
        process_micrograph(micrograph, picker)
    print("done")
    
if __name__ == "__main__":
    main()