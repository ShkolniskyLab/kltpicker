#!/usr/bin/env python

from pathlib import Path
import warnings
from sys import exit
import argparse
import numpy as np
from kltpicker.kltpicker import KLTPicker
from kltpicker.util import trig_interpolation
from kltpicker.kltpicker_input import get_args
import mrcfile
from kltpicker.micrograph import Micrograph
from kltpicker.cryo_utils import downsample, downsample_cp
import cupy as cp
import multiprocessing as mp
import progressbar
from time import sleep

warnings.filterwarnings("ignore")

# Globals:
PERCENT_EIG_FUNC = 0.99
EPS = 10 ** (-2)  # Convergence term for ALS.


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
    parser.add_argument('--no_gpu', action='store_true', default=False)
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
    micrograph.estimate_rpsd(picker.patch_size, picker.max_iter, picker.no_gpu)
    micrograph.approx_noise_psd = micrograph.approx_noise_psd + np.median(micrograph.approx_noise_psd) / 10
    micrograph.prewhiten_micrograph(picker.no_gpu)
    micrograph.estimate_rpsd(picker.patch_size, picker.max_iter, picker.no_gpu)   
    micrograph.psd = np.abs(trig_interpolation(np.pi * micrograph.r.astype('float64'), micrograph.approx_clean_psd,
                                               picker.rho.astype('float64')))
    micrograph.construct_klt_templates(picker)
    num_picked_particles, num_picked_noise = micrograph.detect_particles(picker)
    return num_picked_particles, num_picked_noise

def get_micrograph(mrc_file, mgscale, no_gpu):
    """Reads .mrc files and downsamples them."""
    mrc = mrcfile.open(mrc_file)
    mrc_data = mrc.data.astype('float64').transpose()
    mrc.close()
    mrc_size = mrc_data.shape
    mrc_data = np.rot90(mrc_data)
    if no_gpu:
        mrc_data = downsample(mrc_data, int(np.floor(mgscale * mrc_size[0])))
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

def get_mempool_usage(param):
    """
    Calculate how much memory is used in the GPU by a single process on one mrc.

    """       
    mrc_file = param[0]
    picker = param[1]
    start = cp.cuda.runtime.memGetInfo()[0]
    micrograph = get_micrograph(mrc_file, picker.mgscale, picker.no_gpu)
    process_micrograph(micrograph, picker)
    finish = cp.cuda.runtime.memGetInfo()[0]
    mem_usage = (start-finish)*1.2
    return mem_usage

def calc_cpus_per_gpu(mem_usage):
    """
    Find how many available GPUs there are, and for each available GPU compute
    the maximal number of processes that can use it (based on memory usage).
    """
    num_gpus = cp.cuda.runtime.getDeviceCount()
    num_available_cpus = mp.cpu_count()    
    cpus_per_gpu = []
    for gpu_index in range(num_gpus):
        if sum(cpus_per_gpu) < num_available_cpus:
            cp.cuda.runtime.setDevice(gpu_index)	
            free_mem = cp.cuda.runtime.memGetInfo()[0]
            cpus_per_gpu.append(np.floor(free_mem/mem_usage))
    return cpus_per_gpu
    
def multi_process_micrograph_gpu(param):
    """
    Process micrographs in parallel, using GPU.
    """    
    mrc_file = param[0]
    picker = param[1]
    gpu_index = param[2]
    with cp.cuda.Device(gpu_index):
        micrograph = get_micrograph(mrc_file, picker.mgscale, picker.no_gpu)
        process_micrograph(micrograph, picker)
        
def multi_process_micrograph(param):
    """
    Process micrographs in parallel, no GPU.
    """    
    mrc_file = param[0]
    picker = param[1]
    micrograph = get_micrograph(mrc_file, picker.mgscale, picker.no_gpu)
    process_micrograph(micrograph, picker)
 
def multi_process_micrograph_pool(gpu_index, num_cpus, batch):
    """
    A wrapper function that allows processing many mrcs in parallel using
    a worker pool.
    """
    for b in batch:
        b.append(gpu_index)
    with mp.Pool(processes=int(num_cpus)) as pool:
        pool.map(multi_process_micrograph_gpu, batch, chunksize=5)  
 
def progress_bar(output_dir, num_mrcs):
    """
    Progress bar function that reports the progress of the program, by 
    periodically checking how many output files have been written. Shows both
    percentage completed and time elapsed.
    """
    num_finished = len(list(Path(output_dir).glob("*.star")))
    bar = progressbar.ProgressBar(maxval=num_mrcs, widgets=["[", progressbar.Timer(), "] ", progressbar.Bar('#', '|', '|'), ' (', progressbar.Percentage(), ')'])
    bar.start()
    while num_finished < num_mrcs:
        num_finished = len(list(Path(output_dir).glob("*.star")))
        bar.update(num_finished)
        sleep(1)
    bar.finish()
    print("Finished successfully!")

def get_mrc_batches(params, cpus_per_gpu):
    """
    Split the mrc files into batches proportionally sized to the number of
    processes to be run using each GPU.
    """
    cpus_per_gpu = np.array(cpus_per_gpu)
    cpus_per_gpu = cpus_per_gpu / np.sum(cpus_per_gpu)
    batch_sizes = np.ceil(len(params) * cpus_per_gpu / np.sum(cpus_per_gpu))
    indices = np.zeros(len(batch_sizes) + 1).astype(int)
    indices[1:] = np.cumsum(batch_sizes).astype("int")
    batches = [params[indices[i]:indices[i+1]] for i in range(len(indices)-1)]
    return batches
    
def main():
    args = parse_args()
    if args.output_dir is None or args.input_dir is None:
        args.input_dir, args.output_dir, args.particle_size, args.num_of_particles, args.num_of_noise_images, args.no_gpu = get_args()
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
    params = [[mrc_file, picker] for mrc_file in mrc_files]
  
    if not args.no_gpu:
        mem_usage = get_mempool_usage(params[0])    
        print("Preprocess finished. Picking particles...")
        cpus_per_gpu = calc_cpus_per_gpu(mem_usage)
        batches = get_mrc_batches(params, cpus_per_gpu)
        
        p = mp.Process(target=progress_bar, args=[picker.output_particles / "star", num_files], name="KLTPicker_ProgressBar")
        p.start() 
        
        for i in range(len(cpus_per_gpu)):
            if cpus_per_gpu[i]:
                p = mp.Process(target=multi_process_micrograph_pool, args=[i, cpus_per_gpu[i], batches[i]], name="KLTPicker%d"%i)
                p.start()
    else:
        print("Preprocess finished. Picking particles...")
        p = mp.Process(target=progress_bar, args=[picker.output_particles / "star", num_files], name="KLTPicker_ProgressBar")
        p.start() 
        with mp.Pool(processes=min(8, mp.cpu_count() - 2)) as pool:
            pool.map(multi_process_micrograph, params)  

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()