from pathlib import Path
import warnings
from sys import exit
import numpy as np
from .kltpicker import KLTPicker
from .util import trig_interpolation
from .kltpicker_input import parse_args, get_args, progress_bar, write_summary
import mrcfile
from .micrograph import Micrograph
from .cryo_utils import downsample, downsample_cp
import multiprocessing as mp
import os
warnings.filterwarnings("ignore")

# Check if CuPy is installed:
try:
    import cupy as cp
    HAS_CUPY = 1
except:
    HAS_CUPY = 0
    
# Globals:
MAX_PROCESSES = 8

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
        mrc_data = downsample_cp(cp.asarray(mrc_data), (np.floor(np.multiply(mgscale, mrc_size))).astype(int))
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
        summary = process_micrograph(micrograph, picker)
    if picker.verbose:
        print("Picked %d particles and %d noise images out of %s." %(summary[1], summary[2], summary[0]))
    return summary
        
def multi_process_micrograph(param):
    """
    Process micrographs in parallel, no GPU.
    """    
    mrc_file = param[0]
    picker = param[1]
    micrograph = get_micrograph(mrc_file, picker.mgscale, picker.no_gpu)
    summary = process_micrograph(micrograph, picker)
    if picker.verbose:
        print("Picked %d particles and %d noise images out of %s." %(summary[1], summary[2], summary[0]))
    return summary

 
def multi_process_micrograph_pool(gpu_index, num_cpus, batch, shared_list):
    """
    A wrapper function that allows processing many mrcs in parallel using
    a worker pool.
    """
    for b in batch:
        b.append(gpu_index)
    with mp.Pool(processes=int(num_cpus)) as pool:
        shared_list += [x for x in pool.imap_unordered(multi_process_micrograph_gpu, batch, chunksize=7)]

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
    args = parse_args(HAS_CUPY)
    if args.output_dir is None or args.input_dir is None:
        args.input_dir, args.output_dir, args.particle_size, args.num_of_particles, args.num_of_noise_images, args.no_gpu = get_args(HAS_CUPY)
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
    if not picker.output_dir.exists():
        Path.mkdir(picker.output_dir)
    if args.no_gpu:
        print("Preprocess finished. Picking particles...")
        os.environ["NUMBA_DISABLE_CUDA"] = "1"
        if not picker.verbose:
            p = mp.Process(target=progress_bar, args=[picker.output_particles / "star", num_files], name="KLTPicker_ProgressBar")
            p.start() 
        with mp.Pool(processes=min(MAX_PROCESSES, mp.cpu_count() - 2)) as pool:
            res = [x for x in pool.imap_unordered(multi_process_micrograph, params)]
    else:
        mem_usage = get_mempool_usage(params[0])    
        print("Preprocess finished. Picking particles...")
        cpus_per_gpu = calc_cpus_per_gpu(mem_usage)
        batches = get_mrc_batches(params, cpus_per_gpu)
        if not picker.verbose:
            p = mp.Process(target=progress_bar, args=[picker.output_particles / "star", num_files], name="KLTPicker_ProgressBar")
            p.start()  
        manager = mp.Manager()
        res = manager.list()
        jobs = []
        for i in range(len(cpus_per_gpu)):
            if cpus_per_gpu[i]:
                p = mp.Process(target=multi_process_micrograph_pool, args=[i, cpus_per_gpu[i], batches[i], res], name="KLTPicker%d"%i)
                jobs.append(p)
                p.start()
        
        for proc in jobs:
            proc.join()
    write_summary(picker.output_dir, res)
    num_files = len(res)
    num_particles = sum([row[1] for row in res])
    num_noise = sum([row[2] for row in res]) 
    print("Picked %d particles and %d noise images out of %d micrographs." %(num_particles, num_noise, num_files))