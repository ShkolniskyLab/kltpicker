from pathlib import Path
import warnings
from sys import exit, argv
import numpy as np
from .kltpicker import KLTPicker
from .util import trig_interpolation
from .kltpicker_input import parse_args, get_args, progress_bar, check_num_finished, check_for_newer_version, check_output_dir
import mrcfile
from .micrograph import Micrograph
from .cryo_utils import downsample, downsample_gpu
import multiprocessing as mp
import os
import time
warnings.filterwarnings("ignore")

# Check if CuPy is installed:
try:
    import cupy as cp
    HAS_CUPY = 1
except:
    HAS_CUPY = 0

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
        mrc_data = downsample_gpu(cp.asarray(mrc_data), (np.floor(np.multiply(mgscale, mrc_size))).astype(int))
    if np.mod(mrc_data.shape[0], 2) == 0:  # Odd size is needed.
        mrc_data = mrc_data[0:-1, :]
    if np.mod(mrc_data.shape[1], 2) == 0:  # Odd size is needed.
        mrc_data = mrc_data[:, 0:-1]
    mrc_data = mrc_data - np.mean(mrc_data.transpose().flatten())
    mrc_data = mrc_data / np.linalg.norm(mrc_data, 'fro')
    mc_size = mrc_data.shape
    micrograph = Micrograph(mrc_data, mc_size, mrc_file.name, mrc_size)
    return micrograph

def get_mempool_usage(param, gpu_index):
    """
    Calculate how much memory is used in the GPU by a single process on one mrc.

    """
    with cp.cuda.Device(gpu_index):    
        mrc_file = param[0]
        picker = param[1]
        start = cp.cuda.runtime.memGetInfo()[0]
        micrograph = get_micrograph(mrc_file, picker.mgscale, picker.no_gpu)
        process_micrograph(micrograph, picker)
        finish = cp.cuda.runtime.memGetInfo()[0]
        mem_usage = (start-finish)*1.2
        return mem_usage

def calc_procs_per_gpu(mem_usage, max_processes, gpu_indices):
    """
    Find how many available GPUs there are, and for each available GPU compute
    the maximal number of processes that can use it (based on memory usage).
    """
    num_available_cpus = min(mp.cpu_count(), max_processes)
    procs_per_gpu = {}
    for gpu_index in gpu_indices:
        if sum(procs_per_gpu.values()) < num_available_cpus:
            cp.cuda.runtime.setDevice(gpu_index)	
            free_mem = cp.cuda.runtime.memGetInfo()[0]
            procs_per_gpu[gpu_index] = min(np.floor(free_mem/mem_usage), num_available_cpus - sum(procs_per_gpu.values()))
        else:
            procs_per_gpu[gpu_index] = 0
    return procs_per_gpu

    
def multi_process_micrograph_gpu(param):
    """
    Process micrographs in parallel, using GPU.
    """
    # Unpack parameters (pool.map can map only one argument per function call).
    mrc_file = param[0]
    picker = param[1]
    gpu_index = param[2]
    with cp.cuda.Device(gpu_index):
        micrograph = get_micrograph(mrc_file, picker.mgscale, picker.no_gpu)
        summary = process_micrograph(micrograph, picker)
    if picker.verbose: # User requested detailed output.
        num_finished = check_num_finished(picker.output_particles / 'star', picker.start_time)
        print(time.strftime("%H:%M:%S", time.gmtime(time.time() - picker.start_time)) + " - Picked %d particles and %d noise images out of %s. (%3d%s)" %(summary[1], summary[2], summary[0], round(num_finished/picker.num_mrcs*100), "%"))
    return summary
        
def multi_process_micrograph(param):
    """
    Process micrographs in parallel, no GPU.
    """
    # Unpack parameters (pool.map can map only one argument per function call).
    mrc_file = param[0]
    picker = param[1]
    micrograph = get_micrograph(mrc_file, picker.mgscale, picker.no_gpu)
    summary = process_micrograph(micrograph, picker)
    if picker.verbose: # User requested detailed output.
        num_finished = check_num_finished(picker.output_particles / 'star', picker.start_time)
        print(time.strftime("%H:%M:%S", time.gmtime(time.time() - picker.start_time)) + " - Picked %d particles and %d noise images out of %s. (%3d%s)" %(summary[1], summary[2], summary[0], round(num_finished/picker.num_mrcs*100), "%"))
    return summary

 
def multi_process_micrograph_pool(gpu_index, num_procs, batch, shared_list):
    """
    A wrapper function that allows processing many micrographs in parallel using
    a worker pool.
    For each GPU, this function is called in a new process. This process runs
    a pool of workers of size num_procs, all using the same GPU. 
    """
    for b in batch:
        b.append(gpu_index)
    with mp.Pool(processes=int(num_procs)) as pool:
        shared_list += [x for x in pool.imap_unordered(multi_process_micrograph_gpu, batch, chunksize=7)]

def get_mrc_batches(params, cpus_per_gpu):
    """
    Split the mrc files into batches proportionally sized to the number of
    processes to be run using each GPU.
    """
    batches = {}
    gpus = sorted(list(cpus_per_gpu.keys()))
    cpus = np.array([cpus_per_gpu[gpu] for gpu in gpus])
    batch_sizes = np.ceil(len(params) * cpus / np.sum(cpus))
    indices = np.zeros(len(batch_sizes) + 1).astype(int)
    indices[1:] = np.cumsum(batch_sizes).astype("int")
    param_batches = [params[indices[i]:indices[i+1]] for i in range(len(indices)-1)]
    for i in range(len(param_batches)):
        batches[gpus[i]] = param_batches[i]
    return batches
       
def main():
    try:
        check_for_newer_version()
    except:
        pass
    # Because of CUDA limitations, it is impossible to fork processes after 
    # invoking CUDA. So we need to use 'spawn' start method instead.
    mp.set_start_method('spawn', force=True)
    
    # Get user arguments:
    user_input = argv
    if len(user_input) > 1: # User entered arguments. Use command line mode.
        args = parse_args(HAS_CUPY)
        # Check if user entered the mandatory arguments: input and output 
        # directory and particle size. If not, exit.
        if args.output_dir is None or args.input_dir is None or args.particle_size is None:
            print("Error: one or more of the following arguments are missing: input-dir, output-dir, particle-size. For help run kltpicker -h")
            exit()
        else:
            num_finished_output = check_output_dir(Path(args.input_dir), Path(args.output_dir), args.particle_size)
            if num_finished_output == 2:
                print("The output directory contains coordinate files for all of the micrographs in the input directory. Aborting...")
                exit()
            elif num_finished_output == 1:
                pass
            elif num_finished_output == 0:
                print("Could not find any .mrc files in %s. \nExiting..." % args.input_dir)
                exit(0)        
            else:
                if not args.only_do_unfinished:
                    print("The output directory contains coordinate files for some of the micrographs in the input directory. Use --do-unfinished-only if needed. Aborting...")
                    exit()

    else: # User didn't enter arguments, use interactive mode to get arguments.
        args = parse_args(HAS_CUPY) # Initiate args with default values.
        args.input_dir, args.output_dir, args.particle_size, args.num_particles, args.num_noise, args.no_gpu, args.gpus, args.verbose, args.max_processes, args.only_do_unfinished = get_args(HAS_CUPY)
    
    # Handle user options:     
    # If max_processes limit not set, set it to infinity.
    if args.max_processes == -1:
        args.max_processes = np.inf
    
    # Find number of .mrc files in input directory. 
    # Check if output directory already contains any output coordinate files 
    # for the micrographs in the input directory. If so, remove these 
    # micrographs from the micrographs to be processed.
    
    mrc_files = check_output_dir(Path(args.input_dir), Path(args.output_dir), args.particle_size)
    if mrc_files == 1: # Need to process all the micrographs in the input dir. 
        mrc_files = list(Path(args.input_dir).glob("*.mrc"))
    print("\nRunning on %i files." % len(mrc_files))
    
    if not args.no_gpu:
        print("Using GPUs %s."%(", ".join([str(x) for x in args.gpus])))
    if not Path(args.output_dir).exists(): # If the output directory doesn't exist, create it.
        Path.mkdir(args.output_dir)
    
    picker = KLTPicker(args) # Initiate picker object.
    picker.num_mrcs = len(mrc_files)
    
    
    # Preprocessing. If using GPU, preprocessing includes the calculation of 
    # memory that is taken up in the processing of a single micrograph in the
    # GPU.
    print("Preprocessing (usually takes up to 1 minute)...")
    picker.preprocess()
    params = [[mrc_file, picker] for mrc_file in mrc_files]
    
    if args.no_gpu: # GPU is disabled by user/not available on system.
        print("Preprocess finished. Picking particles...")
        os.environ["NUMBA_DISABLE_CUDA"] = "1"  # Disable use of CUDA by NUMBA.
        if not picker.verbose: # Display simple progress bar.
            p = mp.Process(target=progress_bar, args=[picker.output_particles / "star", len(mrc_files)], name="KLTPicker_ProgressBar")
            p.start() 
        # Pick particles. The number of concurrent processes is the minimum of
        # the limit set by the user and two less than the number of CPUs on the machine.
        with mp.Pool(processes=min(args.max_processes, mp.cpu_count() - 2)) as pool:
            # imap creates an iterator so we don't exhaust the machine's memory
            # (as opposed to map). imap_unordered is slightly faster than imap.
            res = [x for x in pool.imap_unordered(multi_process_micrograph, params)]  

    else: # Using GPU.
        # Calculate the memory usage of the GPU by a single process on one micrograph.
        mem_usage = get_mempool_usage(params[0], args.gpus[0])    
        print("Preprocess finished. Picking particles...")
        
        # Calculate the number of processes to run on each GPU, and partition
        # the micrographs into batches to be passed to each GPU.
        procs_per_gpu = calc_procs_per_gpu(mem_usage, args.max_processes, args.gpus)
        batches = get_mrc_batches(params, procs_per_gpu)
        if not picker.verbose: # Display simple progress bar.
            p = mp.Process(target=progress_bar, args=[picker.output_particles / "star", len(mrc_files)], name="KLTPicker_ProgressBar")
            p.start()
        # We have multiple processes writing results to the same "res" object,
        # so we need a manager (in the version without GPU the pool function
        # takes care of this).
        manager = mp.Manager()
        res = manager.list()
        # Distribute the batches of micrographs to different processes. Each
        # process runs a pool of workers on its own GPU. The size of each 
        # worker pool is according to procs_per_gpu.
        jobs = []
        for i in procs_per_gpu:
            if procs_per_gpu[i]:
                p = mp.Process(target=multi_process_micrograph_pool, args=[i, int(procs_per_gpu[i]), batches[i], res], name="KLTPicker%d"%i)
                jobs.append(p)
                p.start()       
        for proc in jobs:
            proc.join()
    
    # Write summary file and print summary to user.
    num_files = len(res)
    num_particles = sum([row[1] for row in res])
    num_noise = sum([row[2] for row in res]) 
    print("Picked %d particles and %d noise images out of %d micrographs." %(num_particles, num_noise, num_files))