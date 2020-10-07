from pathlib import Path
import sys
import progressbar
import subprocess
import time 
import argparse
import os
import re
import random
try:
    import cupy
    num_gpus = cupy.cuda.runtime.getDeviceCount()
except:
    pass

def parse_args(has_cupy):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-dir', help='Full path of input directory.')
    parser.add_argument('-o', '--output-dir', help='Full path of output directory.', type=check_dir_exists)
    parser.add_argument('-s', '--particle-size', help='Expected size of particles in pixels.', type=check_positive_int)
    parser.add_argument('-p', '--num-particles',
                        help='Number of particles to pick per micrograph. If set to -1 will pick all particles.',
                        default=-1, type=check_positive_int_or_all)
    parser.add_argument('-n', '--num-noise', help='Number of noise images to pick per micrograph.',
                        default=0, type=check_positive_int_or_zero)
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose. Choose this to display number of particles and noise images picked from each micrograph during runtime. Otherwise, you get a simple progress bar.', default=False)
    parser.add_argument('--max-processes', help='Limit the number of concurrent processes to run. -1 to let the program choose.', type=check_positive_int_or_all, default=-1)
    parser.add_argument('--only-do-unfinished', help='Only pick micrographs for which no coordinate file exists in the output directory.', action='store_true', default=False)
    if has_cupy:
        parser.add_argument('--no-gpu', action='store_true', help="Don't use GPUs.", default=False)
        parser.add_argument('--gpus', help='Indices of GPUs to be used. Valid indices: 0,...,%d. Enter -1 to use all available GPUS.'%(num_gpus-1), default=[-1], nargs='+', type=check_range_gpu)
        args = parser.parse_args()
        if args.gpus == [-1]:
            args.gpus = list(range(num_gpus))
        else: 
            args.gpus = [x for x in args.gpus if x in range(num_gpus)]
    else:
        args = parser.parse_args()
        args.no_gpu = 1
        args.gpus = []
    return args

def check_positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue

def check_positive_int_or_zero(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError("%s is an invalid non-negative int value" % value)
    return ivalue

def check_positive_int_or_all(value):
    ivalue = int(value)
    if ivalue == -1:
        return ivalue
    elif ivalue <= 0 and ivalue != -1:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue    

def check_range_gpu(value):
    ivalue = int(value)
    if ivalue == -1:
        return ivalue
    elif ivalue > num_gpus - 1 or ivalue < 0:
        raise argparse.ArgumentTypeError("%s is not in range 0-%s" % (value, num_gpus-1))
    return ivalue

def check_dir_exists(value):
    output_dir = Path(value)
    if output_dir.is_file():
        raise argparse.ArgumentTypeError("There is already a file with the name %s." %value)
    elif not output_dir.exists():
        raise argparse.ArgumentTypeError('Directory %s does not exist. Please specify an existing directory.' % output_dir)
    return value    

def get_args(has_cupy):
    print("\n")
    while True:
        input_dir = Path(input('Enter full path of micrographs MRC files: '))
        num_files = len(list(input_dir.glob("*.mrc")))
        if num_files > 0:
            print("Found %i MRC files." % len(list(input_dir.glob("*.mrc"))))
            break
        elif not input_dir.is_dir():
            print("%s is not a directory." % input_dir)
        else:
            print("Could not find any files in %s." % input_dir)
    
    while True:
        particle_size = input('Enter the particle size in pixels: ')
        try:
            particle_size = int(particle_size)
            if particle_size < 1:
                print("Particle size must be a positive integer.")
            else:
                break
        except ValueError:
            print("Particle size must be a positive integer.")
    
    only_do_unfinished = False
    while True:
        output_path = input('Enter full path of output directory: ')
        output_dir = Path(output_path)
        if output_dir.is_file():
            print("There is already a file with the name you specified. Please specify a directory.")
        elif not output_path:
            print("Please specify a directory.")
        elif output_dir.parent.exists() and not output_dir.exists():
            while True:
                create_dir = input('Output directory does not exist. Create? (Y/N): ')
                if create_dir.strip().lower().startswith('y'):
                    Path.mkdir(output_dir)
                    break
                elif create_dir.strip().lower().startswith('n'):
                    print("OK, aborting...")
                    sys.exit(0)
                else:
                    print("Please choose Y/N.") 
            break
        elif not output_dir.parent.exists():
            print('Parent directory %s does not exist. Please specify an existing directory.' % output_dir.parent)
        elif output_dir.is_dir():
            num_finished = check_output_dir(input_dir, output_dir, particle_size)
            if num_finished == 1:
                break
            elif num_finished == 2:
                print("The directory you specified contains coordinate files for all of the micrographs in the input directory. Aborting...")
                sys.exit()
            else:
                while True:
                    only_do_unfinished = input("The directory you specified contains coordinate files for some of the micrographs in the input directory. Run only on micrographs which have no coordinate file? (Y/N): ")
                    if only_do_unfinished.strip().lower().startswith('y'):
                        only_do_unfinished = True
                        break
                    elif only_do_unfinished.strip().lower().startswith('n'):
                        print("OK, aborting...")
                        sys.exit(0)
                    else:
                        print("Please choose Y/N.") 
                break
            break
        
            
    num_particles_to_pick = 0
    while num_particles_to_pick == 0:
        pick_all = input('Pick all particles? (Y/N): ')
        if pick_all.strip().lower().startswith('y'):
            num_particles_to_pick = -1
        elif pick_all.strip().lower().startswith('n'):
            while True:
                num_particles_to_pick = input('How many particles to pick: ')
                try:
                    num_particles_to_pick = int(num_particles_to_pick)
                    if num_particles_to_pick < 1:
                        print("Number of particles to pick must be a positive integer.")
                    else:
                        break
                except ValueError:
                    print("Number of particles to pick must be a positive integer.")
        else:
            print("Please choose Y/N.")

    num_noise_to_pick = -1
    while num_noise_to_pick == -1:
        pick_noise = input('Pick noise images? (Y/N): ')
        if pick_noise.strip().lower().startswith('n'):
            num_noise_to_pick = 0
        elif pick_noise.strip().lower().startswith('y'):
            while True:
                num_noise_to_pick = input('How many noise images to pick: ')
                try:
                    num_noise_to_pick = int(num_noise_to_pick)
                    if num_noise_to_pick < 1:
                        print("Number of noise images to pick must be a positive integer.")
                    else:
                        break
                except ValueError:
                    print("Number of particles to pick must be a positive integer.")
        else:
            print("Please choose Y/N.")
    
    verbose=0
    while verbose == 0:
        verbose_in = input('Display detailed progress? (Y/N): ')
        if verbose_in.strip().lower().startswith('y'):
            verbose = True
        elif verbose_in.strip().lower().startswith('n'):
            verbose = False
            break
        else:
            print("Please choose Y/N.")
            
    max_processes = -1
    while True:
        max_processes_in = input('Enter maximum number of concurrent processes (-1 to let the program decide): ')
        try:
            max_processes = int(max_processes_in)
            if max_processes < 1 and max_processes != -1:    
                print("Maximum number of concurrent processes must be a positive integer (except -1 to let the program decide).")
            else:
                break
        except ValueError:
            print("Maximum number of concurrent processes must be a positive integer (except -1 to let the program decide).")
        
    if has_cupy:
        no_gpu = 0
        gpu_indices = []
        while no_gpu == 0:
            no_gpu_in = input('Use GPU? (Y/N): ')
            if no_gpu_in.strip().lower().startswith('n'):
                no_gpu = 1
            elif no_gpu_in.strip().lower().startswith('y'):
                no_gpu == 0
                while gpu_indices == []:
                    gpu_indices_in = input('Which GPUs would you like to use? (Valid indices: 0,...,%d. Enter -1 to use all): '%(num_gpus-1))
                    if gpu_indices_in.strip() == '-1':
                        gpu_indices = list(range(num_gpus))
                        break
                    else:
                        gpu_indices_split = re.split(',| ', gpu_indices_in)
                        for gpu_index in gpu_indices_split:
                            try:
                                gpu_index = int(gpu_index)
                                if gpu_index in range(num_gpus):
                                    gpu_indices.append(gpu_index)
                            except ValueError:
                                pass
                        gpu_indices = list(set(gpu_indices))
                        if gpu_indices:
                            break
                        else:
                            print("Please specify valid GPU indices, separated by whitespaces or commas.")
                break
            else:
                print("Please choose Y/N.")
        
    else:
        no_gpu = 1
        gpu_indices = []
    return input_dir, output_dir, particle_size, num_particles_to_pick, num_noise_to_pick, no_gpu, gpu_indices, verbose, max_processes, only_do_unfinished

def check_output_dir(input_dir, output_dir, particle_size):
    """
    Checks how many coordinate files there are in the output directory with
    names matching the micrographs in the input directory, if any.
    If there are no micrographs in input_dir, return 0.
    If the intersection between coordinate file names and micrograph file 
    names is empty, return 1.
    If all micrographs have an output coordinate file, return 2.
    Else, return list of micrograph names which do not have an output file.
    """
    mrcs = [mrc for mrc in input_dir.glob("*.mrc")]
    if mrcs == []:
        return 0
    mrc_names = [mrc.name[:-4] for mrc in mrcs]
    output_particles_box_path = output_dir / ("pickedParticlesParticleSize%i/box"%particle_size)
    output = [f for f in output_particles_box_path.glob("*.box")]
    output_names = [f.name[:-4] for f in output]
    intersection = list(set(output_names) & set(mrc_names))
    if len(intersection) == 0:
        return 1
    elif len(intersection) == len(mrc_names):
        return 2
    else:
        unfinished = list(set(mrc_names) - set(output_names))
        return [mrc for mrc in mrcs if mrc.name[:-4] in unfinished]

def progress_bar(output_dir, num_mrcs):
    """
    Progress bar function that reports the progress of the program, by 
    periodically checking how many output files have been written. Shows both
    percentage completed and time elapsed.
    """
    start_time = get_start_time(output_dir)
    num_finished = check_num_finished(output_dir, start_time)
    bar = progressbar.ProgressBar(maxval=num_mrcs, widgets=["[", progressbar.Timer(), "] ", progressbar.Bar('#', '|', '|'), ' (', progressbar.Percentage(), ')'])
    bar.start()
    while num_finished < num_mrcs:
        num_finished = check_num_finished(output_dir, start_time)
        bar.update(num_finished)
        time.sleep(1)
    bar.finish()
    print("Finished successfully!")

def check_num_finished(output_dir, start_time):
    finished = [f for f in output_dir.glob("*.star") if os.path.getmtime(str(f)) > start_time]
    num_finished = len(finished)
    return num_finished

def get_start_time(output_dir):
    """
    For some reason time.time() and getmtime of a file do not appear to be 
    calculated in the same timezone. So we get the start time by checking the 
    modification time of a file we create (and immediately delete). A bit ugly,
    but works.
    """
    fp = output_dir / ('%010x' % random.randrange(16**10))
    with fp.open("w") as f:
        f.write("hi")
    start_time = os.path.getmtime(str(fp))
    os.remove(str(fp))
    return start_time
    
def check_for_newer_version():
    """
    This function checks whether there is a newer version of kltpicker 
    available on PyPI. If there is, it issues a warning.

    """
    name = 'kltpicker'
    # Use pip to try and install a version of kltpicker which does not exist.
    # In answer, you get all available versions. Find the newest one.
    latest_version = str(subprocess.run([sys.executable, '-m', 'pip', 'install', '%s==random' %name], capture_output=True, text=True))
    latest_version = latest_version[latest_version.find('(from versions:')+15:]
    latest_version = latest_version[:latest_version.find(')')]
    latest_version = latest_version.replace(' ','').split(',')[-1]
    
    if latest_version == 'none': # Got an unexpected response.
        pass
    else: # Use pip to determine the installed version.
        current_version = str(subprocess.run([sys.executable, '-m', 'pip', 'show', name], capture_output=True, text=True))
        current_version = current_version[current_version.find('Version:')+8:]
        current_version = current_version[:current_version.find('\\n')].replace(' ','') 
        if latest_version != current_version:
            print("NOTE: you are running an old version of %s (%s). A newer version (%s) is available, please upgrade."%(name, current_version, latest_version))
