from pathlib import Path
from sys import exit


def get_args():
    while True:
        input_dir = Path(input('Enter full path of micrographs MRC files:\n'))
        num_files = len(list(input_dir.glob("*.mrc")))
        if num_files > 0:
            print("Found %i MRC files." % len(list(input_dir.glob("*.mrc"))))
            break
        elif not input_dir.is_dir():
            print("%s is not a directory." % input_dir)
        else:
            print("Could not find any files in %s." % input_dir)

    while True:
        output_dir = Path(input('Enter full path of output directory:\n'))
        if output_dir.is_file():
            print("There is already a file with the name you specified. Please specify a directory.")
        elif output_dir.parent.exists() and not output_dir.exists():
            create_dir = input('Output directory does not exist. Create? (Y/N):')
            if create_dir.strip().lower()[0] == 'y':
                Path.mkdir(output_dir)
                break
            else:
                print("OK, exiting...")
                exit(0)
        elif not output_dir.parent.exists():
            print('Parent directory %s does not exist. Please specify an existing directory.' % output_dir.parent)
        else:
            break

    while True:
        particle_size = input('Enter the particle size in pixels:\n')
        try:
            particle_size = int(particle_size)
            if particle_size < 1:
                print("Particle size must be a positive integer.")
            else:
                break
        except ValueError:
            print("Particle size must be a positive integer.")

    num_particles_to_pick = 0
    while num_particles_to_pick == 0:
        pick_all = input('Pick all particles? (Y/N):\n')
        if pick_all.strip().lower()[0] == 'y':
            num_particles_to_pick = -1
        elif pick_all.strip().lower()[0] == 'n':
            while True:
                num_particles_to_pick = input('How many particles to pick:\n')
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
        pick_noise = input('Pick noise images? (Y/N):\n')
        if pick_noise.strip().lower()[0] == 'n':
            num_noise_to_pick = 0
        elif pick_noise.strip().lower()[0] == 'y':
            while True:
                num_noise_to_pick = input('How many noise images to pick:\n')
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
    return input_dir, output_dir, particle_size, num_particles_to_pick, num_noise_to_pick
