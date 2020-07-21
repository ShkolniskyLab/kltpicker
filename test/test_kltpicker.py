import numpy as np
from kltpicker import KLTPicker
import argparse
from kltpicker import Micrograph
import mrcfile
from kltpicker.cryo_utils import downsample
from pathlib import Path
from datetime import datetime


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
