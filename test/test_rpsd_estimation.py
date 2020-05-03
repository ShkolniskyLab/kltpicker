import numpy as np
from kltpicker import KLTPicker
import argparse


parser = argparse.ArgumentParser()
args = parser.parse_args()
args.particle_size = 300
args.input_dir = "c:\\users\\dalit\\Documents\\uni\\year3\\shkol_work\\mrcs"
args.output_dir = "c:\\users\\dalit\\Documents\\uni\\year3\\shkol_work\\out"
args.gpu_use = 0
args.max_order = 100
args.num_of_particles = -1
args.num_of_noise_images =0
args.threshold = 0
args.max_iter = 6 * (10 ** 4)


def comp(a,b):
    return np.linalg.norm(a-b)/np.linalg.norm(a)


if __name__ == "__main__":
    picker = KLTPicker(args)
    picker.get_micrographs()
    for micrograph in picker.micrographs:
        micrograph.cutoff_filter(picker.patch_size)
        micrograph.estimate_rpsd(picker.patch_size, picker.max_iter)
        print("done rpsd")
