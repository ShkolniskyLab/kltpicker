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
    picker.quad_ker = np.load("quad_ker.npy")
    picker.quad_nys = np.load("quad_nys.npy")
    picker.rho = np.load("rho.npy")
    picker.j_r_rho = np.load("j_r_rho.npy")
    picker.j_samp = np.load("j_samp.npy")
    picker.cosine = np.load("cosine.npy")
    picker.sine = np.load("sine.npy")
    picker.rsamp_r = np.load("rsamp_r.npy")
    picker.r_r = np.load("r_r.npy")
    picker.rad_mat = np.load("rad_mat.npy")
    picker.rsamp_length = 1521
    for micrograph in picker.micrographs:
        micrograph.psd = np.load("psd.npy")
        micrograph.approx_noise_psd = np.load("approx_noise_psd.npy")
        micrograph.approx_clean_psd = np.load("approx_clean_psd.npy")
        micrograph.approx_noise_var = np.load("approx_noise_var.npy")
        micrograph.noise_mc = np.load("noise_mc.npy")
        micrograph.r = np.load("r.npy")
        print("starting")
        micrograph.construct_klt_templates(picker)
        print("done constructing klt templates")
