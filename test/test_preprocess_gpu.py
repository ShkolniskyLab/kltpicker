import cupy as np
from kltpicker import KLTPicker
import argparse

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


def comp(a,b):
    return np.linalg.norm(a-b)/np.linalg.norm(a)


if __name__ == "__main__":
    picker = KLTPicker(args)
    print("running preprocess")
    picker.preprocess()

    quad_ker = np.load("quad_ker.npy")
    quad_nys = np.load("quad_nys.npy")
    rho = np.load("rho.npy")
    j_r_rho = np.load("j_r_rho.npy")
    j_samp = np.load("j_samp.npy")
    cosine = np.load("cosine.npy")
    sine = np.load("sine.npy")
    rsamp_r = np.load("rsamp_r.npy")
    r_r = np.load("r_r.npy")
    rad_mat = np.load("rad_mat.npy")

    print("quad_ker: %f" %comp(quad_ker, picker.quad_ker))
    print("quad_nys: %f" %comp(quad_nys, picker.quad_nys))
    print("rho: %f" %comp(rho, picker.rho))
    print("j_r_rho: %f" %comp(j_r_rho, picker.j_r_rho))
    print("j_samp: %f" %comp(j_samp, picker.j_samp))
    print("cosine: %f" %comp(cosine, picker.cosine))
    print("sine: %f" %comp(sine, picker.sine))
    print("rsamp_r: %f" %comp(rsamp_r, picker.rsamp_r))
    print("r_r: %f" %comp(r_r, picker.r_r))
    print("rad_mat: %f" %comp(rad_mat, picker.rad_mat))

