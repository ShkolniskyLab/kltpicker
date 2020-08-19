from kltpicker.cryo_utils import cryo_epsds_cp
import cupy as cp

block = cp.random.random((79, 79))
patch_size = 79
for i in range(10):
    psd_block = cryo_epsds_cp(block[:, :, cp.newaxis], 
                           cp.where(cp.zeros((int(patch_size), int(patch_size))) == 0),
                           int(cp.floor(0.3 * patch_size)))

print("done")