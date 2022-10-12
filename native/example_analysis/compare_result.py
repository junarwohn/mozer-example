import numpy as np

a = np.load("./vco_unet_q0.npy")
b = np.load("./vco_unet_q2.npy")

print(np.sum(a.flatten() == b.flatten()) / len(a.flatten()))
