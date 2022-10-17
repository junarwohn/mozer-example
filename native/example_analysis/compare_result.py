import numpy as np

a = np.load("./vco_unet_q0.npy")
b = np.load("./vco_unet_q2.npy")
print(np.sum(a.flatten() == b.flatten()) / len(a.flatten()))

a = np.load("./voc_resnet50_q0.npy")
b = np.load("./voc_resnet50_q2.npy")
print(np.sum(a.flatten() == b.flatten()) / len(a.flatten()))

a = np.load("./voc_resnet101_q0.npy")
b = np.load("./voc_resnet101_q2.npy")
print(np.sum(a.flatten() == b.flatten()) / len(a.flatten()))

a = np.load("./voc_resnet152_q0.npy")
b = np.load("./voc_resnet152_q2.npy")

print(np.sum(a.flatten() == b.flatten()) / len(a.flatten()))
