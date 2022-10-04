# For silent tensorflow build
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from mozer.model_tuner.UNet_v1 import UNet
# from mozer.model_tuner.UNet_v2 import UNet

in_dim = 3
out_dim = 1
num_filter = 64
mutation = [0,0,0,0]

unet = UNet(3, 1, 64, input_shape=(256,256), mutation=mutation)
unet.build(input_shape=(1,256,256,3))
print(unet.summary())