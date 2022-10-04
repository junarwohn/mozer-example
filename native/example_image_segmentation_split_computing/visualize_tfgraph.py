import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import plot_model
# from mozer.model_tuner.UNet_v2 import UNet
# from mozer.model_tuner.UNet_v1 import UNet
from mozer.model_tuner.UNet_v0 import UNet
from datetime import datetime

in_dim = 3
out_dim = 1
num_filter = 64
mutation = [1,1,0,0]

# unet = UNet(3, 1, 64, input_shape=(256,256), mutation=mutation)
unet = UNet(3, 1, 64)
unet.build(input_shape=(1,256,256,3))
unet.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

plot_model(unet, show_shapes=True, show_layer_names=True, to_file='model_{}_{}_{}_{}_{}.png'.format(stamp, *mutation))