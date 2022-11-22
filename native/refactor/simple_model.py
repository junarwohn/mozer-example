from tensorflow import keras
from keras import layers
import numpy as np
import tensorflow as tf

def VerySimpleModel():
    input_layer = layers.Input(shape=(256,256,3))
    cout1 = layers.Conv2D(filters=16, kernel_size=3, strides=1, padding='same')(input_layer)
    pout1 = layers.MaxPool2D(pool_size=(2,2), strides=2, padding='same')(cout1)

    out = pout1

    model = keras.models.Model(inputs=input_layer, outputs=out)
    return model


if __name__ == '__main__':
    model = VerySimpleModel()
    model.compile()
    model.save("simple_model.h5")
