from tensorflow import keras
from keras import layers
import numpy as np
import tensorflow as tf

def VerySimpleModel():
    input_layer = layers.Input(shape=(256,256,3))
    cout1 = layers.Conv2D(filters=16, kernel_size=3, strides=1, padding='same')(input_layer)
    pout1 = layers.MaxPool2D(pool_size=(2,2), strides=2, padding='same')(cout1)
    
    cout2 = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(pout1)
    pout2 = layers.MaxPool2D(pool_size=(2,2), strides=2, padding='same')(cout2)
    
    cout3 = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(pout2)
    pout3 = layers.MaxPool2D(pool_size=(2,2), strides=2, padding='same')(cout3)

    tout1 = layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', output_padding=1)(pout3)

    concat1 = layers.Concatenate(axis=3)([tout1, cout3])
    tout2 = layers.Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding='same', output_padding=1)(concat1)

    concat2 = layers.Concatenate(axis=3)([tout2, cout2])
    tout3 = layers.Conv2DTranspose(filters=1, kernel_size=3, strides=2, padding='same', output_padding=1)(concat2)


    
    out = tout3

    model = keras.models.Model(inputs=input_layer, outputs=out)
    return model


    # out = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(out)
    # out = layers.MaxPool2D(pool_size=(2,2), strides=2, padding='same')(out)
    # out = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(out)
    # out = layers.MaxPool2D(pool_size=(2,2), strides=2, padding='same')(out)
    # out = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(out)
    # out = layers.MaxPool2D(pool_size=(2,2), strides=2, padding='same')(out)
    # out = layers.Conv2D(filters=16, kernel_size=3, strides=1, padding='same')(out)
    # out = layers.MaxPool2D(pool_size=(2,2), strides=2, padding='same')(out)
