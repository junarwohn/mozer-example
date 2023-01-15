# single gpu multiple model

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"
import tvm
import tvm.relay as relay
import cv2
import numpy as np
import time

def check_model(net, input_data):
    warm_up_iteration = 100
    iteration = 1000

    # Preprocess for TVM model
    input_data = input_data.transpose([0, 3, 1, 2])
    shape_dict = {"input_1": input_data.shape}
    mod, params = relay.frontend.from_keras(net, shape_dict)

    # TVM GPU(CUDA) inference 
    target = 'cuda'
    dev = tvm.cuda(0)

    with tvm.transform.PassContext(opt_level=4):
        lib0 = relay.build(mod, target, params=params)
        lib1 = relay.build(mod, target, params=params)

    model0 =  tvm.contrib.graph_executor.GraphModule(lib0["default"](dev))

    # Warm Up Phase
    for _ in range(warm_up_iteration):
        model0.set_input('input_1', input_data)
        model0.run()
        model0.get_output(0).numpy()

    # iteration
    now = time.time()
    for _ in range(iteration):
        model0.set_input('input_1', input_data)
        model0.run()
        model0.get_output(0).numpy()
    running_time = time.time() - now

    print("single {} running_time {}".format(net.name, running_time))

    model1 =  tvm.contrib.graph_executor.GraphModule(lib1["default"](dev))


    # Warm Up Phase
    for _ in range(warm_up_iteration):
        model0.set_input('input_1', input_data)
        model1.set_input('input_1', input_data)
        model0.run()
        model1.run()
        model0.get_output(0).numpy()
        model1.get_output(0).numpy()

    # iteration
    now = time.time()
    for _ in range(iteration):
        model0.set_input('input_1', input_data)
        model1.set_input('input_1', input_data)
        model0.run()
        model1.run()
        model0.get_output(0).numpy()
        model1.get_output(0).numpy()
    running_time = time.time() - now

    print("dual {} running_time {}".format(net.name, running_time))

def preprocess_input(resized_image):
    image_data = np.asarray(resized_image).astype("float32")
    image_data = np.expand_dims(image_data, axis=0)
    image_data[:, :, :, 0] = 2.0 / 255.0 * image_data[:, :, :, 0] - 1
    image_data[:, :, :, 1] = 2.0 / 255.0 * image_data[:, :, :, 1] - 1
    image_data[:, :, :, 2] = 2.0 / 255.0 * image_data[:, :, :, 2] - 1
    return image_data

img_rows,img_cols = 224,224
image_path = "/home/jd/workspace/mozer-example/native/example_pipeline/treefrog.jpg"
resized_image = cv2.resize(cv2.imread(image_path), (224, 224))
image_data = preprocess_input(resized_image)

net = tf.keras.applications.ResNet152(weights='imagenet', input_shape=(img_rows, img_cols, 3))

check_model(net, image_data)
