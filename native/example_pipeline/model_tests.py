import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"
from tensorflow.keras.applications import MobileNet
import tvm
import tvm.relay as relay
from tvm.contrib.download import download_testdata
import numpy as np
import cv2
import time
# For silent execution

img_rows,img_cols = 224,224
# net = MobileNet(weights='imagenet', input_shape=(img_rows, img_cols, 3))

image_path = "/home/jd/workspace/mozer-example/native/example_pipeline/treefrog.jpg"
resized_image = cv2.resize(cv2.imread(image_path), (224, 224))

########################################

def preprocess_input(resized_image):
    image_data = np.asarray(resized_image).astype("float32")
    image_data = np.expand_dims(image_data, axis=0)
    image_data[:, :, :, 0] = 2.0 / 255.0 * image_data[:, :, :, 0] - 1
    image_data[:, :, :, 1] = 2.0 / 255.0 * image_data[:, :, :, 1] - 1
    image_data[:, :, :, 2] = 2.0 / 255.0 * image_data[:, :, :, 2] - 1
    return image_data

########################################

def check_model(net, input_data):
    print("---------------------------------------------------")
    warm_up_iteration = 50
    iteration = 100
    
    print("{}, layers : {}".format(net.name, len(net.layers)))
    
    # Warm Up Phase - 50 iteration
    for _ in range(warm_up_iteration):
        net(input_data)

    now = time.time()
    for _ in range(iteration):
        net(input_data)
    running_time = time.time() - now

    print("tf native running_time", running_time)

    # Preprocess for TVM model
    input_data = input_data.transpose([0, 3, 1, 2])
    shape_dict = {"input_1": input_data.shape}
    mod, params = relay.frontend.from_keras(net, shape_dict)

    # TVM GPU(CUDA) inference 
    target = 'cuda'
    dev = tvm.cuda(0)

    tvm.relay.backend.te_compiler.get().clear()
    with tvm.transform.PassContext(opt_level=4):
        lib = relay.build(mod, target, params=params)

    total_model = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

    # Warm Up Phase - 50 iteration
    for _ in range(warm_up_iteration):
        total_model.set_input('input_1', input_data)
        total_model.run()
        total_model.get_output(0).numpy()

    # 100 iteration
    now = time.time()
    for _ in range(iteration):
        total_model.set_input('input_1', input_data)
        total_model.run()
        total_model.get_output(0).numpy()
    running_time = time.time() - now

    print("GPU(CUDA) tvm running_time", running_time)
    del total_model

    # TVM CPU(LLVM) inference

    target = 'llvm'
    dev = tvm.cpu(0)

    tvm.relay.backend.te_compiler.get().clear()
    with tvm.transform.PassContext(opt_level=4):
        lib = relay.build(mod, target, params=params)

    total_model = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

    # Warm Up Phase - 50 iteration
    for _ in range(warm_up_iteration):
        total_model.set_input('input_1', input_data)
        total_model.run()
        total_model.get_output(0).numpy()

    # 100 iteration
    now = time.time()
    for _ in range(iteration):
        total_model.set_input('input_1', input_data)
        total_model.run()
        total_model.get_output(0).numpy()
    running_time = time.time() - now

    print("CPU(LLVM) tvm running_time", running_time)
    del total_model
    print("---------------------------------------------------")

########################################

def visualize_model(net):
    tf.keras.utils.plot_model(net, show_shapes=True, show_layer_names=True, to_file='{}.png'.format(net.name))

########################################

label_file_url = "".join(
    [
        "https://raw.githubusercontent.com/leferrad/tensorflow-mobilenet/master/imagenet/labels.txt",
    ]
)
label_file = "labels_mobilenet_quant_v1_224.txt"
label_path = download_testdata(label_file_url, label_file, module="data")

# List of 1001 classes
with open(label_path) as f:
    labels = f.readlines()

########################################

image_data = preprocess_input(resized_image)

# result = net(image_data)

# Convert result to 1D data
# predictions = np.squeeze(result)

# Get top 1 prediction
# prediction = np.argmax(predictions)

# predictionss = np.argsort(predictions)[::-1][:3]

# print(*[labels[e].strip('\n') for e in predictionss], sep=' | ')

########################################

# Resnet 50
net = tf.keras.applications.ResNet50(weights='imagenet', input_shape=(img_rows, img_cols, 3))
# check_model(net, image_data)
visualize_model(net)
tf.keras.backend.clear_session()

# Resnet 101
net = tf.keras.applications.ResNet101(weights='imagenet', input_shape=(img_rows, img_cols, 3))
# check_model(net, image_data)
visualize_model(net)
tf.keras.backend.clear_session()

# Resnet 152
net = tf.keras.applications.ResNet152(weights='imagenet', input_shape=(img_rows, img_cols, 3))
# check_model(net, image_data)
visualize_model(net)
tf.keras.backend.clear_session()

# MobileNet
net = tf.keras.applications.mobilenet.MobileNet(weights='imagenet', input_shape=(img_rows, img_cols, 3))
# check_model(net, image_data)
visualize_model(net)
tf.keras.backend.clear_session()

# # MobileNet V2
# net = tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet', input_shape=(img_rows, img_cols, 3))
# check_model(net, image_data)
# tf.keras.backend.clear_session()

# # MobileNet V3 Small
# net = tf.keras.applications.MobileNetV3Small(weights='imagenet', input_shape=(img_rows, img_cols, 3), include_preprocessing=False)
# check_model(net, image_data)
# tf.keras.backend.clear_session()

# # MobileNet V3 Large
# net = tf.keras.applications.MobileNetV3Large(weights='imagenet', input_shape=(img_rows, img_cols, 3), include_preprocessing=False)
# check_model(net, image_data)
# tf.keras.backend.clear_session()

# VGG16
net = tf.keras.applications.VGG16(weights='imagenet', input_shape=(img_rows, img_cols, 3))
# check_model(net, image_data)
visualize_model(net)
tf.keras.backend.clear_session()

# VGG19
net = tf.keras.applications.VGG19(weights='imagenet', input_shape=(img_rows, img_cols, 3))
# check_model(net, image_data)
visualize_model(net)
tf.keras.backend.clear_session()

# # EfficientNet 
# net = tf.keras.applications.efficientnet.EfficientNetB0(weights='imagenet', input_shape=(img_rows, img_cols, 3))
# check_model(net, image_data)
# tf.keras.backend.clear_session()

# # EfficientNet V2
# net = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(weights='imagenet', input_shape=(img_rows, img_cols, 3))
# check_model(net, image_data)
# tf.keras.backend.clear_session()
