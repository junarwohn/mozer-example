import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tvm.contrib.download import download_testdata
import numpy as np
import cv2
import time
# For silent execution
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"

img_rows,img_cols = 224,224
net = MobileNet(weights='imagenet', input_shape=(img_rows, img_cols, 3))

image_path = "/home/jd/workspace/mozer-example/native/example_pipeline/treefrog.jpg"
resized_image = cv2.resize(cv2.imread(image_path), [224, 224])

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
    iteration = 100
    print("{}, layers : {}".format(net.name, len(net.layers)))
    
    # Warm Up Phase
    for _ in range(50):
        net(input_data)

    now = time.time()
    for _ in range(iteration):
        net(input_data)
    running_time = time.time() - now

    print("running_time", running_time)

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

result = net(image_data)

# Convert result to 1D data
predictions = np.squeeze(result)

# Get top 1 prediction
prediction = np.argmax(predictions)

predictionss = np.argsort(predictions)[::-1][:3]

print(*[labels[e].strip('\n') for e in predictionss], sep=' | ')

########################################

# Resnet 50
net = tf.keras.applications.ResNet50(weights='imagenet', input_shape=(img_rows, img_cols, 3))
check_model(net, image_data)

# Resnet 101
net = tf.keras.applications.ResNet101(weights='imagenet', input_shape=(img_rows, img_cols, 3))
check_model(net, image_data)

# Resnet 152
net = tf.keras.applications.ResNet152(weights='imagenet', input_shape=(img_rows, img_cols, 3))
check_model(net, image_data)

# MobileNet
net = tf.keras.applications.mobilenet.MobileNet(weights='imagenet', input_shape=(img_rows, img_cols, 3))
check_model(net, image_data)

# MobileNet V2
net = tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet', input_shape=(img_rows, img_cols, 3))
check_model(net, image_data)

# MobileNet V3 Small
net = tf.keras.applications.MobileNetV3Small(weights='imagenet', input_shape=(img_rows, img_cols, 3))
check_model(net, image_data)

# MobileNet V3 Small
net = tf.keras.applications.MobileNetV3Small(weights='imagenet', input_shape=(img_rows, img_cols, 3))
check_model(net, image_data)

# VGG16
net = tf.keras.applications.VGG16(weights='imagenet', input_shape=(img_rows, img_cols, 3))
check_model(net, image_data)

# VGG19
net = tf.keras.applications.VGG19(weights='imagenet', input_shape=(img_rows, img_cols, 3))
check_model(net, image_data)

# EfficientNet 
net = tf.keras.applications.efficientnet.EfficientNetB0(weights='imagenet', input_shape=(img_rows, img_cols, 3))
check_model(net, image_data)

# EfficientNet V2
net = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(weights='imagenet', input_shape=(img_rows, img_cols, 3))
check_model(net, image_data)
