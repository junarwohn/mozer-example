import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"
import tvm
from tvm import te
import tvm.relay as relay
from tvm.contrib.download import download_testdata
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input

weights_url = "".join(
    [
        " https://storage.googleapis.com/tensorflow/keras-applications/",
        "resnet/resnet50_weights_tf_dim_ordering_tf_kernels.h5",
    ]
)
weights_file = "resnet50_keras_new.h5"


weights_path = download_testdata(weights_url, weights_file, module="keras")
keras_resnet50 = tf.keras.applications.resnet50.ResNet50(
    include_top=True, weights=None, input_shape=(224, 224, 3), classes=1000
)
keras_resnet50.load_weights(weights_path)

img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
img_path = download_testdata(img_url, "cat.png", module="data")
img = Image.open(img_path).resize((224, 224))

data = np.array(img)[np.newaxis, :].astype("float32")
data = preprocess_input(data).transpose([0, 3, 1, 2])
print("input_1", data.shape)


shape_dict = {"input_1": data.shape}
mod, params = relay.frontend.from_keras(keras_resnet50, shape_dict)
# compile the model
target = "cuda"
dev = tvm.cuda(0)



# TODO(mbs): opt_level=3 causes nn.contrib_conv2d_winograd_weight_transform
# to end up in the module which fails memory validation on cuda most likely
# due to a latent bug. Note that the pass context only has an effect within
# evaluate() and is not captured by create_executor().
with tvm.transform.PassContext(opt_level=3):
    model = relay.build_module.create_executor("graph", mod, dev, target, params).evaluate()

dtype = "float32"
tvm_out = model(tvm.nd.array(data.astype(dtype)))
top1_tvm = np.argmax(tvm_out.numpy()[0])

synset_url = "".join(
    [
        "https://gist.githubusercontent.com/zhreshold/",
        "4d0b62f3d01426887599d4f7ede23ee5/raw/",
        "596b27d23537e5a1b5751d2b0481ef172f58b539/",
        "imagenet1000_clsid_to_human.txt",
    ]
)
synset_name = "imagenet1000_clsid_to_human.txt"
synset_path = download_testdata(synset_url, synset_name, module="data")
with open(synset_path) as f:
    synset = eval(f.read())
print("Relay top-1 id: {}, class name: {}".format(top1_tvm, synset[top1_tvm]))
# confirm correctness with keras output
keras_out = keras_resnet50.predict(data.transpose([0, 2, 3, 1]))
top1_keras = np.argmax(keras_out)
print("Keras top-1 id: {}, class name: {}".format(top1_keras, synset[top1_keras]))