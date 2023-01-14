from multiprocessing import Process, Pipe
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"
from tvm.contrib.download import download_testdata
import tensorflow as tf
import tvm
import tvm.relay as relay
from tvm.contrib import graph_executor 
from PIL import Image
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
import time
from mozer.slicer.SlicingMachine import TVMSlicer
from tvm.relay.testing import run_opt_pass
from tvm.relay import transform, build_module

################################################

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--partition', '-p', type=int, default=25)
parser.add_argument('--batch_size', '-b', type=int, default=1)
args = parser.parse_args()

################################################
# import resnet152
if True:
# if False:
    weights_url = "".join(
        [
            " https://storage.googleapis.com/tensorflow/keras-applications/",
            "resnet/resnet152_weights_tf_dim_ordering_tf_kernels.h5",
        ]
    )
    weights_file = "resnet152_keras_new.h5"


    weights_path = download_testdata(weights_url, weights_file, module="keras")
    model_keras = tf.keras.applications.resnet.ResNet152(
        include_top=True, weights=None, input_shape=(224, 224, 3), classes=1000
    )
    model_keras.load_weights(weights_path)

################################################

img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
img_path = download_testdata(img_url, "cat.png", module="data")
img = Image.open(img_path).resize((224, 224))
# data = np.array(img)[np.newaxis, :].astype("float32")
data = np.array(img).astype("float32")
data = np.array([data for _ in range(args.batch_size)])
data = preprocess_input(data).transpose([0, 3, 1, 2])

################################################

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

################################################

shape_dict = {"input_1": data.shape}
mod, params = relay.frontend.from_keras(model_keras, shape_dict)



relay_slicer = TVMSlicer()
split_config = [{"op_name": "add", "op_index": args.partition}]
subgraphs, input_name_hints, output_name_hints = relay_slicer.slice_relay_graph(mod['main'], split_config, params)

# for idx, graph in enumerate(subgraphs):
#     ann = run_opt_pass(graph, transform.ToGraphNormalForm())
#     subgraphs[idx] =  tvm.IRModule.from_expr(ann)


target = 'cuda'
dev = tvm.cuda(0)

################################################

print("load 1")
# Partition 1
mod = subgraphs[0]
ann = run_opt_pass(mod, transform.ToGraphNormalForm())
mod = tvm.IRModule.from_expr(ann)['main']

with tvm.transform.PassContext(opt_level=4):
    lib = relay.build(mod, target, params=params)

model_1 = graph_executor.GraphModule(lib["default"](dev))

################################################

print("load 1")
# Partition 2
mod = subgraphs[1]
ann = run_opt_pass(mod, transform.ToGraphNormalForm())
mod = tvm.IRModule.from_expr(ann)['main']

with tvm.transform.PassContext(opt_level=4):
    lib = relay.build(mod, target, params=params)

model_2 = graph_executor.GraphModule(lib["default"](dev))

################################################


model_1.set_input("input_1", data)
model_1.run()
out_dict = dict()
for i, out_name in enumerate(output_name_hints[0]):
    out_dict[out_name] = model_1.get_output(i).numpy()

for k in out_dict:
    model_2.set_input(k, out_dict[k])

model_2.run()
partition_out = model_2.get_output(0).numpy()
top1_keras = np.argmax(partition_out)
print("Partition top-1 id: {}, class name: {}".format(top1_keras, synset[top1_keras]))
