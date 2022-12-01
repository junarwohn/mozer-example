from faulthandler import disable
from unittest import result
from mozer.slicer.SlicingMachine import TVMSlicer
from mozer.slicer.Quantize import quantize
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import tvm
import tvm.relay as relay
from tvm.contrib import graph_executor 
import numpy as np
import json
import pygraphviz as pgv
from argparse import ArgumentParser
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.dataflow_pattern import *

parser = ArgumentParser()
parser.add_argument('--partition_points', '-p', nargs='+', type=str, default=[], help='set partition points')
parser.add_argument('--img_size', '-i', type=int, default=256, help='set image size')
parser.add_argument('--model', '-m', type=str, default='unet', help='name of model')
parser.add_argument('--target', '-t', type=str, default='cuda', help='name of taget')
parser.add_argument('--device', '-d', type=str, default='2080ti', help='name of device')
parser.add_argument('--opt_level', '-o', type=int, default=3, help='set opt_level')
parser.add_argument('--build', '-b', type=int, default=0, help='build model')
parser.add_argument('--quantization_level', '-q', type=int, default=0, help='set quantization level')
parser.add_argument('--model_config', '-c', nargs='+', type=int, default=0, help='set partition point')
parser.add_argument('--jetson', '-j', type=int, default=0, help='jetson')
parser.add_argument('--base_path', type=str, default=os.environ['TS_DATA_PATH'] + "/model_tf/unet_v1/best/", help='path setting')
parser.add_argument('--tvm_build_path', type=str, default=os.environ['TS_DATA_PATH'] + "/model_tvm/unet_v1/best/", help='path setting')
args = parser.parse_args()

current_file_path = os.path.dirname(os.path.realpath(__file__)) + "/"
model_config = args.model_config
base_path = args.base_path
is_jetson = args.jetson

np.random.seed(0)
img_size = args.img_size

# set path
base_path = args.base_path
tvm_build_path = args.tvm_build_path

weights_url = "".join(
    [
        " https://storage.googleapis.com/tensorflow/keras-applications/",
        "resnet/resnet50_weights_tf_dim_ordering_tf_kernels.h5",
    ]
)
weights_file = "resnet50_keras_new.h5"


weights_path = download_testdata(weights_url, weights_file, module="keras")
model_keras = tf.keras.applications.resnet.ResNet50(
    include_top=True, weights=None, input_shape=(224, 224, 3), classes=1000
)
model_keras.load_weights(weights_path)

img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
img_path = download_testdata(img_url, "cat.png", module="data")
img = Image.open(img_path).resize((224, 224))

if is_jetson == 1:
    # target = tvm.target.Target("nvidia/jetson-nano")
    target = tvm.target.Target("nvidia/jetson-agx-xavier")
    # target = tvm.target.Target("nvidia/jetson-tx2")
    assert target.kind.name == "cuda"
    # assert target.attrs["arch"] == "sm_62"
    # target.attrs["arch"] = "sm_62"
    assert target.attrs["shared_memory_per_block"] == 49152
    assert target.attrs["max_threads_per_block"] == 1024
    assert target.attrs["thread_warp_size"] == 32
    # assert target.attrs["registers_per_block"] == 65536
else:
    if args.target == 'llvm':
        target = 'llvm'
        dev = tvm.cpu()
    elif args.target == 'cuda':
        # 1050ti
        # target = 'cuda -arch=sm_61'
        # dev = tvm.cuda(1)
        # 2080ti
        target = 'cuda -arch=sm_75'
        dev = tvm.device("cuda", 0)
    elif args.target == 'opencl':
        target = 'opencl'
        dev = tvm.opencl()

quantization_level = args.quantization_level

if target == 'llvm':
    disabled_pass={'AlterOpLayout'}
else :
    disabled_pass={}


quantization_level = args.quantization_level


model_info = {
    "name"   : "unet",
    "host"   : "x86_64",
    "target" : "cuda",
    "arch"   : "sm_75",
    "opt"    : "",
    "qaunt"  : "",
    "slice"  : "_".join([])
}
model_info_format = "{name}-{host}-{target}-{arch}-o{opt}-q{qaunt}-s{slice}".format(**model_info)


# TODO adding final_shape 
# do 'extra' job to 
json_format = '/'.join([tvm_build_path, model_info_format + ".json"])
with open(json_format, "r") as json_file:
    graph_json_raw = json.load(json_file)

tvm_slicer = TVMSlicer(graph_json_raw)

graph_info = graph_json_raw

# json format would be {model}_{target}_{img_size}_{opt_level}_{partition_start}-{partition_end}.json
partition_points = args.partition_points
if len(partition_points) > 4:
    partition_points = partition_points[:4]
    
for i in range(len(partition_points) - 1):
    # start_point = partition_points[i]
    # end_point = partition_points[i + 1]
    start_points = [int(i) + 1 for i in partition_points[i].split(',')]
    end_points =  [int(i) for i in partition_points[i + 1].split(',')]
    # graph_json, input_indexs, output_indexs = tvm_sxlicer.slice_graph(start_point + 1, end_point, is_quantize_sliced=True)
    graph_json, input_indexs, output_indexs = tvm_slicer.slice_graph(start_points, end_points, is_quantize_sliced=True)
    model_info_format + "_".join()
    with open('/'.join([base_path,"UNet_M[{}-{}-{}-{}]_Q[{}]_S[{}-{}].json".format(
        *model_config, 
        quantization_level, 
        "_".join(map(str,[i - 1 for i in start_points])), 
        "_".join(map(str, end_points)))]), "w") as json_file:
        graph_json['extra'] = {}
        graph_json['extra']['inputs'] = input_indexs
        graph_json['extra']['outputs'] = output_indexs
        json_file.write(json.dumps(graph_json))