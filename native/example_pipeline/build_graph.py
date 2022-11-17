# from SlicingMachine import TVMSlicer
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
from tvm import rpc
from tvm.autotvm.measure.measure_methods import set_cuda_target_arch
from tensorflow import keras
from tvm.contrib.download import download_testdata
import subprocess

parser = ArgumentParser()
parser.add_argument('--partition_points', '-p', nargs='+', type=str, default=[], help='set partition points')
parser.add_argument('--img_size', '-i', type=int, default=224, help='set image size')
# parser.add_argument('--model', '-m', type=str, default='unet', help='name of model')
parser.add_argument('--model', '-m', type=str, default='resnet152', help='name of model')
parser.add_argument('--target', '-t', type=str, default='cuda', help='name of taget')
# parser.add_argument('--device', '-d', type=str, default='2080ti', help='name of device')
parser.add_argument('--opt_level', '-o', type=int, default=4, help='set opt_level')
parser.add_argument('--build', '-b', type=int, default=0, help='build model')
parser.add_argument('--quantization_level', '-q', type=int, default=2, help='set quantization level')
parser.add_argument('--model_config', '-c', nargs='+', type=int, default=0, help='set partition point')
parser.add_argument('--jetson', '-j', type=int, default=0, help='jetson')
parser.add_argument('--base_path', type=str, default=os.environ['TS_DATA_PATH'] + "/model_tf/unet_v1/best/", help='path setting')
# parser.add_argument('--tvm_build_path', type=str, default=os.environ['TS_DATA_PATH'] + "/model_tvm/unet_v1/best/", help='path setting')
parser.add_argument('--tvm_build_path', type=str, default=os.environ['TS_DATA_PATH'] + "/model_tvm/resnet50/", help='path setting')
parser.add_argument('--gpu_index', type=str, default="0", help='set target gpu index')
args = parser.parse_args()

# set path
base_path = args.base_path
tvm_build_path = args.tvm_build_path

# current_file_path = os.path.dirname(os.path.realpath(__file__)) + "/"
is_jetson = args.jetson
model_config = args.model_config

if args.model == 'resnet50':
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
    input_data = np.random.normal(0,1,(1,224,224,3)).astype(np.float32)
    input_data = input_data.transpose([0, 3, 1, 2])
    shape_dict = {"input_1": input_data.shape}

elif args.model == 'resnet152':
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
    input_data = np.random.normal(0,1,(1,224,224,3)).astype(np.float32)
    input_data = input_data.transpose([0, 3, 1, 2])
    shape_dict = {"input_1": input_data.shape}
else:
    np.random.seed(0)
    img_size = args.img_size
    input_data = np.random.normal(0,1,(1,256,256,3)).astype(np.float32)
    model_keras = tf.keras.models.load_model('/'.join([base_path, "UNet_M[{}-{}-{}-{}].h5"]).format(*model_config))
    # tvm result
    input_data = input_data.transpose([0, 3, 1, 2])
    shape_dict = {"input_1": input_data.shape}

mod, params = relay.frontend.from_keras(model_keras, shape_dict)


presets = {
    "NVIDIA GeForce RTX 2080 Ti" : {
        'arch' : "sm_75"
    },
    "NVIDIA GeForce GTX 1050 Ti" : {
        'arch' : "sm_61"
    }
}


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
    gpu_info = subprocess.check_output("nvidia-smi --query-gpu=index,name --format=csv", shell=True, encoding='utf-8').split("\n")[1:-1]
    gpu_info = {
        x.split(', ')[0] : x.split(', ')[1] 
        for x in gpu_info
    }
    if args.target == 'llvm':
        target = 'llvm'
        dev = tvm.cpu()
    elif args.target == 'cuda':
        gpu_name = gpu_info[args.gpu_index]
        target_arch = presets[gpu_name]['arch']
        target = f'cuda -arch={target_arch}'
        dev = tvm.cuda(int(args.gpu_index))
    elif args.target == 'opencl':
        target = 'opencl'
        dev = tvm.opencl()

quantization_level = args.quantization_level
out = quantize(mod, quantization_level)

if target == 'llvm':
    disabled_pass={'AlterOpLayout'}
else :
    disabled_pass={}

with tvm.transform.PassContext(opt_level=4, disabled_pass=disabled_pass):
    lib = relay.build(out, target, params=params)
    # lib = relay.build(mod, target, params=params)
    # lib = relay.build(out, target, params=params, target_host="llvm -mtriple=aarch64-linux-gnueabihf -device=arm_cpu")
    # lib = relay.build(out, target='cuda -arch=sm_72 -model=tx2', params=params, target_host="llvm -mtriple=aarch64-linux-gnueabihf -device=arm_cpu")
    # lib = relay.build(out, target='cuda -arch=sm_72 -model=tx2', params=params, target_host='llvm -mtriple=aarch64-linux-gnueabihf -device=arm_cpu')

graph_json_raw = lib['get_graph_json']()
print(*[i[0] for i in json.loads(graph_json_raw)["heads"]])
tvm_slicer = TVMSlicer(graph_json_raw)

model_info = {
    # "name"   : "unet",
    "name"   : args.model,
    "host"   : "x86_64",
    "target" : args.target,
    "arch"   : target_arch,
    "opt"    : args.opt_level,
    "qaunt"  : args.quantization_level,
    "slice"  : "_".join([])
}
model_info_format = "{name}-{host}-{target}-{arch}-o{opt}-q{qaunt}-s{slice}".format(**model_info)
# Build lib and params
if args.build == 1:
    if is_jetson == 1:
        model_format = '/'.join([tvm_build_path, model_info_format + ".so"])
    else:
        model_format = '/'.join([tvm_build_path, model_info_format + ".so"])
    lib.export_library(model_format)
    # lib.export_library(model_format.format(*model_config, quantization_level), cc=f'/usr/bin/aarch64-linux-gnu-g++')
    # lib.export_library(lib_path, cc=f'/usr/bin/aarch64-linux-gnu-g++')

    if is_jetson == 1:
        param_format = '/'.join([tvm_build_path, model_info_format + ".params"])
    else:
        param_format = '/'.join([tvm_build_path, model_info_format + ".params"])
    param_bytes = tvm.runtime.save_param_dict(lib.get_params())
    with open(param_format, "wb") as f:
        f.write(param_bytes)

# TODO adding final_shape 
# do 'extra' job to 
if is_jetson == 1:
    json_format = '/'.join([tvm_build_path, model_info_format + ".json"])
else:
    json_format = '/'.join([tvm_build_path, model_info_format + ".json"])

with open(json_format, "w") as json_file:
    json_file.write(graph_json_raw)

graph_info = json.loads(graph_json_raw)
# print(len(graph_info['nodes'])-1)

slice_points = "{}_{}".format(0, len(graph_info['nodes'])-1)
json_format = '/'.join([tvm_build_path, model_info_format + "{}.json"]).format(slice_points)
# print([1], [len(graph_info['nodes'])-1])
graph_json, input_indexs, output_indexs = tvm_slicer.slice_graph([1], [len(graph_info['nodes'])-1], is_quantize_sliced=True)
graph_json['extra'] = {}
graph_json['extra']['inputs'] = input_indexs
graph_json['extra']['outputs'] = output_indexs

with open(json_format, "w") as json_file:
    json_file.write(json.dumps(graph_json))
