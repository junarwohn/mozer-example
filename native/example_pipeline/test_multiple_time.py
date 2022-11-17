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
import time

parser = ArgumentParser()
parser.add_argument('--partition_points', '-p', nargs='+', type=str, default=[], help='set partition points')
parser.add_argument('--img_size', '-i', type=int, default=224, help='set image size')
# parser.add_argument('--model', '-m', type=str, default='unet', help='name of model')
parser.add_argument('--model', '-m', type=str, default='resnet152', help='name of model')
parser.add_argument('--target', '-t', type=str, default='cuda', help='name of taget')
# parser.add_argument('--device', '-d', type=str, default='2080ti', help='name of device')
parser.add_argument('--opt_level', '-o', type=int, default=4, help='set opt_level')
# parser.add_argument('--opt_level', '-o', type=int, default=4, help='set opt_level')
parser.add_argument('--build', '-b', type=int, default=0, help='build model')
parser.add_argument('--quantization_level', '-q', type=int, default=2, help='set quantization level')
parser.add_argument('--model_config', '-c', nargs='+', type=int, default=0, help='set partition point')
parser.add_argument('--jetson', '-j', type=int, default=0, help='jetson')
parser.add_argument('--base_path', type=str, default=os.environ['TS_DATA_PATH'] + "/model_tf/unet_v1/best/", help='path setting')
parser.add_argument('--tvm_build_path', type=str, default=os.environ['TS_DATA_PATH'] + "/model_tvm/resnet50/", help='path setting')
parser.add_argument('--gpu_index', type=str, default="0", help='set target gpu index')
args = parser.parse_args()

# set path
base_path = args.base_path
tvm_build_path = args.tvm_build_path

# current_file_path = os.path.dirname(os.path.realpath(__file__)) + "/"
is_jetson = args.jetson
model_config = args.model_config

presets = {
    "NVIDIA GeForce RTX 2080 Ti" : {
        'arch' : "sm_75"
    },
    "NVIDIA GeForce GTX 1050 Ti" : {
        'arch' : "sm_61"
    }
}

if is_jetson == 1:
    target = tvm.target.Target("nvidia/jetson-agx-xavier")
    assert target.kind.name == "cuda"
    assert target.attrs["shared_memory_per_block"] == 49152
    assert target.attrs["max_threads_per_block"] == 1024
    assert target.attrs["thread_warp_size"] == 32
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

if is_jetson == 1:
    json_format = '/'.join([tvm_build_path, model_info_format + ".json"])
else:
    json_format = '/'.join([tvm_build_path, model_info_format + ".json"])

with open(json_format, "r") as json_file:
    graph_json_raw = json.load(json_file)

partition_points = sorted([int(i[0]) for i in graph_json_raw['heads']])

model_format = '/'.join([tvm_build_path, model_info_format + ".so"])
lib = tvm.runtime.load_module(model_format)
param_format = '/'.join([tvm_build_path, model_info_format + ".params"])
with open(param_format, "rb") as fi:
    loaded_params = bytearray(fi.read())

##################################################################
args.gpu_index = "1"

if is_jetson == 1:
    target = tvm.target.Target("nvidia/jetson-agx-xavier")
    assert target.kind.name == "cuda"
    assert target.attrs["shared_memory_per_block"] == 49152
    assert target.attrs["max_threads_per_block"] == 1024
    assert target.attrs["thread_warp_size"] == 32
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
        dev2 = tvm.cuda(int(args.gpu_index))
    elif args.target == 'opencl':
        target = 'opencl'
        dev = tvm.opencl()

quantization_level = args.quantization_level

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

if is_jetson == 1:
    json_format = '/'.join([tvm_build_path, model_info_format + ".json"])
else:
    json_format = '/'.join([tvm_build_path, model_info_format + ".json"])

with open(json_format, "r") as json_file:
    graph_json_raw2 = json.load(json_file)

partition_points = sorted([int(i[0]) for i in graph_json_raw2['heads']])

model_format = '/'.join([tvm_build_path, model_info_format + ".so"])
lib2 = tvm.runtime.load_module(model_format)
param_format = '/'.join([tvm_build_path, model_info_format + ".params"])
with open(param_format, "rb") as fi:
    loaded_params2 = bytearray(fi.read())
print(json_format)


####################################################

tvm_slicer = TVMSlicer(graph_json_raw)
graph_json, input_indexs, output_indexs = tvm_slicer.slice_graph([1], [partition_points[-1]], is_quantize_sliced=True)
model = graph_executor.create(json.dumps(graph_json), lib, dev)
model.load_params(loaded_params)
for t in range(100):
    model.run()
    dev.sync()
stime = time.time()
for t in range(100):
    model.run()
    dev.sync()
print(gpu_info['0'], (time.time() - stime)/100)

####################################################

tvm_slicer = TVMSlicer(graph_json_raw2)
graph_json, input_indexs, output_indexs = tvm_slicer.slice_graph([1], [partition_points[-1]], is_quantize_sliced=True)
model2 = graph_executor.create(json.dumps(graph_json), lib2, dev2)
model2.load_params(loaded_params2)
for t in range(100):
    model2.run()
    dev2.sync()
stime = time.time()
for t in range(100):
    model2.run()
    dev2.sync()
print(gpu_info['1'], (time.time() - stime)/100)

####################################################

for i in range(len(partition_points)-1):
    start_points = [1]
    end_points =  [partition_points[i]]
    tvm_slicer = TVMSlicer(graph_json_raw)
    graph_json, input_indexs, output_indexs = tvm_slicer.slice_graph(start_points, end_points, is_quantize_sliced=True)
    model = graph_executor.create(json.dumps(graph_json), lib, dev)
    model.load_params(loaded_params)


    start_points = [partition_points[i] + 1]
    end_points =  [partition_points[-1]]
    tvm_slicer = TVMSlicer(graph_json_raw2)
    graph_json, input_indexs, output_indexs = tvm_slicer.slice_graph(start_points, end_points, is_quantize_sliced=True)
    model2 = graph_executor.create(json.dumps(graph_json), lib2, dev2)
    model2.load_params(loaded_params2)

    for t in range(100):
        model2.run()
        model.run()
        dev.sync()
        dev2.sync()
    stime = time.time()
    for t in range(100):
        model2.run()
        model.run()
        dev.sync()
        dev2.sync()
    print([1], gpu_info['0'], start_points, gpu_info['1'], end_points, (time.time() - stime)/100)

for i in range(len(partition_points)-1):
    start_points = [partition_points[i] + 1]
    end_points =  [partition_points[-1]]
    tvm_slicer = TVMSlicer(graph_json_raw)
    graph_json, input_indexs, output_indexs = tvm_slicer.slice_graph(start_points, end_points, is_quantize_sliced=True)
    model = graph_executor.create(json.dumps(graph_json), lib, dev)
    model.load_params(loaded_params)


    start_points = [1]
    end_points =  [partition_points[i]]
    tvm_slicer = TVMSlicer(graph_json_raw2)
    graph_json, input_indexs, output_indexs = tvm_slicer.slice_graph(start_points, end_points, is_quantize_sliced=True)
    model2 = graph_executor.create(json.dumps(graph_json), lib2, dev2)
    model2.load_params(loaded_params2)

    for t in range(100):
        model2.run()
        model.run()
        dev.sync()
        dev2.sync()
    stime = time.time()
    for t in range(100):
        model2.run()
        model.run()
        dev.sync()
        dev2.sync()
    # print(start_points, end_points, (time.time() - stime)/100)
    print(start_points, gpu_info['1'], end_points, gpu_info['0'], [partition_points[-1]], (time.time() - stime)/100)
