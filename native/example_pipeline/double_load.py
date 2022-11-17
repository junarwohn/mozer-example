# from SlicingMachine import TVMSlicer
from mozer.slicer.SlicingMachine import TVMSlicer
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
import time

# I have to reformat the way of saving model
parser = ArgumentParser()
parser.add_argument('--partition_points', '-p', nargs='+', type=str, default=[], help='set partition points')
parser.add_argument('--img_size', '-i', type=int, default=256, help='set image size')
parser.add_argument('--model', '-m', type=str, default='unet', help='name of model')
parser.add_argument('--target', '-t', type=str, default='cuda', help='name of taget')
parser.add_argument('--device', '-d', type=str, default='2080ti', help='name of device')
# parser.add_argument('--target', '-t', type=str, default='llvm', help='name of taget')
parser.add_argument('--opt_level', '-o', type=int, default=3, help='set opt_level')
parser.add_argument('--build', '-b', type=int, default=0, help='build model')
parser.add_argument('--quantization_level', '-q', type=int, default=0, help='set quantization level')
parser.add_argument('--model_config', '-c', nargs='+', type=int, default=0, help='set partition point')
parser.add_argument('--jetson', '-j', type=int, default=0, help='jetson')
parser.add_argument('--base_path', type=str, default=os.environ['TS_DATA_PATH'] + "/model_tf/unet_v1/best/", help='path setting')
args = parser.parse_args()

base_path = os.environ['TS_DATA_PATH'] + "/model_tf/unet_v1/best/"
model_path_2080ti = base_path + "UNet_M[{}-{}-{}-{}]_Q[{}]_full_2080ti.so".format(0,0,0,0,2)
model_path_1050ti = base_path + "UNet_M[{}-{}-{}-{}]_Q[{}]_full_1050ti.so".format(0,0,0,0,2)
model_path_llvm = base_path + "UNet_M[{}-{}-{}-{}]_Q[{}]_full_llvm.so".format(0,0,0,0,2)
# model_path_llvm = base_path + "UNet_M[{}-{}-{}-{}]_Q[{}]_full_llvm_o.so".format(0,0,0,0,2)

lib_2080ti = tvm.runtime.load_module(model_path_2080ti)
# # lib_2080ti = tvm.runtime.load_module(model_path_1050ti)
lib_1050ti = tvm.runtime.load_module(model_path_1050ti)
# # lib_1050ti = tvm.runtime.load_module(model_path_2080ti)
lib_llvm = tvm.runtime.load_module(model_path_llvm)

dev_2080ti = tvm.cuda(0)
dev_1050ti = tvm.cuda(1)
dev_cpu   = tvm.cpu(0)
# print(dev_2080ti)
# print(dev_1050ti)
# time.sleep(10)

model_2080ti = graph_executor.GraphModule(lib_2080ti['default'](dev_2080ti))
model_1050ti = graph_executor.GraphModule(lib_1050ti['default'](dev_1050ti))
model_llvm = graph_executor.GraphModule(lib_llvm['default'](dev_cpu))
# time.sleep(10)

# model_1050ti = graph_executor.GraphModule(lib_1050ti['default'](dev_1050ti))

# time.sleep(10)

data = np.random.normal(0, 1, (1,3,256,256))

# for i in range(10):
#     model_2080ti.set_input("input_1", data)
#     model_2080ti.run()
#     model_2080ti.get_output(0).numpy()

# stime = time.time()
# for i in range(50):
#     model_2080ti.set_input("input_1", data)
#     model_2080ti.run()
#     model_2080ti.run()
#     model_2080ti.get_output(0).numpy()
# print("2080ti :", (time.time() - stime) / 100)

# # del data

# for i in range(10):
#     model_1050ti.set_input("input_1", data)
#     model_1050ti.run()
#     model_1050ti.get_output(0).numpy()

# stime = time.time()
# for i in range(100):
#     model_1050ti.set_input("input_1", data)
#     model_1050ti.run()
#     model_1050ti.get_output(0).numpy()
# print("1050ti :", (time.time() - stime) / 100)

    # model_llvm.set_input("input_1", data)
#     model_llvm.run()
#     model_llvm.get_output(0).numpy()

# Mix execution
for i in range(10):
    model_2080ti.set_input("input_1", data)
    model_1050ti.set_input("input_1", data)
    model_llvm.set_input("input_1", data)
    model_1050ti.run()
    model_2080ti.run()
    model_llvm.run()
    model_2080ti.get_output(0).numpy()
    model_1050ti.get_output(0).numpy()
    model_llvm.get_output(0).numpy()

stime = time.time()
for i in range(100):
    model_2080ti.set_input("input_1", data)
    model_1050ti.set_input("input_1", data)
    model_llvm.set_input("input_1", data)
    model_llvm.run()
    model_1050ti.run()
    model_2080ti.run()
    model_2080ti.get_output(0).numpy()
    model_1050ti.get_output(0).numpy()
    model_llvm.get_output(0).numpy()
print("mixed execution 2080ti + 1050ti + llvm :", (time.time() - stime) / 100)

# del data

# for i in range(10):
#     model_1050ti.set_input("input_1", data)
#     model_1050ti.run()
#     model_1050ti.get_output(0).numpy()

# stime = time.time()
# for i in range(100):
#     model_1050ti.set_input("input_1", data)
#     model_1050ti.run()
#     model_1050ti.get_output(0).numpy()
# print("1050ti :", (time.time() - stime) / 100)


# for i in range(10):
#     model_llvm.set_input("input_1", data)
#     model_llvm.run()
#     model_llvm.get_output(0).numpy()

# stime = time.time()
# for i in range(100):
#     model_llvm.set_input("input_1", data)
#     model_llvm.run()
#     model_llvm.get_output(0).numpy()
# print("llvm :", (time.time() - stime) / 100)