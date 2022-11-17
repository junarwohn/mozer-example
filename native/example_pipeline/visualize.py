import numpy as np
from tvm.relay import testing
import tvm
from tvm import te
from tvm.contrib import graph_executor
import tvm.testing
import tensorflow as tf
import numpy as np
import tvm.relay as relay
from tvm import relay
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.dataflow_pattern import *
import os
import json
import numpy as np
import pygraphviz as pgv
from argparse import ArgumentParser
import subprocess

parser = ArgumentParser()
parser.add_argument('--partition_points', '-p', nargs='+', type=str, default=[], help='set partition points')
parser.add_argument('--img_size', '-i', type=int, default=256, help='set image size')
# parser.add_argument('--model', '-m', type=str, default='unet', help='name of model')
parser.add_argument('--model', '-m', type=str, default='resnet50', help='name of model')
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

presets = {
    "NVIDIA GeForce RTX 2080 Ti" : {
        'arch' : "sm_75"
    },
    "NVIDIA GeForce GTX 1050 Ti" : {
        'arch' : "sm_61"
    }
}
# current_file_path = os.path.dirname(os.path.realpath(__file__)) + "/"
is_jetson = args.jetson
model_config = args.model_config

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

model_info = {
    # "name"   : "unet",
    "name"   : "resnet50",
    "host"   : "x86_64",
    "target" : args.target,
    "arch"   : target_arch,
    "opt"    : args.opt_level,
    "qaunt"  : args.quantization_level,
    "slice"  : "_".join([])
}
def show_graph(json_data, file_name=None):
    if type(json_data) == str:
        json_data = json.loads(json_data)
    A = pgv.AGraph(directed=True)
    for node_idx, node in enumerate(json_data['nodes']):
        for src in node['inputs']:
            # if args.show_size == 1:
            if 1 == 1:
                src_size = 1
                for i in json_data['attrs']['shape'][1][src[0]]:
                    src_size = src_size * i
                
                dst_size = 1
                for i in json_data['attrs']['shape'][1][node_idx]:
                    dst_size = dst_size * i
                # print(src[0], json_data['nodes'][src[0]]['name'], src_size)

                A.add_edge(json_data['nodes'][src[0]]['name'] + '[{}]'.format(src[0]) + '{}'.format(json_data['attrs']['dltype'][1][src[0]]) + "[{}]".format(src_size), node['name'] + '[{}]'.format(node_idx) + '{}'.format(json_data['attrs']['dltype'][1][node_idx]) + "[{}]".format(dst_size))
            else:
                A.add_edge(json_data['nodes'][src[0]]['name'] + '[{}]'.format(src[0]) + '{}'.format(json_data['attrs']['dltype'][1][src[0]]), node['name'] + '[{}]'.format(node_idx) + '{}'.format(json_data['attrs']['dltype'][1][node_idx]))
    if file_name:
        A.draw(file_name + '.png', format='png', prog='dot')

# set path
base_path = args.base_path
tvm_build_path = args.tvm_build_path

model_config = args.model_config
quantization_level = args.quantization_level
partition_points = args.partition_points
partition_points = ["_".join(l.split(',')) for l in partition_points]
base_path = args.base_path

model_info = {
    # "name"   : "unet",
    "name"   : "resnet50",
    "host"   : "x86_64",
    "target" : args.target,
    "arch"   : target_arch,
    "opt"    : args.opt_level,
    "qaunt"  : args.quantization_level,
    "slice"  : "_".join([])
}
model_info_format = "{name}-{host}-{target}-{arch}-o{opt}-q{qaunt}-s{slice}".format(**model_info)

# path = '/'.join([base_path, "./UNet_M[{}-{}-{}-{}]_Q[{}]_S[{}].json".format(*model_config, quantization_level, "-".join(partition_points))])
json_format = '/'.join([tvm_build_path, model_info_format + ".json"])

# img_path = "./UNet_M[{}-{}-{}-{}]_Q[{}]_S[{}]".format(*model_config, quantization_level, "-".join(partition_points))
img_format = '/'.join([tvm_build_path, model_info_format])

with open(json_format, "r") as json_file:
    json_graph = json.load(json_file)
    show_graph(json_graph, img_format)
