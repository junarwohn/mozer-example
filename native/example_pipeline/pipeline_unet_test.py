import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"
import tvm
from tvm import relay
from tvm.relay.dataflow_pattern import rewrite
from tvm.relay.dataflow_pattern import *
import numpy as np
from tvm.relay import transform
from tvm.relay.testing import run_opt_pass
from tvm.relay import transform, build_module
from tvm.relay import testing
import tvm.testing
# from SlicingMachine import TVMSlicer
from mozer.slicer.SlicingMachine import TVMSlicer
from mozer.slicer.Quantize import quantize
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
from tvm.contrib import relay_viz
from tvm.relay import build_module
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input
import time
from tvm.contrib import graph_executor, pipeline_executor, pipeline_executor_build
import cv2

parser = ArgumentParser()
parser.add_argument('--partition', '-p', type=int, default=6)
parser.add_argument('--batch_size', '-b', type=int, default=1)
args = parser.parse_args()

################################################

# import unet
model_keras = tf.keras.models.load_model("/home/jd/workspace/data/UNet.h5")
tf.keras.utils.plot_model(model_keras, show_shapes=True, show_layer_names=True, to_file='{}.png'.format(model_keras.name))

################################################

img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
img_path = download_testdata(img_url, "cat.png", module="data")
img = Image.open(img_path).resize((256, 256))
# data = np.array(img)[np.newaxis, :].astype("float32")
data = np.array(img).astype("float32")
data = np.array([data for _ in range(args.batch_size)])
data = preprocess_input(data).transpose([0, 3, 1, 2])

################################################

shape_dict = {"input_1": data.shape}
mod, params = relay.frontend.from_keras(model_keras, shape_dict)

# target = "cuda"
# dev = tvm.device("cuda", 0)

# with tvm.transform.PassContext(opt_level=4):
#     lib = relay.build(mod, target, params=params)
# total_outs = []
# iter = 100
# model = graph_executor.GraphModule(lib["default"](dev))
# for _ in range(iter):
#     model.set_input('input_1', data)
#     model.run()
#     out = model.get_output(0).numpy()
#     total_outs.append(1)

# iter = 100
# stime = time.time()
# model = graph_executor.GraphModule(lib["default"](dev))
# for _ in range(iter):
#     model.set_input('input_1', data)
#     model.run()
#     out = model.get_output(0).numpy()
#     total_outs.append(1)
#     out = np.squeeze(out.transpose([0,2,3,1]))
#     total_outs.append(1)

# etime = time.time()

# print(f"single time : {etime-stime}")

# original_out = out
original_out = []
# exit()
# del model

# visualized
# print(out.shape)
# cv2.imshow("as", original_out)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

relay_slicer = TVMSlicer()
print(relay_slicer.get_node_count(mod, is_op('nn.leaky_relu')(wildcard())))

split_config = [{"op_name": "nn.leaky_relu", "op_index": args.partition}]

subgraphs, input_name_hints, output_name_hints = relay_slicer.slice_relay_graph(mod['main'], split_config, params)
# print(len(subgraphs))
# exit()
for idx, graph in enumerate(subgraphs):
    ann = run_opt_pass(graph, transform.ToGraphNormalForm())
    subgraphs[idx] =  tvm.IRModule.from_expr(ann)

pipe_config = pipeline_executor_build.PipelineConfig()
mod0, mod1 = subgraphs[0], subgraphs[1]

# pipe_config[mod0].target = "cuda"
pipe_config[mod0].target = "cuda -arch=sm_86"
pipe_config[mod0].dev = tvm.device("cuda", 0)
# # pipe_config[mod0].build_func = cutlass_build
pipe_config[mod0].export_cc = "nvcc"


# pipe_config[mod0].target = "cuda -arch=sm_61"
# pipe_config[mod0].dev = tvm.device("cuda", 1)
# pipe_config[mod0].export_cc = "nvcc"


#pipe_config[mod1].target = "llvm"
#pipe_config[mod1].dev = tvm.device("cpu", 0)

# pipe_config[mod1].target = "cuda"
# pipe_config[mod1].target = "cuda -arch=sm_61"
# pipe_config[mod1].dev = tvm.device("cuda", 1)
pipe_config[mod1].target = "cuda -arch=sm_75"
pipe_config[mod1].dev = tvm.device("cuda", 1)
# pipe_config[mod1].build_func = cutlass_build
pipe_config[mod1].export_cc = "nvcc"

# Create the pipeline by connecting the subgraph modules.
# The global input will be forwarded to the input interface of the first module named mod0
for name_hint in input_name_hints[0]:
    pipe_config["input"][name_hint].connect(pipe_config[mod0]["input"][name_hint])

in_out_mapper = dict()
for idx, name in enumerate(output_name_hints[0]):
    in_out_mapper[name] = idx
# The first output of mod0 will be forwarded to the input interface of mod1
for name in input_name_hints[1]:
    pipe_config[mod0]["output"][in_out_mapper[name]].connect(pipe_config[mod1]["input"][name])

# The first output of mod1 will be the first global output.
pipe_config[mod1]["output"][0].connect(pipe_config["output"][0])

with tvm.transform.PassContext(opt_level=4):
    pipeline_mod_factory = pipeline_executor_build.build(pipe_config)

# config_file_name = pipeline_mod_factory.export_library("./unet_pipeline")
# pipeline_module = pipeline_executor.PipelineModule.load_library("./unet_pipeline/config")

# exit()
# directory_path = tvm.contrib.utils.tempdir().temp_dir
# os.makedirs(directory_path, exist_ok=True)
# config_file_name = pipeline_mod_factory.export_library(directory_path)
# pipeline_module = pipeline_executor.PipelineModule.load_library(config_file_name)

# pipeline_module.module.pipe_config[mod0].target = "cuda -arch=sm_86"
# pipeline_module.module.pipe_config[mod0].dev = tvm.device("cuda", 0)
# pipeline_module.module.pipe_config[mod0].export_cc = "nvcc"

# pipeline_module.module.pipe_config[mod1].target = "cuda -arch=sm_75"
# pipeline_module.module.pipe_config[mod1].dev = tvm.device("cuda", 1)
# pipeline_module.module.pipe_config[mod1].export_cc = "nvcc"

pipeline_module = pipeline_executor.PipelineModule(pipeline_mod_factory)
# idata = tvm.nd.array(data)
iter = 100
# iter = 1
total_outs = []
after_burn = 0
for i in range(iter):
    idata = tvm.nd.array(data)
    pipeline_module.set_input("input_1", idata)
    # time.sleep(5)
    pipeline_module.run()
    # time.sleep(5)
    # outputs = pipeline_module.get_output(synchronize=False)

    outputs = pipeline_module.get_output(synchronize=True)
    # time.sleep(5)
    # if i - len(total_outs) + 1 > 2 and len(outputs) == 0:
    #     outputs = pipeline_module.get_output(synchronize=True)

    if outputs:
        for out in outputs:
            partition_out = np.squeeze(out.numpy().transpose([0,2,3,1]))
            total_outs.append(1)

while len(total_outs) != iter:
    if True:
        outputs = pipeline_module.get_output(synchronize=True)

    if outputs:
        for out in outputs:
            partition_out = out.numpy()
            partition_out = np.squeeze(partition_out.transpose([0,2,3,1]))
            total_outs.append(1)


total_outs = []
iter = 100
# iter = 1
stime = time.time()
for i in range(iter):
    pipeline_module.set_input("input_1", idata)
    pipeline_module.run()
    outputs = pipeline_module.get_output(synchronize=False)

    if outputs:
        for out in outputs:
            partition_out = out.numpy()
            partition_out = np.squeeze(partition_out.transpose([0,2,3,1]))
            total_outs.append(1)


while len(total_outs) != iter:
    if True:
        outputs = pipeline_module.get_output(synchronize=True)

    if outputs:
        for out in outputs:
            partition_out = out.numpy()
            partition_out = np.squeeze(partition_out.transpose([0,2,3,1]))
            total_outs.append(1)

etime = time.time()
print("end pipeline")
print(f"pipeline time : {etime-stime}")



print("compare two result", np.array_equal(original_out, partition_out))
# vis = np.concatenate((original_out, partition_out), axis=1)
# cv2.imshow('out', vis)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# tvm_out = top1_keras
# top1_tvm = top1_keras
# print("Relay top-1 id: {}, class name: {}".format(top1_tvm, synset[top1_tvm]))
# """
################################################
# from tvm.contrib import relay_viz
# graph_attr = {"color": "red"}
# node_attr = {"color": "blue"}
# edge_attr = {"color": "black"}

# target_node_cnt = 0

# def get_node_attr(node):
#     global target_node_cnt
#     if "nn.leaky_relu" in node.type_name:
#         # print(node.type_name)
#         target_node_cnt += 1
#         return {
#             "fillcolor": "green",
#             "style": "filled",
#             "shape": "box",
#         }
#     if "Var" in node.type_name:
#         return {"shape": "ellipse"}
#     return {"shape": "box"}
    
# dot_plotter = relay_viz.DotPlotter(
#     graph_attr=graph_attr,
#     node_attr=node_attr,
#     edge_attr=edge_attr,
#     get_node_attr=get_node_attr)

# viz = relay_viz.RelayVisualizer(
#     subgraphs[1],
#     relay_param=params,
#     plotter=dot_plotter,
#     parser=relay_viz.DotVizParser())
# viz.render("subgraphs1")
# print(target_node_cnt)
# ann = run_opt_pass(subgraphs[0], transform.ToGraphNormalForm())
# mod1 = tvm.IRModule.from_expr(ann)

# viz = relay_viz.RelayVisualizer(
#     mod1,
#     relay_param=params,
#     plotter=dot_plotter,
#     parser=relay_viz.DotVizParser())
# viz.render("slice_0")

# ann = run_opt_pass(subgraphs[1], transform.ToGraphNormalForm())
# mod2 = tvm.IRModule.from_expr(ann)

# viz = relay_viz.RelayVisualizer(
#     mod2,
#     relay_param=params,
#     plotter=dot_plotter,
#     parser=relay_viz.DotVizParser())
# viz.render("sliced_1")

# ################################################
# print("load original")
# time.sleep(5)
# # # Original
# mod = mod
# with tvm.transform.PassContext(opt_level=4):
#     lib = relay.build(mod, target, params=params)
# total_model = graph_executor.GraphModule(lib["default"](dev))

# total_model.set_input('input_1', data)
# total_model.run()
# total_model.get_output(0).numpy()

# ################################################
# print("load 1")
# time.sleep(5)
# # Partition 1
# mod = subgraphs[0]
# ann = run_opt_pass(mod, transform.ToGraphNormalForm())
# mod = tvm.IRModule.from_expr(ann)['main']

# with tvm.transform.PassContext(opt_level=4):
#     lib = relay.build(mod, target, params=params)

# model_1 = graph_executor.GraphModule(lib["default"](dev))

# ################################################
# print("load 1")
# time.sleep(5)
# # Partition 2
# mod = subgraphs[1]
# ann = run_opt_pass(mod, transform.ToGraphNormalForm())
# mod = tvm.IRModule.from_expr(ann)['main']

# with tvm.transform.PassContext(opt_level=4):
#     lib = relay.build(mod, target, params=params)

# model_2 = graph_executor.GraphModule(lib["default"](dev))

# ################################################
# dtype = "float32"
# total_model.set_input('input_1', data)
# total_model.run()
# tvm_out = total_model.get_output(0)
# top1_tvm = np.argmax(tvm_out.numpy()[0])


# print("Relay top-1 id: {}, class name: {}".format(top1_tvm, synset[top1_tvm]))

################################################

# model_1.set_input("input_1", data)
# model_1.run()
# out_dict = dict()
# for i, out_name in enumerate(output_name_hints[0]):
#     out_dict[out_name] = model_1.get_output(i).numpy()

# for k in out_dict:
#     model_2.set_input(k, out_dict[k])

# model_2.run()
# partition_out = model_2.get_output(0).numpy()
"""
print(np.all(total_outs == total_outs[0]), after_burn)
pipeline_module.set_input("input_1", tvm.nd.array(data))
pipeline_module.run()
outputs = pipeline_module.get_output()

partition_out = outputs[0].numpy()
top1_keras = np.argmax(partition_out)
print("Partition top-1 id: {}, class name: {}".format(top1_keras, synset[top1_keras]))
"""
################################################
