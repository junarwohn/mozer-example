import json
from argparse import ArgumentParser
from multiprocessing import pool
import numpy as np
import pickle
import tvm
import tvm.relay as relay
from tvm.contrib import graph_executor
from itertools import permutations
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from mozer.slicer.SlicingMachine import TVMSlicer
from tvm.contrib.download import download_testdata
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.applications.resnet import preprocess_input
from mozer.slicer.SlicingMachine import TVMSlicer
from tvm.contrib.tar import tar
from tvm.relay.dataflow_pattern import *
import tensorflow as tf
import pygraphviz as pgv
from tensorflow.keras.utils import plot_model

class UnetPreProcessCallback(DFPatternCallback):
    # A callback class to rewrite the matched pattern to a batch_norm op.
    def __init__(self, require_type=False):
        super().__init__(require_type)
        super().__init__(rewrite_once=True)

        self.var2 = wildcard()
        tuple_node = is_tuple([wildcard(), self.var2])
        concat_node = is_op('concatenate')(tuple_node)
        self.pattern = concat_node
        self.match_node = []
        self.match_node2 = []

    def callback(self, pre, post, node_map):
        var2 = node_map[self.var2][0]
        self.match_node.append(var2)
        self.match_node2.append(pre)
        return pre 
        
class UnetCallback(DFPatternCallback):
    # A callback class to rewrite the matched pattern to a batch_norm op.
    def __init__(self, match_node, require_type=False):
        super().__init__(require_type)
        super().__init__(rewrite_once=True)

        self.tuple_get_item_node = is_tuple_get_item(wildcard(), 0)
        self.pattern_1 = self.tuple_get_item_node

        self.pattern = self.pattern_1 
        self.match_node = match_node
        self.counter = 0
        self.tmp = []

    def quant(self, node):
        # cast_to_int8 = relay.cast(
        #     relay.clip(
        #         relay.round(
        #             relay.multiply(node, relay.const(8.0))
        #         ), 
        #         a_min=-127.0, a_max=127.0
        #     ),
        #     dtype="int8"
        # )

        cast_to_int8 = relay.cast(
            relay.clip(
                relay.round(
                    relay.multiply(
                        relay.subtract(node, relay.const(18.0))
                        , relay.const(7.0))
                ), 
                a_min=-127.0, a_max=127.0
            ),
            dtype="int8"
        )
        result_node = relay.annotation.stop_fusion(cast_to_int8)
        self.tmp.append(result_node)
        return result_node

    def dequant(self, node):
        # cast_to_float32 = relay.divide(
        #     relay.cast(node, dtype='float32'), relay.const(8.0)
        # )
        cast_to_float32 = relay.add(
                relay.divide(
                relay.cast(node, dtype='float32'), relay.const(7.0))
            , relay.const(18.0)
        )
        return cast_to_float32

    def callback(self, pre, post, node_map):
        if self.pattern_1.match(pre):
            if pre in self.match_node:
                # print("pat 1")
                return self.dequant(self.quant(post))
        return post

class UnetCallback2(DFPatternCallback):
    # A callback class to rewrite the matched pattern to a batch_norm op.
    def __init__(self, match_node, require_type=False):
        super().__init__(require_type)
        super().__init__(rewrite_once=True)

        # self.tuple_get_item_node = is_tuple_get_item(wildcard(), 0)
        # self.pattern_1 = self.tuple_get_item_node
        self.var2 = wildcard()
        tuple_node = is_tuple([wildcard(), self.var2])
        concat_node = is_op('concatenate')(tuple_node)
        self.pattern = concat_node
        # self.pattern = self.pattern_1 
        self.match_node = match_node
        self.counter = 0
        self.tmp = []

    def quant(self, node):
        # cast_to_int8 = relay.cast(
        #     relay.clip(
        #         relay.round(
        #             relay.multiply(node, relay.const(8.0))
        #         ), 
        #         a_min=-127.0, a_max=127.0
        #     ),
        #     dtype="int8"
        # )
        cast_to_int8 = relay.cast(
            relay.clip(
                relay.round(
                    relay.multiply(
                        relay.subtract(node, relay.const(18.0))
                        , relay.const(7.0))
                ), 
                a_min=-127.0, a_max=127.0
            ),
            dtype="int8"
        )
        result_node = relay.annotation.stop_fusion(cast_to_int8)
        self.tmp.append(result_node)
        return result_node

    def dequant(self, node):
        # cast_to_float32 = relay.divide(
        #     relay.cast(node, dtype='float32'), relay.const(8.0)
        # )
        cast_to_float32 = relay.add(
                relay.divide(
                relay.cast(node, dtype='float32'), relay.const(7.0))
            , relay.const(18.0)
        )
        return cast_to_float32

    def callback(self, pre, post, node_map):
        if self.pattern.match(pre):
            if pre in self.match_node:
                # print("pat 1")
                return self.dequant(self.quant(post))
        return post


class UnetMaxPool2dCallback(DFPatternCallback):
    # A callback class to rewrite the matched pattern to a batch_norm op.
    def __init__(self, require_type=False):
        super().__init__(require_type)
        super().__init__(rewrite_once=True)

        max_pool2d_node = is_op('nn.max_pool2d')(wildcard())
        self.pattern = max_pool2d_node
        self.match_node = []

    def callback(self, pre, post, node_map):
        self.match_node.append(pre)
        return post


class UnetCallback3(DFPatternCallback):
    # A callback class to rewrite the matched pattern to a batch_norm op.
    def __init__(self, match_node, require_type=False):
        super().__init__(require_type)
        super().__init__(rewrite_once=True)

        # self.tuple_get_item_node = is_tuple_get_item(wildcard(), 0)
        # self.pattern_1 = self.tuple_get_item_node
        max_pool2d_node = is_op('nn.max_pool2d')(wildcard())
        self.pattern = max_pool2d_node
        self.match_node = match_node
        self.counter = 0
        self.tmp = []

    def quant(self, node):
        # cast_to_int8 = relay.cast(
        #     relay.clip(
        #         relay.round(
        #             relay.multiply(node, relay.const(8.0))
        #         ), 
        #         a_min=-127.0, a_max=127.0
        #     ),
        #     dtype="int8"
        # )
        cast_to_int8 = relay.cast(
            relay.clip(
                relay.round(
                    relay.multiply(
                        relay.subtract(node, relay.const(18.0))
                        , relay.const(7.0))
                ), 
                a_min=-127.0, a_max=127.0
            ),
            dtype="int8"
        )
        result_node = relay.annotation.stop_fusion(cast_to_int8)
        self.tmp.append(result_node)
        return result_node

    def dequant(self, node):
        # cast_to_float32 = relay.divide(
        #     relay.cast(node, dtype='float32'), relay.const(8.0)
        # )

        cast_to_float32 = relay.add(
                relay.divide(
                relay.cast(node, dtype='float32'), relay.const(7.0))
            , relay.const(18.0)
        )
        return cast_to_float32

    def callback(self, pre, post, node_map):
        # print("match pool2d")

        if self.pattern.match(pre):
            if pre in self.match_node:
                # print("pat 1")
                return self.dequant(self.quant(post))
        return post

class UnetLeakyReLUCallback(DFPatternCallback):
    # A callback class to rewrite the matched pattern to a batch_norm op.
    def __init__(self, require_type=False):
        super().__init__(require_type)
        super().__init__(rewrite_once=True)

        leaky_relu_node = is_op('nn.leaky_relu')(wildcard()) | is_op('nn.relu')(is_op('add')(wildcard(), wildcard()))
        self.pattern = leaky_relu_node
        self.match_node = []

    def callback(self, pre, post, node_map):
        self.match_node.append(pre)
        return post


class UnetCallback4(DFPatternCallback):
    # A callback class to rewrite the matched pattern to a batch_norm op.
    def __init__(self, match_node, require_type=False):
        super().__init__(require_type)
        super().__init__(rewrite_once=True)

        # self.tuple_get_item_node = is_tuple_get_item(wildcard(), 0)
        # self.pattern_1 = self.tuple_get_item_node
        leaky_relu_node = is_op('nn.leaky_relu')(wildcard())| is_op('nn.relu')(wildcard())
        self.pattern = leaky_relu_node
        self.match_node = match_node
        self.counter = 0
        self.tmp = []

    def quant(self, node):
        # cast_to_int8 = relay.cast(
        #     relay.clip(
        #         relay.round(
        #             relay.multiply(node, relay.const(8.0))
        #         ), 
        #         a_min=-127.0, a_max=127.0
        #     ),
        #     dtype="int8"
        # )

        cast_to_int8 = relay.cast(
            relay.clip(
                relay.round(
                    relay.multiply(
                        relay.subtract(node, relay.const(18.0))
                        , relay.const(7.0))
                ), 
                a_min=-127.0, a_max=127.0
            ),
            dtype="int8"
        )
        result_node = relay.annotation.stop_fusion(cast_to_int8)
        self.tmp.append(result_node)
        return result_node

    def dequant(self, node):
        # cast_to_float32 = relay.divide(
        #     relay.cast(node, dtype='float32'), relay.const(8.0)
        # )
        cast_to_float32 = relay.add(
                relay.divide(
                relay.cast(node, dtype='float32'), relay.const(7.0))
            , relay.const(18.0)
        )
        return cast_to_float32

    def callback(self, pre, post, node_map):
        # print("match leaky_relu_node")

        if self.pattern.match(pre):
            if pre in self.match_node:
                # print("pat 1")
                return self.dequant(self.quant(post))
        return post

# class AddCollector(DFPatternCallback):
#     # A callback class to rewrite the matched pattern to a batch_norm op.
#     def __init__(self, require_type=False):
#         super().__init__(require_type)
#         super().__init__(rewrite_once=True)

#         add_node = is_op('add')(wildcard(), wildcard())

#         self.pattern = add_node
#         self.match_node = []

#     def callback(self, pre, post, node_map):
#         # print(pre)
#         self.match_node.append(pre)
#         return post


class Int8Collector(DFPatternCallback):
    # A callback class to rewrite the matched pattern to a batch_norm op.
    def __init__(self, require_type=False):
        super().__init__(require_type)
        super().__init__(rewrite_once=True)

        int8_cast_node = is_op('cast')(wildcard()).has_attr({'dtype': 'int8'})

        self.pattern = int8_cast_node
        self.match_node = []

    def callback(self, pre, post, node_map):
        # print(pre)
        self.match_node.append(pre)
        return post

weights_url = "".join(
    [
        " https://storage.googleapis.com/tensorflow/keras-applications/",
        "resnet/resnet101_weights_tf_dim_ordering_tf_kernels.h5",
    ]
)
weights_file = "resnet101_keras_new.h5"


weights_path = download_testdata(weights_url, weights_file, module="keras")
model_keras = tf.keras.applications.resnet.ResNet101(
    include_top=True, weights=None, input_shape=(224, 224, 3), classes=1000
)
model_keras.load_weights(weights_path)

img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
img_path = download_testdata(img_url, "cat.png", module="data")
img = Image.open(img_path).resize((224, 224))


parser = ArgumentParser()
parser.add_argument('--partition_points', '-p', nargs='+', type=str, default=[], help='set partition points')
parser.add_argument('--img_size', '-i', type=int, default=224, help='set image size')
parser.add_argument('--model', '-m', type=str, default='resnet', help='name of model')
parser.add_argument('--target', '-t', type=str, default='cuda', help='name of taget')
parser.add_argument('--opt_level', '-o', type=int, default=3, help='set opt_level')
parser.add_argument('--build', '-b', type=int, default=0, help='build model')
parser.add_argument('--quantization_level', '-q', type=int, default=0, help='set quantization level')
parser.add_argument('--model_config', '-c', nargs='+', type=int, default=0, help='set partition point')
parser.add_argument('--jetson', '-j', type=int, default=0, help='jetson')
parser.add_argument('--base_path', type=str, default=os.environ['TS_DATA_PATH'] + "/tf_model/unet_v1/best/", help='path setting')
args = parser.parse_args()

model_config = args.model_config
quantization_level = args.quantization_level
is_jetson = args.jetson
img_size = args.img_size
base_path = args.base_path
input_data = np.random.normal(0,1,(1,img_size,img_size,3)).astype(np.float32)
# tvm result
input_data = input_data.transpose([0, 3, 1, 2])
shape_dict = {"input_1": input_data.shape}
mod, params = relay.frontend.from_keras(model_keras, shape_dict)

if args.target == 'llvm':
    target = 'llvm'
    dev = tvm.cpu()
elif args.target == 'cuda':
    target = 'cuda'
    dev = tvm.cuda()
elif args.target == 'opencl':
    target = 'opencl'
    dev = tvm.opencl()


quantization_level = args.quantization_level

upc = UnetPreProcessCallback()
out = rewrite(upc, mod['main'])

if quantization_level == 0:
    maxpool = UnetMaxPool2dCallback()
    rewrite(maxpool, out)
    leakyrelu = UnetLeakyReLUCallback()
    rewrite(leakyrelu, out)
    callnodes = upc.match_node + upc.match_node2 + maxpool.match_node + leakyrelu.match_node + [out.body]
    callnodes_str = [str(node) for node in callnodes]
    callnodes_str = list(set(callnodes_str))
    callnodes_str.sort(key=lambda x: len(x))
    callnodes_str = callnodes_str[::-1]
    out_nodes = [None for i in range(len(callnodes_str))]
    for node in callnodes:
        out_nodes[callnodes_str.index(str(node))] = node
    # out = relay.Function(out.params, relay.Tuple(upc.match_node + upc.match_node2 + maxpool.match_node + [out.body]), out.ret_type, out.type_params, out.attrs)
    out = relay.Function(out.params, relay.Tuple(out_nodes), out.ret_type, out.type_params, out.attrs)
else:
    uc = UnetCallback(upc.match_node)
    out = rewrite(uc, mod['main'])
    upc = UnetPreProcessCallback()
    rewrite(upc, out)
    uc2 = UnetCallback2(upc.match_node2)
    out = rewrite(uc2, out)
    
    if quantization_level == 1:
        callnodes = uc.tmp + [out.body]
        callnodes_str = [str(node) for node in callnodes]
        callnodes_str = list(set(callnodes_str))
        callnodes_str.sort(key=lambda x: len(x))
        callnodes_str = callnodes_str[::-1]
        out_nodes = [None for i in range(len(callnodes_str))]
        for node in callnodes:
            out_nodes[callnodes_str.index(str(node))] = node
        # out = relay.Function(out.params, relay.Tuple(uc.tmp + [out.body]), out.ret_type, out.type_params, out.attrs)
        out = relay.Function(out.params, relay.Tuple(out_nodes), out.ret_type, out.type_params, out.attrs)

    elif quantization_level == 2:

        upc = UnetMaxPool2dCallback()
        rewrite(upc, out)
        # print(len(upc.match_node))
        uc2 = UnetCallback3(upc.match_node)
        out = rewrite(uc2, out)

        upc = UnetLeakyReLUCallback()
        rewrite(upc, out)
        # print(len(upc.match_node))

        # add_collecter = AddCollector()
        # rewrite(add_collecter, out)

        uc2 = UnetCallback4(upc.match_node)
        out = rewrite(uc2, out)

        int8_collector = Int8Collector()
        rewrite(int8_collector, out)


        callnodes = int8_collector.match_node + [out.body]
        callnodes_str = [str(node) for node in callnodes]
        callnodes_str = list(set(callnodes_str))
        callnodes_str.sort(key=lambda x: len(x))
        callnodes_str = callnodes_str[::-1]
        out_nodes = [None for i in range(len(callnodes_str))]
        for node in callnodes:
            out_nodes[callnodes_str.index(str(node))] = node
        # out = relay.Function(out.params, relay.Tuple(int8_collector.match_node + [out.body]), out.ret_type, out.type_params, out.attrs)
        out = relay.Function(out.params, relay.Tuple(out_nodes), out.ret_type, out.type_params, out.attrs)

with tvm.transform.PassContext(opt_level=args.opt_level):
    lib = relay.build(out, target, params=params)

graph_raw_json = lib['get_graph_json']()
graph_raw_json = json.loads(graph_raw_json)

# with open(base_path + "/UNet_M[{}-{}-{}-{}]_Q[{}]_full.json".format(
#             *model_config, 
#             quantization_level 
#             ), "r") as json_file:
#     graph_raw_json = json.load(json_file)

def network_simulator(data_size, network='lan'):
    lan_weight = 0.000000008501843269
    lan_bias = -0.000009457928473
    
    wifi_weight = 0.0000000314463409
    wifi_bias = -0.003767568519
    
    if network == 'lan':
        result_time = lan_weight * data_size + lan_bias
        if result_time < 0:
            result_time = 0
            
    else:
        result_time = wifi_weight * data_size + wifi_bias
        if result_time < 0.0005:
            result_time = 0.0005
            
    return result_time
    
def slice_graph(partition_points):
    start_points = [i + 1 for i in partition_points[0]]
    end_points =  partition_points[1]
    # print(start_points, end_points)
    # graph_json, input_indexs, output_indexs = tvm_sxlicer.slice_graph(start_point + 1, end_point, is_quantize_sliced=True)
    tvm_slicer = TVMSlicer(graph_raw_json)
    graph_json, input_indexs, output_indexs = tvm_slicer.slice_graph(start_points, end_points, is_quantize_sliced=True)
    graph_json['extra'] = {}
    graph_json['extra']['inputs'] = input_indexs
    graph_json['extra']['outputs'] = output_indexs
    return graph_json


def get_model_info(partition_points):
    model_input_indexs = []
    model_output_indexs = []
    model_graph_json_strs = []
    model_dummy_inputs = []
    # If there is no model to be executed
    if len(partition_points) == 1:
        partition_points = list(map(int, partition_points))
        return [partition_points], [partition_points], []
    
    # Load front model json infos
    partition_points = list(map(str, partition_points))
    for i in range(len(partition_points) - 1):
        start_points = [int(i) for i in partition_points[i].split(',')]
        end_points =  [int(i) for i in partition_points[i + 1].split(',')]

        graph_json = slice_graph([start_points, end_points])
        input_indexs = graph_json['extra']["inputs"]
        output_indexs = graph_json['extra']["outputs"]
        
        model_input_indexs.append(input_indexs)
        model_output_indexs.append(output_indexs)
        del graph_json['extra']
        model_graph_json_strs.append(json.dumps(graph_json))
        input_node_indexs = [i for i, node in enumerate(graph_json['nodes']) if 'input' in node['name']]
        input_types = [graph_json['attrs']['dltype'][1][i] for i in input_node_indexs]
        input_shapes = [graph_json['attrs']['shape'][1][i] for i in input_node_indexs]
        dummy = []
        for dt, sh in zip(input_types, input_shapes):
            dummy.append(np.random.normal(0, 1, sh).astype(dt))
        model_dummy_inputs.append(dummy)
    return model_input_indexs, model_output_indexs, model_graph_json_strs, model_dummy_inputs

def get_time(lib, dev, loaded_params, input_idxs, output_idxs, graph_json_strs, dummy_inputs):
    set_input_time = 0
    run_time = 0
    get_output_time = 0
    for input_idx, output_idx, graph_json_str, dummy_input in zip(input_idxs, output_idxs, graph_json_strs, dummy_inputs):
        model = graph_executor.create(graph_json_str, lib, dev)
        model.load_params(loaded_params)
        total_frames = 100
        # print(input_idx, len(dummy_input))
        for ii, di in zip(input_idx,dummy_input):
            indata = tvm.nd.array(di, device=dev)
            set_input_time += model.module.time_evaluator(func_name='set_input', dev=dev, number=total_frames)('input_{}'.format(ii), indata).results[0]
        run_time += model.module.time_evaluator(func_name='run', dev=dev, number=total_frames)().results[0]
        
        for i, oi in enumerate(output_idx):
            get_output_time += model.module.time_evaluator(func_name='get_output', dev=dev, number=total_frames)(i).results[0]
        
    return set_input_time + run_time + get_output_time

def get_size(dummy):
    dsz = 99999999999
    if dummy.dtype == 'float32' or dummy.dtype == 'float64':
        dsz = 4
    elif dummy.dtype == 'int8':
        # print("int8")
        dsz = 1
    else:
        print("WRONG DTYPE")
        exit(1)
    for s in dummy.shape:
        dsz *= s
    return dsz

# naive way...
def avoid_narrow_slice(points):
    pre = points[0]
    for i in points[1:]:
        if i - pre < 5:
            return False
        pre = i
    return True

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

def acc_test(lib, dev, loaded_params, input_idxs, output_idxs, graph_json_strs):
    data_path = os.environ['TS_DATA_PATH'] + "VOC_data"
    file_list = sorted(os.listdir(data_path))
    for input_idx, output_idx, graph_json_str in zip(input_idxs, output_idxs, graph_json_strs):
        model = graph_executor.create(graph_json_str, lib, dev)
        model.load_params(loaded_params)
        for file_path in file_list[:100]:
            path = data_path + '/' + file_path
            img = Image.open(path).resize((224, 224))
            data = np.array(img)[np.newaxis, :].astype("float32")
            data = preprocess_input(data).transpose([0, 3, 1, 2])
            dtype = "float32"
            model.set_input("input_{}".format(0), tvm.nd.array(data.astype(dtype)))
            model.run()
            # tvm_out = model.set_input(tvm.nd.array(data.astype(dtype)))
            tvm_out = model.get_output(0).numpy()[0]
            top1_tvm = np.argmax(tvm_out)
            print(top1_tvm)

show_graph(graph_raw_json, "resnet")
plot_model(model_keras, show_shapes=True, show_layer_names=True, to_file='resnet_tf.png')

# target and dev set
if args.target == 'llvm':
    target = 'llvm'
    dev = tvm.cpu()
elif args.target == 'cuda':
    target = 'cuda'
    dev = tvm.cuda()
elif args.target == 'opencl':
    target = 'opencl'
    dev = tvm.opencl()

# # Load models
# if is_jetson == 1:
#     model_format = "UNet_M[{}-{}-{}-{}]_Q[{}]_full_jetson.so"
# else:
#     model_format = "UNet_M[{}-{}-{}-{}]_Q[{}]_full.so"
    
# model_format = "/".join([base_path, model_format])
# model_path = model_format.format(*model_config, quantization_level)
# lib = tvm.runtime.load_module(model_path)

# if is_jetson == 1:
#     param_format = "UNet_M[{}-{}-{}-{}]_Q[{}]_full_jetson.params"
# else:
#     param_format = "UNet_M[{}-{}-{}-{}]_Q[{}]_full.params"

# param_format = "/".join([base_path, param_format])
# param_path = param_format.format(*model_config, quantization_level)
# with open(param_path, "rb") as fi:
    # loaded_params = bytearray(fi.read())

# with open(base_path + "/UNet_M[{}-{}-{}-{}]_Q[{}]_full.json".format(
    #         *model_config, 
    #         quantization_level 
    #         ), "r") as json_file:
    # json_data = json.load(json_file)
# graph_raw_json = json_data

possible_points = [0] + sorted([i[0] for i in graph_raw_json['heads']])
print(possible_points)
candidates1 = [[]]
candidates2 = list(permutations(possible_points[1:-1], 1))
# candidates3 = list(permutations(possible_points[1:-1], 2))
sp = 0
ep = possible_points[-1]

#######
# TODO : Check the original time
#  
raw_input_dummy = np.random.normal(0, 1, (1,224,224,3))
raw_output_dummy = np.random.normal(0, 1, (1, 1000, 1))
original_size = get_size(raw_input_dummy) + get_size(raw_output_dummy)
min_val = 999999999999999999
for candidate in candidates1:
# for candidate in candidates1 + candidates2:
    partition_points = [sp, *candidate, ep]
    if not avoid_narrow_slice(partition_points):
        continue

    input_idxs, output_idxs, graph_json_strs, dummy_inputs = get_model_info(partition_points)
    # print(partition_points)
    communication_size = 0
    inference_time = 0

    # Solo
    if len(input_idxs) == 1:
        front_input_idxs, front_output_idxs, front_graph_json_strs, front_dummy_inputs = get_model_info(partition_points[:2])
        total_front_output_idxs = []
        for i in front_output_idxs:
            total_front_output_idxs += i
        inference_time += get_time(lib.get_lib(), dev, tvm.runtime.save_param_dict(lib.get_params()), input_idxs, output_idxs, graph_json_strs, dummy_inputs)
        print("original size", original_size, "inference", inference_time)
        acc_test(lib.get_lib(), dev, tvm.runtime.save_param_dict(lib.get_params()), input_idxs, output_idxs, graph_json_strs)
        continue

    if len(input_idxs) == 2:
        front_input_idxs, front_output_idxs, front_graph_json_strs, front_dummy_inputs = get_model_info(partition_points[:2])
        back_input_idxs, back_output_idxs, back_graph_json_strs, back_dummy_inputs = get_model_info(partition_points[1:3])

        total_front_output_idxs = []
        for i in front_output_idxs:
            total_front_output_idxs += i

        total_back_input_idxs = []
        for i in back_input_idxs:
            total_back_input_idxs += i

        total_front_output_idxs = []
        for i in front_output_idxs:
            total_front_output_idxs += i

        total_back_input_idxs = []
        for i in back_input_idxs:
            total_back_input_idxs += i

        total_back_output_idxs = []
        for i in back_output_idxs:
            total_back_output_idxs += i

        send_queue_idxs = total_back_input_idxs
        recv_queue_idxs = total_back_output_idxs
        for back_dummy_input in back_dummy_inputs:
            for di in back_dummy_input:
                communication_size += get_size(di)
        
        # print(communication_size + get_size(raw_output_dummy))
        communication_size += get_size(raw_output_dummy)
        inference_time += get_time(lib.get_lib(), dev, tvm.runtime.save_param_dict(lib.get_params()), input_idxs, output_idxs, graph_json_strs, dummy_inputs)

    # Middle
    if len(input_idxs) == 3:
        front_input_idxs, front_output_idxs, front_graph_json_strs, front_dummy_inputs = get_model_info(partition_points[:2])
        server_input_idxs, server_output_idxs, _, server_dummy_inputs= get_model_info(partition_points[1:3])
        back_input_idxs, back_output_idxs, back_graph_json_strs, back_dummy_inputs = get_model_info(partition_points[2:4])

        total_front_output_idxs = []
        for i in front_output_idxs:
            total_front_output_idxs += i

        total_server_input_idxs = []
        for i in server_input_idxs:
            total_server_input_idxs += i

        total_back_input_idxs = []
        for i in back_input_idxs:
            total_back_input_idxs += i

        total_front_output_idxs = []
        for i in front_output_idxs:
            total_front_output_idxs += i

        total_server_output_idxs = []
        for i in server_output_idxs:
            total_server_output_idxs += i

        total_back_input_idxs = []
        for i in back_input_idxs:
            total_back_input_idxs += i

        send_queue_idxs = total_server_input_idxs
        pass_queue_idxs = np.intersect1d(total_front_output_idxs, total_back_input_idxs)
        recv_queue_idxs = np.intersect1d(total_server_output_idxs, total_back_input_idxs)
        
        total_network_time = 0
        total_inference_time = 0
        ### TIME
        inputs = list(front_input_idxs[0]) + list(server_input_idxs[0]) + list(back_input_idxs[0])
        input_dummys = list(front_dummy_inputs[0]) + list(server_dummy_inputs[0]) + list(back_dummy_inputs[0])
        for data_idx in list(send_queue_idxs) + list(recv_queue_idxs):
            idx = inputs.index(data_idx)
            # msg_size = len(pickle.dumps(input_dummys[idx]))
            # total_network_time += network_simulator(msg_size, 'wifi')
            communication_size += get_size(input_dummys[idx])

        inference_time += get_time(lib.get_lib(), dev, tvm.runtime.save_param_dict(lib.get_params()), input_idxs, output_idxs, graph_json_strs, dummy_inputs)

    print("communication size", communication_size, "inference", inference_time)
