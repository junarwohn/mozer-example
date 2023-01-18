from multiprocessing import Process, Pipe, set_start_method
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
import json
import pickle

################################################

from argparse import ArgumentParser

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


def src(input_send_conn, iter):
    for _ in range(iter):
        in_dict = dict()
        in_dict['input_1'] = data
        input_send_conn.send(in_dict)
    input_send_conn.send(None)

def sink(output_recv_conn):
    while True:
        result = output_recv_conn.recv()
        if not result:
            break
        for k in result:
            top1 = np.argmax(result[k])

def func(recv_conn, send_conn, idx, cuda_idx, ohint):

    if cuda_idx == 0:
        dev = tvm.device("cuda", 0)
    elif cuda_idx == 1:
        dev = tvm.device("cuda", 1)

    stime = time.time()
    lib = tvm.runtime.load_module(f"unet{idx}.so")

    with open(f"unet{idx}.params", "rb") as param_file:
        params = bytearray(param_file.read())
    model = graph_executor.GraphModule(lib["default"](dev))
    model.load_params(params)
    etime = time.time()
    print("Graph Module", etime - stime)

    while True:
        data = recv_conn.recv()
        if data:
            for k in data:
                model.set_input(k, data[k])
            model.run()
            out_dict = dict()
            for i, out_name in enumerate(ohint):
                out_dict[out_name] = model.get_output(i).numpy()
            send_conn.send(out_dict)
        else:
            send_conn.send(data)
            break

if __name__ == "__main__":
    set_start_method('spawn')

    shape_dict = {"input_1": data.shape}
    mod, params = relay.frontend.from_keras(model_keras, shape_dict)
    
    # mod_str = tvm.ir.save_json(mod)
    # param_bytes = tvm.runtime.save_param_dict(params)
    # with open("unet.json", "w") as json_file:
    #     json.dump(json.loads(mod_str), json_file)

    # with open("unet.params", "wb") as param_file:
    #     param_file.write(param_bytes)

    # target = 'cuda -arch=sm_86'
    # dev = tvm.device("cuda", 0)

    # #########################################

    # mod = mod['main']
    # ann = run_opt_pass(mod, transform.ToGraphNormalForm())
    # mod = tvm.IRModule.from_expr(ann)['main']
    # with tvm.transform.PassContext(opt_level=4):
    #     lib = relay.build(mod, target, params=params)

    # lib.export_library("unet.so")
    # param_bytes = tvm.runtime.save_param_dict(lib.get_params())
    # with open("uent_lib.params", "wb") as f:
    #     f.write(param_bytes)

    # #########################################

    relay_slicer = TVMSlicer()
    split_config = [{"op_name": "nn.leaky_relu", "op_index": args.partition}]
    subgraphs, input_name_hints, output_name_hints = relay_slicer.slice_relay_graph(mod['main'], split_config, params)
    
    data_info = [input_name_hints, output_name_hints]

    with open("data_info", "wb") as fp:   #Pickling
        pickle.dump(data_info, fp)
    
    with open("data_info", "rb") as fp:   # Unpickling
        data_info = pickle.load(fp)

    input_name_hints, output_name_hints = data_info

    src_iter = 1000
    # src_iter = 2

    # Single execution
    # input_recv_conn, input_send_conn = Pipe(duplex=False)
    input_recv_conn, input_send_conn = Pipe()
    # output_recv_conn, output_send_conn = Pipe(duplex=False)
    output_recv_conn, output_send_conn = Pipe()

    p1 = Process(target=func, args=(input_recv_conn, output_send_conn, '', 0, [0]))
    p_src = Process(target=src, args=(input_send_conn, src_iter))
    p_sink = Process(target=sink, args=(output_recv_conn,))

    p1.start()
    p_sink.start()
    time.sleep(10)
    stime = time.time()
    p_src.start()

    p1.join()
    p_src.join()
    p_sink.join()
    etime = time.time()

    print(f"Single process {etime - stime}")

    # 2 Dual Process : Pipelining

    target = 'cuda -arch=sm_86'
    dev = tvm.device("cuda", 0)

    mod = subgraphs[0]
    ann = run_opt_pass(mod, transform.ToGraphNormalForm())
    mod = tvm.IRModule.from_expr(ann)['main']

    with tvm.transform.PassContext(opt_level=4):
        lib = relay.build(mod, target, params=params)

    lib.export_library("unet0.so")
    param_bytes = tvm.runtime.save_param_dict(lib.get_params())
    with open("unet0.params", "wb") as f:
        f.write(param_bytes)


    target = 'cuda -arch=sm_75'
    dev = tvm.device("cuda", 1)

    mod = subgraphs[1]
    ann = run_opt_pass(mod, transform.ToGraphNormalForm())
    mod = tvm.IRModule.from_expr(ann)['main']

    with tvm.transform.PassContext(opt_level=4):
        lib = relay.build(mod, target, params=params)

    lib.export_library("unet1.so")
    param_bytes = tvm.runtime.save_param_dict(lib.get_params())
    with open("unet1.params", "wb") as f:
        f.write(param_bytes)

    # input_recv_conn, input_send_conn = Pipe(duplex=False)
    input_recv_conn, input_send_conn = Pipe()
    # mid_recv_conn, mid_send_conn = Pipe(duplex=False)
    mid_recv_conn, mid_send_conn = Pipe()
    # output_recv_conn, output_send_conn = Pipe(duplex=False)
    output_recv_conn, output_send_conn = Pipe()

    p_src = Process(target=src, args=(input_send_conn, src_iter))
    p1 = Process(target=func, args=(input_recv_conn, mid_send_conn, 0, 0, output_name_hints[0]))
    p2 = Process(target=func, args=(mid_recv_conn, output_send_conn, 1, 1, output_name_hints[1]))
    p_sink = Process(target=sink, args=(output_recv_conn,))

    p1.start()
    p2.start()
    p_sink.start()
    time.sleep(10)
    stime = time.time()
    p_src.start()

    p1.join()
    p2.join()
    p_src.join()
    p_sink.join()
    etime = time.time()

    print(f"Pipeline process {etime - stime}")