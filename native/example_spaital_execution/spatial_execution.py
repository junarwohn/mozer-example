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
    print("top-1 id: {}, class name: {}".format(top1, synset[top1]))

def func(recv_conn, send_conn, idx, ohint):
    
    dev = tvm.device("cuda", 0)

    stime = time.time()
    lib = tvm.runtime.load_module(f"mod{idx}.so")

    with open(f"mod{idx}.params", "rb") as param_file:
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
    # with open("mod.json", "w") as json_file:
    #     json.dump(json.loads(mod_str), json_file)

    # with open("mod.params", "wb") as param_file:
    #     param_file.write(param_bytes)

    # target = 'cuda -arch=sm_86'
    # dev = tvm.device("cuda", 0)

    # #########################################

    # mod = mod['main']
    # ann = run_opt_pass(mod, transform.ToGraphNormalForm())
    # mod = tvm.IRModule.from_expr(ann)['main']
    # with tvm.transform.PassContext(opt_level=4):
    #     lib = relay.build(mod, target, params=params)

    # lib.export_library("mod.so")
    # param_bytes = tvm.runtime.save_param_dict(lib.get_params())
    # with open("lib.params", "wb") as f:
    #     f.write(param_bytes)

    # #########################################

    # relay_slicer = TVMSlicer()
    # split_config = [{"op_name": "add", "op_index": args.partition}]
    # subgraphs, input_name_hints, output_name_hints = relay_slicer.slice_relay_graph(mod['main'], split_config, params)
    
    # data_info = [input_name_hints, output_name_hints]

    # with open("data_info", "wb") as fp:   #Pickling
    #     pickle.dump(data_info, fp)
    
    with open("data_info", "rb") as fp:   # Unpickling
        data_info = pickle.load(fp)

    input_name_hints, output_name_hints = data_info

    src_iter = 10000

    # Single execution
    # input_recv_conn, input_send_conn = Pipe(duplex=False)
    input_recv_conn, input_send_conn = Pipe()
    # output_recv_conn, output_send_conn = Pipe(duplex=False)
    output_recv_conn, output_send_conn = Pipe()

    p1 = Process(target=func, args=(input_recv_conn, output_send_conn, '', [0]))
    p_src = Process(target=src, args=(input_send_conn, src_iter))
    p_sink = Process(target=sink, args=(output_recv_conn,))

    stime = time.time()
    p1.start()
    p_src.start()
    p_sink.start()

    p1.join()
    p_src.join()
    p_sink.join()
    etime = time.time()

    print(f"Single process {etime - stime}")

    # 2 Dual Process : Pipelining

    # target = 'cuda -arch=sm_86'
    # dev = tvm.device("cuda", 0)

    # mod = subgraphs[0]
    # ann = run_opt_pass(mod, transform.ToGraphNormalForm())
    # mod = tvm.IRModule.from_expr(ann)['main']

    # with tvm.transform.PassContext(opt_level=4):
    #     lib = relay.build(mod, target, params=params)

    # lib.export_library("mod0.so")
    # param_bytes = tvm.runtime.save_param_dict(lib.get_params())
    # with open("mod0.params", "wb") as f:
    #     f.write(param_bytes)


    # mod = subgraphs[1]
    # ann = run_opt_pass(mod, transform.ToGraphNormalForm())
    # mod = tvm.IRModule.from_expr(ann)['main']

    # with tvm.transform.PassContext(opt_level=4):
    #     lib = relay.build(mod, target, params=params)

    # lib.export_library("mod1.so")
    # param_bytes = tvm.runtime.save_param_dict(lib.get_params())
    # with open("mod1.params", "wb") as f:
    #     f.write(param_bytes)

    # input_recv_conn, input_send_conn = Pipe(duplex=False)
    input_recv_conn, input_send_conn = Pipe()
    # mid_recv_conn, mid_send_conn = Pipe(duplex=False)
    mid_recv_conn, mid_send_conn = Pipe()
    # output_recv_conn, output_send_conn = Pipe(duplex=False)
    output_recv_conn, output_send_conn = Pipe()

    p_src = Process(target=src, args=(input_send_conn, src_iter))
    p1 = Process(target=func, args=(input_recv_conn, mid_send_conn, 0, output_name_hints[0]))
    p2 = Process(target=func, args=(mid_recv_conn, output_send_conn, 1, output_name_hints[1]))
    p_sink = Process(target=sink, args=(output_recv_conn,))

    stime = time.time()
    p1.start()
    p2.start()
    p_src.start()
    p_sink.start()

    p1.join()
    p2.join()
    p_src.join()
    p_sink.join()
    etime = time.time()

    print(f"Pipeline process {etime - stime}")