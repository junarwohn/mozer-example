import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"
from tensorflow.keras.applications import MobileNet
import tvm
import tvm.relay as relay
from tvm.contrib.download import download_testdata
import numpy as np
import cv2
import time

import mozer
# from SlicingMachine import TVMSlicer
from mozer.slicer.SlicingMachine import TVMSlicer
from tvm.relay.testing import run_opt_pass
from argparse import ArgumentParser
from tvm.relay import transform
from tvm.contrib import graph_executor, pipeline_executor, pipeline_executor_build






def preprocess_input(resized_image):
    image_data = np.asarray(resized_image).astype("float32")
    image_data = np.expand_dims(image_data, axis=0)
    image_data[:, :, :, 0] = 2.0 / 255.0 * image_data[:, :, :, 0] - 1
    image_data[:, :, :, 1] = 2.0 / 255.0 * image_data[:, :, :, 1] - 1
    image_data[:, :, :, 2] = 2.0 / 255.0 * image_data[:, :, :, 2] - 1
    return image_data





def get_sliced_modules(net, shape_dict, split_config):
    
    def slice_connect(subgraphs, input_name_hints, output_name_hints, pipe_config):
        # Global input
        for name_hint in input_name_hints[0]:
            pipe_config["input"][name_hint].connect(pipe_config[subgraphs[0]]["input"][name_hint])

        for i in range(len(subgraphs) - 1):
            in_out_mapper = dict()
            for idx, name in enumerate(output_name_hints[i]):
                in_out_mapper[name] = idx
            for idx, name in enumerate(output_name_hints[i]):
                in_out_mapper[name] = idx
                for name in input_name_hints[i + 1]:
                    pipe_config[subgraphs[i]]["output"][in_out_mapper[name]].connect(pipe_config[subgraphs[i + 1]]["input"][name])
        
        pipe_config[subgraphs[len(subgraphs) - 1]]["output"][0].connect(pipe_config["output"][0])

    

    mod, params = relay.frontend.from_keras(net, shape_dict)
    tvm.relay.backend.te_compiler.get().clear()
    
    relay_slicer = TVMSlicer()
    subgraphs, input_name_hints, output_name_hints = relay_slicer.slice_relay_graph(mod['main'], split_config, params)

    print(input_name_hints, output_name_hints)

    for idx, graph in enumerate(subgraphs):
        ann = run_opt_pass(graph, transform.ToGraphNormalForm())
        subgraphs[idx] =  tvm.IRModule.from_expr(ann)

    pipe_config = pipeline_executor_build.PipelineConfig()

    for subgraph in subgraphs:    
        pipe_config[subgraph].target = "cuda"
        pipe_config[subgraph].dev = tvm.device("cuda",0)
        pipe_config[subgraph].export_cc = "nvcc"

    slice_connect(subgraphs, input_name_hints, output_name_hints, pipe_config)

    with tvm.transform.PassContext(opt_level=3):
        pipeline_mod_factory = pipeline_executor_build.build(pipe_config)

    return pipeline_executor.PipelineModule(pipeline_mod_factory)


##################################################################



img_rows,img_cols = 224,224

image_path = "treefrog.jpg"
resized_image = cv2.resize(cv2.imread(image_path), (224, 224))

input_data = preprocess_input(resized_image)

net = tf.keras.applications.ResNet50(weights='imagenet', input_shape=(img_rows, img_cols, 3))

input_data = input_data.transpose([0, 3, 1, 2])
shape_dict = {"input_1": input_data.shape}


"""
resnet 50 IR

def @main(%input_1: Tensor[(1, 3, 224, 224), float32], %v_param_1: Tensor[(64, 3, 7, 7), float32], %v_param_2: Tensor[(64), float32], %v_param_3: Tensor[(64), float32], %v_param_4: Tensor[(64), float32], %v_param_5: Tensor[(64), float32], %v_param_6: Tensor[(64), float32], %v_param_19: Tensor[(256, 64, 1, 1), float32], %v_param_20: Tensor[(256), float32], %v_param_23: Tensor[(256), float32], %v_param_24: Tensor[(256), float32], %v_param_25: Tensor[(256), float32], %v_param_26: Tensor[(256), float32], %v_param_7: Tensor[(64, 64, 1, 1), float32], %v_param_8: Tensor[(64), float32], %v_param_9: Tensor[(64), float32], %v_param_10: Tensor[(64), float32], %v_param_11: Tensor[(64), float32], %v_param_12: Tensor[(64), float32], %v_param_13: Tensor[(64, 64, 3, 3), float32], %v_param_14: Tensor[(64), float32], %v_param_15: Tensor[(64), float32], %v_param_16: Tensor[(64), float32], %v_param_17: Tensor[(64), float32], %v_param_18: Tensor[(64), float32], %v_param_21: Tensor[(256, 64, 1, 1), float32], %v_param_22: Tensor[(256), float32], %v_param_27: Tensor[(256), float32], %v_param_28: Tensor[(256), float32], %v_param_29: Tensor[(256), float32], %v_param_30: Tensor[(256), float32], %v_param_31: Tensor[(64, 256, 1, 1), float32], %v_param_32: Tensor[(64), float32], %v_param_33: Tensor[(64), float32], %v_param_34: Tensor[(64), float32], %v_param_35: Tensor[(64), float32], %v_param_36: Tensor[(64), float32], %v_param_37: Tensor[(64, 64, 3, 3), float32], %v_param_38: Tensor[(64), float32], %v_param_39: Tensor[(64), float32], %v_param_40: Tensor[(64), float32], %v_param_41: Tensor[(64), float32], %v_param_42: Tensor[(64), float32], %v_param_43: Tensor[(256, 64, 1, 1), float32], %v_param_44: Tensor[(256), float32], %v_param_45: Tensor[(256), float32], %v_param_46: Tensor[(256), float32], %v_param_47: Tensor[(256), float32], %v_param_48: Tensor[(256), float32], %v_param_49: Tensor[(64, 256, 1, 1), float32], %v_param_50: Tensor[(64), float32], %v_param_51: Tensor[(64), float32], %v_param_52: Tensor[(64), float32], %v_param_53: Tensor[(64), float32], %v_param_54: Tensor[(64), float32], %v_param_55: Tensor[(64, 64, 3, 3), float32], %v_param_56: Tensor[(64), float32], %v_param_57: Tensor[(64), float32], %v_param_58: Tensor[(64), float32], %v_param_59: Tensor[(64), float32], %v_param_60: Tensor[(64), float32], %v_param_61: Tensor[(256, 64, 1, 1), float32], %v_param_62: Tensor[(256), float32], %v_param_63: Tensor[(256), float32], %v_param_64: Tensor[(256), float32], %v_param_65: Tensor[(256), float32], %v_param_66: Tensor[(256), float32], %v_param_79: Tensor[(512, 256, 1, 1), float32], %v_param_80: Tensor[(512), float32], %v_param_83: Tensor[(512), float32], %v_param_84: Tensor[(512), float32], %v_param_85: Tensor[(512), float32], %v_param_86: Tensor[(512), float32], %v_param_67: Tensor[(128, 256, 1, 1), float32], %v_param_68: Tensor[(128), float32], %v_param_69: Tensor[(128), float32], %v_param_70: Tensor[(128), float32], %v_param_71: Tensor[(128), float32], %v_param_72: Tensor[(128), float32], %v_param_73: Tensor[(128, 128, 3, 3), float32], %v_param_74: Tensor[(128), float32], %v_param_75: Tensor[(128), float32], %v_param_76: Tensor[(128), float32], %v_param_77: Tensor[(128), float32], %v_param_78: Tensor[(128), float32], %v_param_81: Tensor[(512, 128, 1, 1), float32], %v_param_82: Tensor[(512), float32], %v_param_87: Tensor[(512), float32], %v_param_88: Tensor[(512), float32], %v_param_89: Tensor[(512), float32], %v_param_90: Tensor[(512), float32], %v_param_91: Tensor[(128, 512, 1, 1), float32], %v_param_92: Tensor[(128), float32], %v_param_93: Tensor[(128), float32], %v_param_94: Tensor[(128), float32], %v_param_95: Tensor[(128), float32], %v_param_96: Tensor[(128), float32], %v_param_97: Tensor[(128, 128, 3, 3), float32], %v_param_98: Tensor[(128), float32], %v_param_99: Tensor[(128), float32], %v_param_100: Tensor[(128), float32], %v_param_101: Tensor[(128), float32], %v_param_102: Tensor[(128), float32], %v_param_103: Tensor[(512, 128, 1, 1), float32], %v_param_104: Tensor[(512), float32], %v_param_105: Tensor[(512), float32], %v_param_106: Tensor[(512), float32], %v_param_107: Tensor[(512), float32], %v_param_108: Tensor[(512), float32], %v_param_109: Tensor[(128, 512, 1, 1), float32], %v_param_110: Tensor[(128), float32], %v_param_111: Tensor[(128), float32], %v_param_112: Tensor[(128), float32], %v_param_113: Tensor[(128), float32], %v_param_114: Tensor[(128), float32], %v_param_115: Tensor[(128, 128, 3, 3), float32], %v_param_116: Tensor[(128), float32], %v_param_117: Tensor[(128), float32], %v_param_118: Tensor[(128), float32], %v_param_119: Tensor[(128), float32], %v_param_120: Tensor[(128), float32], %v_param_121: Tensor[(512, 128, 1, 1), float32], %v_param_122: Tensor[(512), float32], %v_param_123: Tensor[(512), float32], %v_param_124: Tensor[(512), float32], %v_param_125: Tensor[(512), float32], %v_param_126: Tensor[(512), float32], %v_param_127: Tensor[(128, 512, 1, 1), float32], %v_param_128: Tensor[(128), float32], %v_param_129: Tensor[(128), float32], %v_param_130: Tensor[(128), float32], %v_param_131: Tensor[(128), float32], %v_param_132: Tensor[(128), float32], %v_param_133: Tensor[(128, 128, 3, 3), float32], %v_param_134: Tensor[(128), float32], %v_param_135: Tensor[(128), float32], %v_param_136: Tensor[(128), float32], %v_param_137: Tensor[(128), float32], %v_param_138: Tensor[(128), float32], %v_param_139: Tensor[(512, 128, 1, 1), float32], %v_param_140: Tensor[(512), float32], %v_param_141: Tensor[(512), float32], %v_param_142: Tensor[(512), float32], %v_param_143: Tensor[(512), float32], %v_param_144: Tensor[(512), float32], %v_param_157: Tensor[(1024, 512, 1, 1), float32], %v_param_158: Tensor[(1024), float32], %v_param_161: Tensor[(1024), float32], %v_param_162: Tensor[(1024), float32], %v_param_163: Tensor[(1024), float32], %v_param_164: Tensor[(1024), float32], %v_param_145: Tensor[(256, 512, 1, 1), float32], %v_param_146: Tensor[(256), float32], %v_param_147: Tensor[(256), float32], %v_param_148: Tensor[(256), float32], %v_param_149: Tensor[(256), float32], %v_param_150: Tensor[(256), float32], %v_param_151: Tensor[(256, 256, 3, 3), float32], %v_param_152: Tensor[(256), float32], %v_param_153: Tensor[(256), float32], %v_param_154: Tensor[(256), float32], %v_param_155: Tensor[(256), float32], %v_param_156: Tensor[(256), float32], %v_param_159: Tensor[(1024, 256, 1, 1), float32], %v_param_160: Tensor[(1024), float32], %v_param_165: Tensor[(1024), float32], %v_param_166: Tensor[(1024), float32], %v_param_167: Tensor[(1024), float32], %v_param_168: Tensor[(1024), float32], %v_param_169: Tensor[(256, 1024, 1, 1), float32], %v_param_170: Tensor[(256), float32], %v_param_171: Tensor[(256), float32], %v_param_172: Tensor[(256), float32], %v_param_173: Tensor[(256), float32], %v_param_174: Tensor[(256), float32], %v_param_175: Tensor[(256, 256, 3, 3), float32], %v_param_176: Tensor[(256), float32], %v_param_177: Tensor[(256), float32], %v_param_178: Tensor[(256), float32], %v_param_179: Tensor[(256), float32], %v_param_180: Tensor[(256), float32], %v_param_181: Tensor[(1024, 256, 1, 1), float32], %v_param_182: Tensor[(1024), float32], %v_param_183: Tensor[(1024), float32], %v_param_184: Tensor[(1024), float32], %v_param_185: Tensor[(1024), float32], %v_param_186: Tensor[(1024), float32], %v_param_187: Tensor[(256, 1024, 1, 1), float32], %v_param_188: Tensor[(256), float32], %v_param_189: Tensor[(256), float32], %v_param_190: Tensor[(256), float32], %v_param_191: Tensor[(256), float32], %v_param_192: Tensor[(256), float32], %v_param_193: Tensor[(256, 256, 3, 3), float32], %v_param_194: Tensor[(256), float32], %v_param_195: Tensor[(256), float32], %v_param_196: Tensor[(256), float32], %v_param_197: Tensor[(256), float32], %v_param_198: Tensor[(256), float32], %v_param_199: Tensor[(1024, 256, 1, 1), float32], %v_param_200: Tensor[(1024), float32], %v_param_201: Tensor[(1024), float32], %v_param_202: Tensor[(1024), float32], %v_param_203: Tensor[(1024), float32], %v_param_204: Tensor[(1024), float32], %v_param_205: Tensor[(256, 1024, 1, 1), float32], %v_param_206: Tensor[(256), float32], %v_param_207: Tensor[(256), float32], %v_param_208: Tensor[(256), float32], %v_param_209: Tensor[(256), float32], %v_param_210: Tensor[(256), float32], %v_param_211: Tensor[(256, 256, 3, 3), float32], %v_param_212: Tensor[(256), float32], %v_param_213: Tensor[(256), float32], %v_param_214: Tensor[(256), float32], %v_param_215: Tensor[(256), float32], %v_param_216: Tensor[(256), float32], %v_param_217: Tensor[(1024, 256, 1, 1), float32], %v_param_218: Tensor[(1024), float32], %v_param_219: Tensor[(1024), float32], %v_param_220: Tensor[(1024), float32], %v_param_221: Tensor[(1024), float32], %v_param_222: Tensor[(1024), float32], %v_param_223: Tensor[(256, 1024, 1, 1), float32], %v_param_224: Tensor[(256), float32], %v_param_225: Tensor[(256), float32], %v_param_226: Tensor[(256), float32], %v_param_227: Tensor[(256), float32], %v_param_228: Tensor[(256), float32], %v_param_229: Tensor[(256, 256, 3, 3), float32], %v_param_230: Tensor[(256), float32], %v_param_231: Tensor[(256), float32], %v_param_232: Tensor[(256), float32], %v_param_233: Tensor[(256), float32], %v_param_234: Tensor[(256), float32], %v_param_235: Tensor[(1024, 256, 1, 1), float32], %v_param_236: Tensor[(1024), float32], %v_param_237: Tensor[(1024), float32], %v_param_238: Tensor[(1024), float32], %v_param_239: Tensor[(1024), float32], %v_param_240: Tensor[(1024), float32], %v_param_241: Tensor[(256, 1024, 1, 1), float32], %v_param_242: Tensor[(256), float32], %v_param_243: Tensor[(256), float32], %v_param_244: Tensor[(256), float32], %v_param_245: Tensor[(256), float32], %v_param_246: Tensor[(256), float32], %v_param_247: Tensor[(256, 256, 3, 3), float32], %v_param_248: Tensor[(256), float32], %v_param_249: Tensor[(256), float32], %v_param_250: Tensor[(256), float32], %v_param_251: Tensor[(256), float32], %v_param_252: Tensor[(256), float32], %v_param_253: Tensor[(1024, 256, 1, 1), float32], %v_param_254: Tensor[(1024), float32], %v_param_255: Tensor[(1024), float32], %v_param_256: Tensor[(1024), float32], %v_param_257: Tensor[(1024), float32], %v_param_258: Tensor[(1024), float32], %v_param_271: Tensor[(2048, 1024, 1, 1), float32], %v_param_272: Tensor[(2048), float32], %v_param_275: Tensor[(2048), float32], %v_param_276: Tensor[(2048), float32], %v_param_277: Tensor[(2048), float32], %v_param_278: Tensor[(2048), float32], %v_param_259: Tensor[(512, 1024, 1, 1), float32], %v_param_260: Tensor[(512), float32], %v_param_261: Tensor[(512), float32], %v_param_262: Tensor[(512), float32], %v_param_263: Tensor[(512), float32], %v_param_264: Tensor[(512), float32], %v_param_265: Tensor[(512, 512, 3, 3), float32], %v_param_266: Tensor[(512), float32], %v_param_267: Tensor[(512), float32], %v_param_268: Tensor[(512), float32], %v_param_269: Tensor[(512), float32], %v_param_270: Tensor[(512), float32], %v_param_273: Tensor[(2048, 512, 1, 1), float32], %v_param_274: Tensor[(2048), float32], %v_param_279: Tensor[(2048), float32], %v_param_280: Tensor[(2048), float32], %v_param_281: Tensor[(2048), float32], %v_param_282: Tensor[(2048), float32], %v_param_283: Tensor[(512, 2048, 1, 1), float32], %v_param_284: Tensor[(512), float32], %v_param_285: Tensor[(512), float32], %v_param_286: Tensor[(512), float32], %v_param_287: Tensor[(512), float32], %v_param_288: Tensor[(512), float32], %v_param_289: Tensor[(512, 512, 3, 3), float32], %v_param_290: Tensor[(512), float32], %v_param_291: Tensor[(512), float32], %v_param_292: Tensor[(512), float32], %v_param_293: Tensor[(512), float32], %v_param_294: Tensor[(512), float32], %v_param_295: Tensor[(2048, 512, 1, 1), float32], %v_param_296: Tensor[(2048), float32], %v_param_297: Tensor[(2048), float32], %v_param_298: Tensor[(2048), float32], %v_param_299: Tensor[(2048), float32], %v_param_300: Tensor[(2048), float32], %v_param_301: Tensor[(512, 2048, 1, 1), float32], %v_param_302: Tensor[(512), float32], %v_param_303: Tensor[(512), float32], %v_param_304: Tensor[(512), float32], %v_param_305: Tensor[(512), float32], %v_param_306: Tensor[(512), float32], %v_param_307: Tensor[(512, 512, 3, 3), float32], %v_param_308: Tensor[(512), float32], %v_param_309: Tensor[(512), float32], %v_param_310: Tensor[(512), float32], %v_param_311: Tensor[(512), float32], %v_param_312: Tensor[(512), float32], %v_param_313: Tensor[(2048, 512, 1, 1), float32], %v_param_314: Tensor[(2048), float32], %v_param_315: Tensor[(2048), float32], %v_param_316: Tensor[(2048), float32], %v_param_317: Tensor[(2048), float32], %v_param_318: Tensor[(2048), float32], %v_param_319: Tensor[(1000, 2048), float32], %v_param_320: Tensor[(1000), float32]) {
  %0 = nn.pad(%input_1, 0, pad_width=[[0, 0], [0, 0], [3, 3], [3, 3]]);
  %1 = nn.conv2d(%0, %v_param_1, strides=[2, 2], padding=[0, 0, 0, 0], channels=64, kernel_size=[7, 7]);
  %2 = nn.bias_add(%1, %v_param_2);
  %3 = nn.batch_norm(%2, %v_param_3, %v_param_4, %v_param_5, %v_param_6, epsilon=1.001e-05f);
  %4 = %3.0;
  %5 = nn.relu(%4);
  %6 = nn.pad(%5, 0, pad_width=[[0, 0], [0, 0], [1, 1], [1, 1]]);
  %7 = nn.max_pool2d(%6, pool_size=[3, 3], strides=[2, 2], padding=[0, 0, 0, 0]);
  %8 = nn.conv2d(%7, %v_param_19, padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1]);
  %9 = nn.bias_add(%8, %v_param_20);
  %10 = nn.batch_norm(%9, %v_param_23, %v_param_24, %v_param_25, %v_param_26, epsilon=1.001e-05f);
  %11 = nn.conv2d(%7, %v_param_7, padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1]);
  %12 = nn.bias_add(%11, %v_param_8);
  %13 = nn.batch_norm(%12, %v_param_9, %v_param_10, %v_param_11, %v_param_12, epsilon=1.001e-05f);
  %14 = %13.0;
  %15 = nn.relu(%14);
  %16 = nn.conv2d(%15, %v_param_13, padding=[1i64, 1i64, 1i64, 1i64], channels=64, kernel_size=[3, 3]);
  %17 = nn.bias_add(%16, %v_param_14);
  %18 = nn.batch_norm(%17, %v_param_15, %v_param_16, %v_param_17, %v_param_18, epsilon=1.001e-05f);
  %19 = %18.0;
  %20 = nn.relu(%19);
  %21 = nn.conv2d(%20, %v_param_21, padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1]);
  %22 = nn.bias_add(%21, %v_param_22);
  %23 = nn.batch_norm(%22, %v_param_27, %v_param_28, %v_param_29, %v_param_30, epsilon=1.001e-05f);
  %24 = %10.0;
  %25 = %23.0;
  %26 = add(%24, %25);
  %27 = nn.relu(%26);
"""

split_config_ok_1 = [{"op_name": "nn.relu", "op_index": 3},]

split_config_ok_2 = [{"op_name": "add", "op_index": 0},]

split_config_err = [{"op_name": "nn.relu", "op_index": 2},]

# sliced_modules = get_sliced_modules(net, shape_dict, split_config_ok_1)
# sliced_modules.set_input("input_1", tvm.nd.array(input_data))
# sliced_modules.run()
# outputs = sliced_modules.get_output()

# sliced_modules = get_sliced_modules(net, shape_dict, split_config_ok_2)
# sliced_modules.set_input("input_1", tvm.nd.array(input_data))
# sliced_modules.run()
# outputs = sliced_modules.get_output()

sliced_modules = get_sliced_modules(net, shape_dict, split_config_err)
sliced_modules.set_input("input_1", tvm.nd.array(input_data))
sliced_modules.run()
outputs = sliced_modules.get_output()