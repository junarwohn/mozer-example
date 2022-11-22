import tvm
import os
import numpy as np
from mozer.slicer.SlicingMachine import TVMSlicer
from tvm.relay.testing import run_opt_pass
from tvm import relay
from tvm.relay import transform, build_module
from tensorflow import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"

################################################

def dequant(node, scale=7.0, zero_point=18.0):
    deqnode = relay.cast(node, dtype='float32')
    deqnode = relay.divide(deqnode, relay.const(scale))
    deqnode = relay.add(deqnode, relay.const(zero_point))
    return deqnode

def quant(node, scale=7.0, zero_point=18.0):
    qnode = relay.subtract(node, relay.const(zero_point))
    qnode = relay.multiply(qnode, relay.const(scale))
    qnode = relay.round(qnode)
    qnode = relay.clip(qnode, a_min=-128.0, a_max=127.0)
    qnode = relay.cast(qnode, dtype='int8')
    return qnode

################################################

model = keras.models.load_model("./simple_model.h5")
data = np.random.randint(0, 256, size=(1, 256,256,3)) / 255
data = data.transpose([0, 3, 1, 2])

shape_dict = {"input_1": data.shape}

################################################

mod, params = relay.frontend.from_keras(model, shape_dict)

################################################

relay_slicer = TVMSlicer()
split_config = [{"op_name": "nn.conv2d", "op_index": 2}]

# subgraphs, inputs, outputs, output_names = relay_slicer.slice_relay_graph(mod['main'], split_config, params)
subgraphs, input_name_hints, output_name_hints = relay_slicer.slice_relay_graph(mod['main'], split_config, params, is_quantize=True)

print(input_name_hints)
print(output_name_hints)
exit()
print("################################################")
print(subgraphs[0])
print("################################################")
print(subgraphs[1])
print("################################################")

################################################
from tvm.contrib import relay_viz
graph_attr = {"color": "red"}
node_attr = {"color": "blue"}
edge_attr = {"color": "black"}
def get_node_attr(node):
    if "nn.conv2d" in node.type_name and "NCHW" in node.detail:
        return {
            "fillcolor": "green",
            "style": "filled",
            "shape": "box",
        }
    if "Var" in node.type_name:
        return {"shape": "ellipse"}
    return {"shape": "box"}
    
dot_plotter = relay_viz.DotPlotter(
    graph_attr=graph_attr,
    node_attr=node_attr,
    edge_attr=edge_attr,
    get_node_attr=get_node_attr)

viz = relay_viz.RelayVisualizer(
    mod,
    relay_param=params,
    plotter=dot_plotter,
    parser=relay_viz.DotVizParser())
viz.render("total")

ann = run_opt_pass(subgraphs[0], transform.ToGraphNormalForm())
mod1 = tvm.IRModule.from_expr(ann)

viz = relay_viz.RelayVisualizer(
    mod1,
    relay_param=params,
    plotter=dot_plotter,
    parser=relay_viz.DotVizParser())
viz.render("slice_0")

ann = run_opt_pass(subgraphs[1], transform.ToGraphNormalForm())
mod2 = tvm.IRModule.from_expr(ann)

viz = relay_viz.RelayVisualizer(
    mod2,
    relay_param=params,
    plotter=dot_plotter,
    parser=relay_viz.DotVizParser())
viz.render("sliced_1")

exit()
# Get input nodes

# # Exclude 1st model : get input directly at beginning obviously

for subgraph in subgraphs[1:]:
    new_input_vars = []
    for free_var in relay.analysis.free_vars(subgraph):
        new_input_var = tvm.relay.expr.Var(free_var.name_hint+"_deq", relay.TensorType(free_var.type_annotation.shape, 'int8'))
        new_input_vars.append([free_var, dequant(new_input_var)])

modmod = subgraphs[1]

new_params = []
for ov, nv in new_input_vars:
    new_anf = tvm.relay.expr.Let(ov, nv, modmod)
    modmod = new_anf 

################################################

# Get output nodes

def setting_outputs(anf, name_hints, outputs, names):
    if isinstance(anf, tvm.relay.Function):
        return tvm.relay.Function(
            anf.params,
            setting_outputs(anf.body, name_hints, outputs, names),
            anf.ret_type,
            anf.type_params,
            anf.attrs,
        )
    if isinstance(anf, tvm.relay.expr.Let):
        value = anf.value
        if anf.var.name_hint in name_hints:
            print(anf.var.name_hint)
            outputs.append(anf)
        return tvm.relay.expr.Let(
            anf.var,
            value,
            setting_outputs(anf.body, name_hints, outputs, names),
        )
    else:
        new_outputs = []
        print(len(outputs))
        for o in outputs:
            new_outputs.append(o.var)
            names.append(o.var.name_hint)
        new_map = tvm.relay.expr.Tuple(list(map(quant, new_outputs)))
        return new_map

new_q_modmod = setting_outputs(modmod, output_names[1], [], [])

# print(output_names[1])
new_q_modmod = run_opt_pass(new_q_modmod, transform.ToGraphNormalForm())
print(new_q_modmod)


print("################################################")
