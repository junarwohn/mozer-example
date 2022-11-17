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
from tvm.contrib import relay_viz
from tvm.relay import build_module


def graph_split2(expr, split_conf, params=None):
    """Splitting the graph into a list of subgraphs"""

    def get_dep_var(sub_var_dep):
        return [var for var in sub_var_dep[len(sub_var_dep) - 1]["ref_nodes"]]

    def parse_dependency(value, snode_dep, new_input_idx):
        new_args = []
        need_update = False
        for var in value.args:
            is_free_var = False
            for dep in snode_dep[:-1]:
                if var in dep["nodes"]:
                    # Mark the previous subgraph node as a dependency.
                    dep["nodes"][var] += 1
                    dep["ref_nodes"][var] = dep["nodes"][var]
                    # The var of this call is a free_var
                    is_free_var = True
            # if the var of this call is a free_var, recreate it and give it a fixed input name.
            # if is_free_var:
            #     need_update = True
            #     new_args.append(relay.var(f"data_n_{new_input_idx}", var.checked_type))
            #     new_input_idx += 1
            # else:
            new_args.append(var)
        # if the 'tvm.relay.expr.Call' has a free_var, recreate it with new name as 'data_n_*'.
        if need_update:
            value = tvm.relay.expr.Call(
                value.op, new_args, value.attrs, value.type_args, value.span
            )
        return value, snode_dep, new_input_idx

    def merge_constant_expr(constant_expr, expr):
        # merge constant express with a express
        if not isinstance(constant_expr.body, tvm.relay.expr.Let):
            return tvm.relay.expr.Let(constant_expr.var, constant_expr.value, expr)

        return tvm.relay.expr.Let(
            constant_expr.var, constant_expr.value, merge_constant_expr(constant_expr.body, expr)
        )

    def _recursion(anf, pipeline_mods, split_conf, constant_expr):
        # Enumurate all operators of compute graph, then split the compute graph into a group of
        # subgraph.
        nonlocal operator_index_map
        nonlocal new_input_idx
        nonlocal snode_dep
        # Get last element in snode_dep : current node's dependency
        cur_node_dep = snode_dep[len(snode_dep) - 1]
        # If function -> decouple
        if isinstance(anf, tvm.relay.Function):
            return tvm.relay.Function(
                anf.params,
                _recursion(anf.body, pipeline_mods, split_conf, constant_expr),
                anf.ret_type,
                anf.type_params,
                anf.attrs,
            )
        # Function of Let
        if isinstance(anf, tvm.relay.expr.Let):
            value = anf.value
            # record the constant expr to make sure all sugraphs can find correct constant.
            if isinstance(value, tvm.relay.expr.Constant):
                # cosntant_expr is initally None
                if not constant_expr:
                    constant_expr = tvm.relay.expr.Let(anf.var, value, anf.var)
                else:
                    constant_expr = tvm.relay.expr.Let(anf.var, value, constant_expr)
            if isinstance(value, tvm.relay.expr.Call):
                new_args = []
                # build current var list
                cur_node_dep["nodes"][anf.var] = 0
                # Get the dependency information of the nodes.
                value, snode_dep, new_input_idx = parse_dependency(value, snode_dep, new_input_idx)
                if isinstance(value.op, tvm.ir.Op):
                    if value.op.name in operator_index_map:
                        operator_index_map[value.op.name] += 1
                    else:
                        operator_index_map[value.op.name] = 0
                    split_operator_name = split_conf[0]["op_name"] if split_conf else ""
                    split_operator_index = split_conf[0]["op_index"] if split_conf else ""
                    # if a operator name and repeating count in the network match with the values
                    # of the 'split configuration', then this place is where we should do the
                    # graph splitting.
                    if (
                        split_conf
                        and split_operator_name in operator_index_map
                        and operator_index_map[split_operator_name] >= split_operator_index
                    ):
                        # Do graph splitting.
                        split_conf.pop(0)
                        snode_dep.append({"nodes": {}, "ref_nodes": {}})
                        ann = _recursion(
                            anf.body,
                            pipeline_mods,
                            split_conf,
                            constant_expr,
                        )
                        snode_dep.pop()
                        dep_vars = get_dep_var(snode_dep)
                        # When the nodes of the current subgraph are the depedency node of another
                        # subgraph, we need to set them as the output of current subgraph.
                        body = relay.Tuple(dep_vars) if len(dep_vars) > 1 else anf.var
                        # when the operator of current subgraph uses previous subgraph constant
                        # as the argument of a "relay.expr.call", such constant may become a free
                        # varaible if the constant does not exist in the current subgraph.
                        # merge the previous constant with current subgraph to avoid such issue.
                        if constant_expr:
                            ann = merge_constant_expr(constant_expr, ann)
                        # ann = run_opt_pass(ann, transform.ToGraphNormalForm())
                        # mod = tvm.IRModule.from_expr(ann)
                        pipeline_mods.insert(0, ann)
                        # Return the last node of the current subgraph.
                        return tvm.relay.expr.Let(anf.var, value, body)
            return tvm.relay.expr.Let(
                anf.var,
                value,
                _recursion(anf.body, pipeline_mods, split_conf, constant_expr),
            )
        # Or End
        else:
            return anf
    
    def getting_inputs(mod):
        name_hints = []
        ann = run_opt_pass(mod.body, transform.ToGraphNormalForm())
        mod = tvm.IRModule.from_expr(ann)
        for param in mod['main'].params:
            name_hints.append(param.name_hint)
        return name_hints

    def setting_outputs(anf, name_hints, outputs):
        # Get last element in snode_dep : current node's dependency
        # If function -> decouple
        if isinstance(anf, tvm.relay.Function):
            return tvm.relay.Function(
                anf.params,
                setting_outputs(anf.body, name_hints, outputs),
                anf.ret_type,
                anf.type_params,
                anf.attrs,
            )
        # Function of Let
        if isinstance(anf, tvm.relay.expr.Let):
            value = anf.value
            # print(anf.var.name_hint)
            if anf.var.name_hint in name_hints:
                outputs.append(anf)
            # value, snode_dep, new_input_idx = parse_dependency(value, snode_dep, new_input_idx)
            return tvm.relay.expr.Let(
                anf.var,
                value,
                setting_outputs(anf.body, name_hints, outputs),
            )
        # Or End
        else:
            # new_map = 
            return anf

    snode_dep = [{"nodes": {}, "ref_nodes": {}}]
    pipeline_mods = []
    operator_index_map = {}
    # Used to tracking new input which caused by graph splitting.
    new_input_idx = 0
    constant_expr = None
    subgraph_split_conf = split_conf.copy()
    # Binding the parameters.
    if params:
        expr = build_module.bind_params_by_name(expr, params)
    anf = run_opt_pass(expr, transform.ToANormalForm())
    anf = run_opt_pass(anf, transform.InferType())
    ann = _recursion(
        anf,
        pipeline_mods,
        subgraph_split_conf,
        constant_expr,
    )
    # ann = run_opt_pass(ann.body, transform.ToGraphNormalForm())
    # mod = tvm.IRModule.from_expr(ann)
    pipeline_mods.insert(0, ann.body)
    inputs = []
    for mod in pipeline_mods:
        inputs.extend(getting_inputs(mod))
    print(inputs)
    total_outputs = []
    for mod in pipeline_mods:
        outputs = []
        setting_outputs(mod, inputs, outputs)
        total_outputs.append(outputs)
    return pipeline_mods, total_outputs


def setting_outputs(anf, outputs):
    # Get last element in snode_dep : current node's dependency
    # If function -> decouple
    if isinstance(anf, tvm.relay.Function):
        return tvm.relay.Function(
            anf.params,
            setting_outputs(anf.body, outputs),
            anf.ret_type,
            anf.type_params,
            anf.attrs,
        )
    # Function of Let
    if isinstance(anf, tvm.relay.expr.Let):
        value = anf.value
        # print(anf.var.name_hint)
        # if anf.var.name_hint in name_hints:
            # outputs.append(anf)
        # value, snode_dep, new_input_idx = parse_dependency(value, snode_dep, new_input_idx)
        return tvm.relay.expr.Let(
            anf.var,
            value,
            setting_outputs(anf.body, outputs),
        )
    # Or End
    else:
        new_outputs = []
        for o in outputs:
            new_outputs.append(o.var)
        # for o in outputs:
        #     new_outputs.append(tvm.relay.expr.Let(
        #             o.var,
        #             o.value,
        #             o.body,
        #     ))
        new_map = tvm.relay.expr.Tuple(new_outputs)
        return new_map