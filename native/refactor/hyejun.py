from tvm.relay import ExprMutator

@transform.function_pass(opt_level=0)
class DW2InsertZeroBiasAddPass:
    def transform_function(self, func, mod, _):
        class DW2InsertZeroBiasAdd(ExprMutator):
            def __init__(self):
                super().__init__()
                self.is_converted = False
                self.conv2d = op.get("nn.conv2d")
                self.bias_add = op.get("nn.bias_add")
                self.add = op.get("add")
                self.quantize_info = op.get("annotation.quantize_info")
            def visit_function(self, fn):
                fn = super().visit_function(fn)
                new_params = [self.visit(x) for x in fn.params]
                new_body = fn.body
                if fn.attrs :
                    fn_attrs = dict(fn.attrs)
                    if ("Composite" in fn_attrs.keys()) and ("dnnweaver2"in fn_attrs["Composite"]):
                        new_body = self.rewrite_call(fn.body)
                return Function(list(new_params), new_body, fn.ret_type, fn.type_params, fn.attrs)
            def rewrite_call(self, call):
                args = [self.rewrite_call(arg) if isinstance(arg, Call) else self.visit(arg) for arg in call.args]
                new_args = []
                for i, arg in enumerate(args) :
                    if (
                        op_instance_check(arg) and
                        (call.op != self.add and call.op != self.bias_add) and
                        isinstance(args[i-1], Constant)
                        ) :
                        data = arg.args[0] if arg.op == self.quantize_info else arg
                        if (
                            op_instance_check(data) and
                            data.op == self.conv2d and
                            check_conv2d(data)
                            ) :
                            zero_bias = relay.const(value=np.zeros(shape=(int(data.attrs.channels)), dtype=‘int32’)) # TODO: This cord is hard cording for DW2
                            bias_add = nn.bias_add(arg, zero_bias, axis=3)
                            new_args.append(bias_add)
                        else :
                            new_args.append(arg)
                    else :
                        new_args.append(arg)
                return Call(self.visit(call.op), new_args, call.attrs)
        return DW2InsertZeroBiasAdd().visit(func)

@transform.function_pass(opt_level=0)
class DW2AnnotateQuantizeInfoPass:
    def transform_function(self, func, mod, _):
        class DW2AnnotateQuantizeInfo(ExprMutator):
            def __init__(self):
                super().__init__()
                self.simulated_quantize = op.get(“relay.op.annotation.simulated_quantize”)
                self.cast_hint = op.get(“annotation.cast_hint”)
                self.stop_fusion = op.get(“annotation.stop_fusion”)
            def annotate_quantiztion_info(self, args):
                new_args = []
                for arg in args:
                    orgin_arg = arg
                    dtype = None
                    if (
                        op_instance_check(arg) and
                        arg.op == self.stop_fusion
                        ):
                        arg = arg.args[0]
                    if (
                        op_instance_check(arg) and
                        arg.op == self.cast_hint
                        ):
                        dtype = arg.attrs.dtype
                        arg = arg.args[0]
                    if (
                        op_instance_check(arg) and
                        arg.op == self.simulated_quantize
                        ):
                        _, dom_scale, clip_min, clip_max = arg.args
                        dom_scale = float(dom_scale.data.numpy())
                        clip_min = int(clip_min.data.numpy())
                        clip_max = int(clip_max.data.numpy())
                        arg = op.annotation.quantize_info(data=orgin_arg,
                                                          dom_scale=dom_scale,
                                                          clip_min=clip_min,
                                                          clip_max=clip_max,
                                                          dtype=dtype if dtype != None else [“”],
                                                          **arg.attrs)
                    new_args.append(arg)
                return new_args
            def visit_call(self, call):
                args = [self.visit(arg) for arg in call.args]
                if not ( call.op in [self.simulated_quantize, self.cast_hint, self.stop_fusion] ) :
                    new_args = self.annotate_quantiztion_info(args)
                    return Call(call.op, new_args, call.attrs)
                return Call(self.visit(call.op), args, call.attrs)
        return DW2AnnotateQuantizeInfo().visit(func)