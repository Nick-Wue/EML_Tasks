import torch
import eml_ext_function_cpp

class FunctionCpp(torch.autograd.Function):
          
    def forward(ctx, input, weight):
        in_data = input.contiguous()
        weight_data = weight.contiguous()
        out = eml_ext_function_cpp.forward(in_data, weight_data)
        ctx.save_for_backward(input, weight)
        return out
    
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        in_data = input.contiguous()
        weight_data = weight.contiguous()
        grad_output_data = grad_output.contiguous()
        grad_input, grad_weight = eml_ext_function_cpp.backward(grad_output_data, in_data, weight_data)
        return grad_input, grad_weight
    