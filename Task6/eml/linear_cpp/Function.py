import torch
class Function(torch.autograd.Function):
        
        
    def forward(ctx, input, weight):
        out = input @ weight.T
        ctx.save_for_backward(input, weight)
        return out
    
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input = grad_output @ weight
        grad_weight = grad_output.T @ input
        return grad_input, grad_weight
    