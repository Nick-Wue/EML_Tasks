import torch 
from eml.ext.linear_python.Layer import Layer
from eml.ext.linear_cpp.Layer import Layer_cpp
W = torch.Tensor([[1,2,3],
                  [4,5,6]])
W.requires_grad = True

X = torch.Tensor([[7,8,9],
                  [10,11,12]])

X_test_copy = torch.Tensor([[7,8,9],
                  [10,11,12]])
X_test_copy.requires_grad = True
X.requires_grad = True

dLdY = torch.Tensor([[1,2],
                    [2,3]])

linear_Layer = torch.nn.Linear(in_features=3, out_features=2, bias=False)
linear_Layer.weight = torch.nn.Parameter(W)

dLdW = dLdY.T @ X
dLdX = dLdY @ W

print("Manually calculated Gradients: \n")
print(dLdX)
print(dLdW)

man_out = X @ W.T
print(man_out)


print("Gradients with torch linear layer: \n")
out = linear_Layer.forward(X)
out.backward(dLdY)
print(linear_Layer.weight.grad)
print(X.grad)

test_layer = Layer(3, 2)
test_layer.weight = torch.nn.Parameter(W)


print("Gradients and result with custom linear layer: \n")
out_test = test_layer.forward(X_test_copy)
out_test.backward(dLdY)
print(test_layer.weight.grad)
print(X_test_copy)

print("Gradient of weight and result with custom cpp Function: \n")
cpp_layer = Layer_cpp(3,2)
cpp.weight = torch.nn.Parameter(W)
test_cpp = cpp_layer.forward(X)
print(test_cpp)
test_cpp.backward(dLdY)
print(cpp_layer.weight.grad)