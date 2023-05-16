import torch 
from eml.ext.linear_python.Layer import Layer
W = torch.Tensor([[1,2,3],
                  [4,5,6]])
W.requires_grad = True

X = torch.Tensor([[7,8,9],
                  [10,11,12]])
X.requires_grad = True
dLdY = torch.Tensor([[1,2],
                    [2,3]])

linear_Layer = torch.nn.Linear(in_features=3, out_features=2, bias=False)
linear_Layer.weight = torch.nn.Parameter(W)

dLdW = dLdY.T @ X
dLdX = dLdY @ W

print(dLdX)
print(dLdW)

man_out = X @ W.T
print(man_out)



out = linear_Layer.forward(X)
out.backward(dLdY)
print(linear_Layer.weight.grad)
print(X.grad)

test_layer = Layer(3, 2)
test_layer.weight = torch.nn.Parameter(W)

out_test = test_layer.forward(X)
out_test.backward(dLdY)
print(test_layer.weight.grad)

