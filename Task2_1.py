import torch
import numpy as np
import ctypes
# Task 1.1
print("ones with (3,2)")
ones = torch.ones(3, 2)
print(ones)
print("zeros with (2,2)")
zeros = torch.zeros(2,2)
print(zeros)
print("rand with (2,5)")
rand = torch.rand(2, 5)
print(rand)
print("Ones_like with shape of rand")
ones_like = torch.ones_like(rand)
print(ones_like)

# Task 1.2

t_0 = [[0, 1, 2], [3, 4, 5]]
t_1 = [[6, 7, 8], [9, 10, 11]]
t_2 = [[12, 13, 14], [15, 16, 17]]
t_3 = [[18, 19, 20], [21, 22, 23]]

T = [t_0, t_1, t_2, t_3]
print(T)

torch_T = torch.tensor(T)
print(torch_T)

numpy_T = np.array(T)
print(numpy_T)

# Task 1.3
torch_T_from_np = torch.tensor(numpy_T)
print(torch_T_from_np)


#Task 2.1

P = torch.tensor(t_0)
Q = torch.tensor(t_1)

print("torch.add(P,Q) addition:")
print(torch.add(P,Q))

print("torch.mul(P,Q) multiplication:")
print(torch.mul(P,Q))

print("P + Q overloaded addition:")
print( P + Q)

print("P + Q overloaded multiplication:")
print( P * Q)

# Task 2.2
print("torch.matmul == @")
print(torch.matmul(P,Q.T))
print(P @ Q.T)

# Task 2.3
print("torch.sum(P) == 15")
print(torch.sum(P))

print("torch.max(Q) == 11")
print(torch.max(Q))

#Task 2.4

l_tensor_0 = torch.ones(2,2)
l_tensor_1 = torch.ones(2,2) + 1

# one tensor filled with 1 and one filled with 2
print(l_tensor_0)
print(l_tensor_1)

l_tmp = l_tensor_0
l_tmp[:] = 0 

# changing l_tmp changes l_tensor_0 as well
print( l_tmp)
print(l_tensor_0)

# with .clone().detach() l_tmp is assigned to a detached copy of l_tensor_1, so changes to l_tmp won't affect l_tensor_1
l_tmp = l_tensor_1.clone().detach()
l_tmp[:] = 0 
print(l_tmp)
print(l_tensor_1)

# Task 3.1

print(torch_T.stride())
print(torch_T.size())
print(torch_T.dtype)
print(torch_T.layout)
print(torch_T.device)

# Task 3.2
l_tensor_float = torch.tensor(torch_T.clone().detach(), dtype=torch.float32)
print(l_tensor_float)

# Task 3.3
# size and stride changed
l_tensor_fixed = l_tensor_float[:, 0, :]
print(l_tensor_fixed)
print(l_tensor_fixed.stride())
print(l_tensor_fixed.size())
print(l_tensor_fixed.dtype)
print(l_tensor_fixed.layout)
print(l_tensor_fixed.device)

# Task 3.4
# this view only accesses each second object, so the stride is doubled to access the according element in memory
# this also reduces the size of the first dimension to half of the original tensor
# the second dimension is fixed so the resulting view is only 2 dimensional
# similar to the previous example the 3. dimension is "reduced" to the second dimension of this view so the size of the 2. view dimension is that of 
# the 3. tensor dimension
l_tensor_complex_view = l_tensor_float[::2,1,:]
print(l_tensor_complex_view.stride())
print(l_tensor_complex_view.size())

# Task 3.5
# a new tensor is created that represents the view, this tensor is organized in memory in its standard form 
# (1 step for last dimension, size of dim 2 for 2. dimension and so on)
print(l_tensor_complex_view.contiguous().stride())

#Task 3.6
l_data_ptr = l_tensor_float.data_ptr()

memory_list = list()
for i in range(24):
    l_data_raw = (ctypes.c_float).from_address( l_data_ptr )
    memory_list.append(l_data_raw)
    l_data_ptr += 4

print(memory_list)