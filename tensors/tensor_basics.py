import torch
import numpy as np

# Initialize tensors
data = [[1, 2], [3, 4]]
x = torch.tensor(data)

# Initialize from numpy
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# Initialize from another tensor
x_ones = torch.ones_like(x)
print(f"Ones Tensor: \n {x_ones}")

x_rand = torch.rand_like(x, dtype=torch.float)
print(f"Random Tensor: \n {x_rand}")

# Initialize with random or constant values
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
print(f"Random Tensor: \n {rand_tensor}")
print(f"Ones Tensor: \n {ones_tensor}")
print(f"Zeros Tensor: \n {zeros_tensor}")

# Tensor attributes
tensor = torch.rand(3, 4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# Tensor operations
if torch.accelerator.is_available():
    tensor = tensor.to(torch.accelerator.current_accelerator())
    print(f"Device tensor is stored on: {tensor}")

# Standard numpy-like indexing and slicing
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[:, -1]}")
tensor[:, 1] = 0
print(f"New tensor: {tensor}")

# Join tensors
tensor_1 = torch.cat([tensor, tensor, tensor], dim=1)
print(f"New tensor: {tensor_1}")

# Arithmetic operations
# This computes the matrix multiplication between two tensors.
tensor_a = tensor @ tensor.T
tensor_b = tensor.matmul(tensor.T)
tensor_c = torch.randn_like(tensor_a)
torch.matmul(tensor, tensor.T, out=tensor_c)

# This computes the element-wise product.
tensor_x = tensor * tensor
tensor_y = tensor.mul(tensor)
tensor_z = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=tensor_z)

# Single-element tensors
tensor_agg = tensor.sum()
tensor_item = tensor_agg.item()
print(f"Tensor agg: {tensor_agg}\ntype: {type(tensor_agg)}")

# In-place operations
print(f"Tensor: {tensor}")
tensor.add_(5)
print(f"Tensor: {tensor}")

# Tensor to NumPy array
ones_tensor = torch.ones(5)
print(f"Tensor: {ones_tensor}")
new_tensor = ones_tensor.numpy()
print(f"New tensor: {new_tensor}")

# A change in the tensor reflects in the NumPy array.
ones_tensor.add_(1)
print(f"Mutate tensor: {ones_tensor}")
print(f"New tensor: {new_tensor}")

# NumPy array to Tensor
numpy_ones = np.ones(5)
tensor_from_numpy = torch.from_numpy(numpy_ones)
print(f"Numpy: {numpy_ones}")
print(f"Tensor: {tensor_from_numpy}")

# Changes in the NumPy array reflects in the tensor.
np.add(numpy_ones, 1, out=numpy_ones)
print(f"Tensor: {tensor_from_numpy}")
print(f"Numpy: {numpy_ones}")