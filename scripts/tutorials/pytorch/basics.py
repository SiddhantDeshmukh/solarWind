# Just a basic tutorial for PyTorch going through its tensors and later
# with its deep learning
# %%
# Basics of tensors (a replacement to Numpy?)
# =========================================================================
import torch

# Empty 5x3 array of floats (whatever values are in memory)
x = torch.empty(5, 3)
print(x)

# Random 5x3 array of floats
x = torch.rand(5, 3)
print(x)

# 5x3 matrix of zeros, dtype 'long'
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# Construct tensor directly from data
x = torch.tensor([5.5, 3])
print(x)

# Create tensor based on existing tensor. Methods will reuse properties of
# input tensor (e.g. dtype), unless new values provided by user
x = x.new_ones(5, 3, dtype=torch.double)  # new_* methods take in sizes
print(x)
print(x.size())

x = torch.randn_like(x, dtype=torch.float)  # override dtype
print(x)  # result has same size
print(x.size())
# =========================================================================

# %%
# Operations
# =========================================================================
# Addition
y = torch.rand(5, 3)
print(x + y)

# Alternatively, can use the torch function
print(torch.add(x, y))

# Providing output tensor as argument from addition
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

# In-place
y.add_(x)
print(y)
# =========================================================================

# %%
# Utilities
# =========================================================================
# NumPy fancy indexing works in PyTorch
print(x[:, 1])

# Resize/reshape tensor using 'torch.view'
x = torch.randn(4, 4)  # 2D
y = x.view(16)  # into 1D
z = x.view(-1, 8)  # size '-1' inferred from other dim (2D: 2 x 8)

print(x.size(), y.size(), z.size())

# For one-element tensor, use '.item()' to get value as Python number
x = torch.randn(1)
print(x)
print(x.item())
# =========================================================================

# %%
# NumPy bridge
# =========================================================================
# Converting Torch <-> NumPy is very simple. Torch Tensor and NumPy array
# share underlying memory locations (as long as Torch Tensor on CPU!) and
# *changing one will change the other*
# Torch to NumPy
a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

# If we add to 'a' (Tensor), we change 'b' (NumPy)
a.add_(1)
print(a)
print(b)

# NumPy to Torch
import numpy as np

a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

# All Tensors except CharTensor support NumPy conversion
# =========================================================================

# %%
# CUDA Tensors
# =========================================================================
# Tensors can be moved onto any device using '.to()' method
# This cell will execeute *only if* CUDA is available
# 'torch.device' objects are used to move tensors in/out of GPU
if torch.cuda.is_available():
  device = torch.device("cuda")  # CUDA device object
  y = torch.ones_like(x, device=device)  # directly create tensor on GPU
  x = x.to(device)  # send tensor to GPU
  z = x + y
  print(z)
  print(z.to("cpu", torch.double))  # '.to()' can change dtype as well
