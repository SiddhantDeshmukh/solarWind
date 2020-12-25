# Autograd: Automatic differentiation
# =========================================================================
# The 'autograd' package is central to all NNs in PyTorch. It provides
# automatic differentiation for all operations on Tensors. It is a
# define-by-run framework, meaning backprop defined by how code is run, and
# every single iteration can be different

# 'torch.Tensor' is central class of package. Can set '.requires_grad=True'
# as attribute to track all operations on it. When computation is done, can
# call '.backward()' to auto compute all gradients! The gradient for this
# tensor will be accumulated into its '.grad' attribute

# Can use '.detach()' to stop tensor from tracking history

# To prevent tracking history (and using memory), can wrap code block in
# 'with torch.no_grad():'

# 'torch.Function' is the second important class here. 'Tensor' and
# 'Function' are interconnected and build up acyclic graph that encodes
# complete history of computation. Each tensor has '.grad_fn' attribute
# that references a 'Function' that created the 'Tensor' (except Tensors
# created by user; their '.grad_fn' is 'None')

# To compute derivatives, we use '.backward()' method on 'Tensor'. If
# 'Tensor' is a scalar, no arguments to '.backward()', otherwise need to
# specify 'gradient' argument that is tensor of matching shape
# %%
# Autograd implementation
# =========================================================================
import torch

x = torch.ones(2, 2, requires_grad=True)  # set 'requires_grad=True'
print(x)

# Tensor operation
y = x + 2  # 'grad_fn' attribute will remember addition
print(y)
z = y * y * 3  # 'grad_fn' attribute will remember multiplication
out = z.mean()  # 'grad_fn' attribute will remember mean
print(z, out)
# =========================================================================

# %%
# In-place grad initialisation
# =========================================================================
# '.requires_grad_(...)' changes existing Tensor's 'requires_grad' flag
# in-place. Input flag defaults to 'False'
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)
# =========================================================================

# %%
# Gradients and backpropagation
# =========================================================================
# Since 'out' contains single scalar, 'out.backward()' doesn't need any
# additional arguments and is equivalent to 'out.backward(torch.tensor(1.))'
out.backward()
print(x.grad)

# Warning: if we call this a second time, we will find that the memory has
# already been freed. We have to specify 'retain_graph=True' in
# '.backward()' to keep the graph for future calls
# =========================================================================

# %%
# Vector-Jacobian product example
# =========================================================================
# Generally speaking, 'torch.autograd' computes vector-Jacobian product!
x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
  y = y * 2

print(y)

# 'y' is no longer a scalar, so 'torch.autograd' could not compute full
# Jacobian directly. If we want just vector-Jacobian product, we simply
# pass vector to '.backward()' as an argument
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print(x.grad)

# Stop autograd from tracking history with 'with torch.no_grad()' wrap
print(x.requires_grad)
print((x**2).requires_grad)

with torch.no_grad():
  print((x**2).requires_grad)

# Use '.detach()' to get a new Tensor with same content but that does not
# require gradients
print(x.requires_grad)
y = x.detach()
print(y.requires_grad)
print(x.eq(y).all())