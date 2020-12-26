# Neural networks
# =========================================================================
# We can use the 'torch.nn' package to construct neural networks
# 'nn' depends on 'autograd' to define models and differentiate them. An
# 'nn.Module' contains layers and method 'forward(input)' that returns
# 'output'. Let's do a simple image classification feed-forward NN.
# Typical training procedure:
# - define network that has some learnable params (weights)
# - iterate over dataset of inputs
# - process input through network
# - compute loss
# - propagate gradients back into network's params
# - update weights, typically using simple rule: 
#   "weight -= learning_rate * gradient"
# %%
# Define network
# =========================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    # 1 input image, 6 output channels, 3x3 square conv kernel
    self.conv1 = nn.Conv2d(1, 6, 3)
    self.conv2 = nn.Conv2d(6, 16, 3)

    # affine operation y = Wx + b
    self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    # Max pooling over (2, 2) window
    x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
    # If size is square, can only specify single number
    x = F.max_pool2d(F.relu(self.conv2(x)), 2)
    x = x.view(-1, self.num_flat_features(x))
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)

    return x
  
  def num_flat_features(self, x):
    size = x.size()[1:]  # all dimensions except batch dimension
    num_features = 1
    for s in size:
      num_features *= s
    
    return num_features

net = Net()
print(net)

# We just define the 'forward()' function; 'backward()' is automatically
# defined using 'autograd'.

# %%
# Learnable parameters
# =========================================================================
# Learnable network params returned by 'net.parameters()'
params = list(net.parameters())
print(len(params))
print(params[0].size())  # 'conv1' layer weights

# Try a random 32x32 input (this is expected input size of this net, LeNet)
# For MNIST, resize images to 32x32
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

# Zero gradient buffers of all params and backprops with random gradients
net.zero_grad()
out.backward(torch.randn(1, 10))

# %%
# Loss function
# =========================================================================
# Takes (output, target) pair of inputs and computes value that estimates
# how far away the output is from the target
# Simple one is 'nn.MSELoss'
output = net(input)
target = torch.randn(10)  # dummy target
target = target.view(1, -1)
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

# When calling 'loss.backward()', whole graph differentiated wrt loss and
# all Tensors in graph with 'requires_grad=True' will have their '.grad'
# Tensor accumulated with gradient
print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

# %%
# Backprop
# =========================================================================
# Use 'loss.backward()' to backprop error. Need to clear existing gradients
# since this accumulates gradients.
# Let's call this and look at conv1's bias gradients before and after
net.zero_grad()  # zeroes gradient buffers of all params
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

# %%
# Updating weights
# =========================================================================
# Simplest update rule in practice is SGD:
# weight = weight - learning_rate * gradient
# 'torch.optim' provides various different update rules including SGD,
# Nesterov-SGD, Adam, RMSProp, etc.
import torch.optim as optim

# create optimiser
optimiser = optim.SGD(net.parameters(), lr=0.01)

# in training loop
optimiser.zero_grad()  # zeroes gradient buffers (gradients accumulate)
output = net(input)
loss = criterion(output, target)
loss.backward()
optimiser.step()  # update weights