# Let's train a classifier. For loading in data, generally want to use
# packages that load data into a NumPy array and then convert those to
# a Tensor.
# Images: Pillow, OpenCV, etc
# Audio: scipy, librosa, etc
# Text: raw Python, Cython, NLTK, SpaCy

# 'torchvision' is specifically for computer vision that has data loaders
# for common datasets such as Imagenet, CIFAR10, MNIST, etc & data
# transformers for images, viz., 'torchvision.datasets' &
# 'torch.utils.data.DataLoader'. All of this provides huge convenience
# and avoids writing boilerplate code

# We will use CIFAR10 which has classes 'airplane', 'automobile', 'bird',
# 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
# Images are 3x3x32

# %%
# Training classifier
# =========================================================================
# - Load and normalize CIFAR10 training and test sets with 'torchvision'
# Define CNN
# Define loss
# Train network on training data
# Test network on test data

# Loading data
# =========================================================================
import torch
import torchvision
import torchvision.transforms as transforms

# output of torchvision datasets are PILImage images of range [0, 1]. We
# transform them to Tensors of normalised range [-1, 1]
transform = transforms.Compose(
  [transforms.ToTensor(),
   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
    download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
    shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
    download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
    shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
           'ship', 'truck')


# %%
# Viewing training images
# =========================================================================
# just for fun, let's view some of the images
import matplotlib.pyplot as plt
import numpy as np

def imsave(img):
  img = img / 2 + 0.5  # unnormalise
  npimg = img.numpy()
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.savefig('./img.png', bbox_inches="tight")


# Random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# Show images
imsave(torchvision.utils.make_grid(images))

# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

# %%
# Define CNN
# =========================================================================
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 16 * 5 * 5)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)

    return x

net = Net()
print(net)

# %%
# Loss function and optimiser
# =========================================================================
# Classification Cross-Entropy and SGD with momentum
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimiser = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# %%
# Training network
# =========================================================================
num_epochs = 2
for epoch in range(num_epochs):
  running_loss = 0.0
  for i, data in enumerate(trainloader, 0):
    # get inputs; data is list of [inputs, labels]
    inputs, labels = data

    # zero param gradients
    optimiser.zero_grad()

    # forward + backward + optimise
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimiser.step()

    # update loss and print stats
    running_loss += loss.item()
    if i % 2000 == 1999:  # print every 2000 mini-batches
      print(f'[{epoch + 1}, {i + 1}] loss: {(running_loss / 2000):.3f}')
      running_loss = 0.0

print("Finished training")

# Save the model
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

# %%
# Testing network
# =========================================================================
# Display image from test set to get familiar
dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imsave(torchvision.utils.make_grid(images))
print(f"GT: {' '.join('%5s' % classes[labels[j]] for j in range(4))}")

# Load in saved model
net = Net()
net.load_state_dict(torch.load(PATH))

# What is the network's prediction?
outputs = net(images)

# Outputs are energies for 10 classes (higher energy = higher confidence).
_, predicted = torch.max(outputs, 1)
print(f"Predicted: {' '.join('%5s' % classes[predicted[j]] for j in range(4))}")

# Performance on entire dataset
correct = 0
total = 0

with torch.no_grad():
  for data in testloader:
    images, labels = data
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print(f"Accuracy of network on 10,000 test images: {100 * correct / total}%")

# Random chance is 10% (since there are 10 classes), so if it's
# significantly higher than this, the network learnt something!
# Which classes performed well, and which didn't?
num_classes = 10
class_correct = list(0. for i in range(num_classes))
class_total = list(0. for i in range(num_classes))

with torch.no_grad():
  for data in testloader:
    images, labels = data
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    c = (predicted == labels).squeeze()

    for i in range(4):
      label = labels[i]
      class_correct[label] += c[i].item()
      class_total[label] += 1

for i in range(num_classes):
  print(f"Accuracy of {classes[i]}: {100 * class_correct[i] / class_total[i]}%")

# Now we can do what we please with this info and see how to improve it...

# %%
# Training on GPU
# =========================================================================
# We can transfer the net onto GPU like we transferred a Tensor before
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)