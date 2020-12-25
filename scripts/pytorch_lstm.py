# Let's try out PyTorch! Eventually, would also like to incorporate MLFlow
# and optuna
# Imports
# =========================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

import numpy as np


# Classes - Networks
# =========================================================================
# Set up a framework for the network
class LSTMNetwork(nn.Module):
  def __init__(self, dropout=0.0):
    super(LSTMNetwork, self).__init__()

    # Layers
    self.lstm1 = nn.LSTM(1, 10)  # LSTM input
    self.linear1 = nn.Linear(10, 5)  # intermediate
    self.linear2 = nn.Linear(5, 1)  # output

  def forward(self, x: np.ndarray):
    # 'x' must have shape (sequence_length, batch_size, input_size)
    (all_outs, (final_output, final_state)) = self.lstm1(x)

    output = self.linear1(all_outs[-1])
    output = self.linear2(output)

    return output
