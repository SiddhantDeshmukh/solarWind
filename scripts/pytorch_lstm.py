# %%
# Imports
# =========================================================================
import torch
import torch.nn as nn
import data_processing as dp
from datetime import datetime

# %%
# Networks
# =========================================================================
class LSTMNet(nn.Module):
  def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
    super().__init__()
    self.hidden_layer_size = hidden_layer_size
    self.lstm = nn.LSTM(input_size, hidden_layer_size)
    self.linear = nn.Linear(hidden_layer_size, output_size)
    self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                        torch.zeros(1, 1, self.hidden_layer_size))
    
  def forward(self, x: torch.Tensor):
    lstm_out, self.hidden_cell = self.lstm((x.view(len(x)), 1, -1), self.hidden_cell)
    predictions = self.linear(lstm_out.view(len(x), -1))

    return predictions[-1]


# %%
# Data preprocessing
# =========================================================================
START_TIME = (datetime(1995, 1, 1))
END_TIME = (datetime(2018, 2, 28))

# Get data split into training, validation, testing in 24 hour sections
data = dp.omni_preprocess(START_TIME, END_TIME, ['BR'], make_tensors=True)['BR']

print(data.keys())
print(type(data['train_in']))

# %%
# Create model
# =========================================================================
INPUT_LENGTH = 24
model = LSTMNet(INPUT_LENGTH, 20, 1)
print(model)


# %%
# Train model
# =========================================================================

# %%
# Validate model
# =========================================================================
