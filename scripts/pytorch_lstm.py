# %%
# Imports
# =========================================================================
from tensorflow.python.keras.engine import training
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

print(type(data), type(data['train_in']))

# %%
# Create model
# =========================================================================
INPUT_LENGTH = 24
model = LSTMNet(INPUT_LENGTH, 20, 1)
print(model)

# %%
# Set up optimiser and loss
# =========================================================================
import torch.optim as optim

criterion = nn.MSELoss()
optimiser = optim.RMSprop(model.parameters(), lr=0.001)
metrics = []  # how to use mae loss as a metric?

for key in data.keys():
  if isinstance(data[key], torch.Tensor):
    print(key, data[key].size())

# %%
# Train model
# =========================================================================
num_epochs = 2
for epoch in range(num_epochs):
  running_loss = 0.0
  val_loss = 0.0

  # Remove loop and pass whole dataset in? Or maybe I have to split into
  # mini-batches?
  for i, (train_in, val_in, train_out, val_out) in\
      enumerate(zip(data['train_in'], data['val_in'],
                    data['train_out'], data['val_out'])):
    # Zero param gradients
    optimiser.zero_grad()

    # forward = backward + optimise (training)
    print(train_in.size())
    print(val_in.size())
    print(train_out.size())
    print(val_out.size())
    outputs = model(train_in)
    loss = criterion(outputs, train_out)
    loss.backward()
    optimiser.step()

    # forward + loss (validation)
    with torch.no_grad():
      model_val_outputs = model(val_in)
      val_loss += criterion(model_val_outputs, val_out).item()

    # update loss and print stats
    running_loss += loss.item()
    if i % 2000 == 1999:  # print every 2000 mini-batches
      print(f"[{epoch + 1}, {i + 1}] - ")
      print(f"Loss (MSE): {(running_loss / 2000):.3f}")
      print(f"Val. Loss (MSE): {(val_loss / 2000):.3f}")

      running_loss = 0.0
      val_loss = 0.0
    
print("Finished training")

# Save model
MODEL_PATH = '../models/torch_lstm.pth'
torch.save(model.state_dict(), MODEL_PATH)

# %%
# Validate model
# =========================================================================
