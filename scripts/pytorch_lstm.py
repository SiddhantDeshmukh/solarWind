# %%
# Imports
# =========================================================================
import torch
import torch.nn as nn
import data_processing as dp
from datetime import datetime
import pandas as pd

# %%
# Networks
# =========================================================================
class LSTMNet(nn.Module):
  def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
    super().__init__()
    self.hidden_layer_size = hidden_layer_size
    self.lstm = nn.LSTM(input_size, hidden_layer_size)
    self.linear = nn.Linear(hidden_layer_size, output_size)
    
  def forward(self, x: torch.Tensor):
    # 'x' has size [seq_len, batch_size, n_feat]
    (all_outs, (final_output, final_state)) = self.lstm(x)
    output = self.linear(all_outs[-1])

    return output.view(-1)


class LSTM(nn.Module):
  def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1, num_layers=2):
    super(LSTM, self).__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.batch_size = batch_size
    self.num_layers = num_layers

    # Define LSTM layer
    self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

    # Define output layer
    self.linear = nn.Linear(self.hidden_dim, output_dim)

  def init_hidden(self):
    # Initialise hidden state
    return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
            torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))
  
  def forward(self, x):
    # Forward pass through LSTM layer
    # lstm_out: [input_size, batch_size, hidden_dim]
    # self.hidden: (a, b) where both a, b, have shape [num_layers, batch_size, hidden_dim]
    lstm_out, self.hidden = self.lstm(x.view(len(x), self.batch_size, -1))
    y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))

    return y_pred.view(-1)

# Datetime utility functions
# =========================================================================
def datetime_from_cycle(solar_cycles: pd.DataFrame, cycle: int,
                        key='start_min', fmt='%Y-%m-%d'):
  # From the 'solar_cycles' DataFrame, get the 'key' datetime (formatted as
  # %Y-%m-%d by default in the csv) as a datetime
  return datetime.strptime(solar_cycles.loc[cycle][key], fmt)


# =========================================================================
solar_cycles_csv = '../res/solar_cycles.csv'
solar_cycles = pd.read_csv(solar_cycles_csv, index_col=0)

# %%
# Data preprocessing
# =========================================================================
START_TIME = datetime_from_cycle(solar_cycles, 21)  # start cycle 21
END_TIME = datetime_from_cycle(solar_cycles, 24, key='end')  # end cycle 24

# Get data split into training, validation, testing in 24 hour sections
# Change this to have cycle 21 and 22 for training, 23 for val, 24 for test
print("Loading data...")
data = dp.omni_preprocess(START_TIME, END_TIME, ['BR'],
    make_tensors=True, split_mini_batches=True)['BR']

# %%
# Create model
# =========================================================================
print("Creating model")
model = LSTMNet(1, 20, 1)
# model = LSTM(1, 20, 2048)

print(model)

# %%
# Set up optimiser and loss
# =========================================================================
import torch.optim as optim

criterion = nn.MSELoss()
optimiser = optim.RMSprop(model.parameters(), lr=1e-3)
metrics = []  # how to use mae loss as a metric?

# %%
# Training and validation functions
# =========================================================================  
print("Training start")
num_epochs = 30
for epoch in range(num_epochs):
  running_loss = 0.0
  val_loss = 0.0
  
  # Train has 66 batches, val has 22 batches - redo loop!
  print(len(data['train_in']))
  print(len(data['train_out']))
  print(len(data['val_in']))
  print(len(data['val_out']))
 
  # Loop over mini-batches
  for i, (train_in, val_in, train_out, val_out) in\
      enumerate(zip(data['train_in'], data['val_in'],
                    data['train_out'], data['val_out'])):
    # Zero param gradients
    optimiser.zero_grad()

    # print(train_in.size())
    # print(train_out.size())

    # print(val_in.size())
    # print(val_out.size())

    # forward = backward + optimise (training)
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

    if i % 10 == 0:
      print(f"[Epoch {epoch + 1}; {i + 1}] - ")
      print(f"Loss (MSE): {running_loss:.3f}")
      print(f"Val. Loss (MSE): {val_loss:3f}")

print("Finished training")

# Save model
MODEL_PATH = '../models/torch_lstm.pth'
torch.save(model.state_dict(), MODEL_PATH)

# %%
# Validate model
# =========================================================================
