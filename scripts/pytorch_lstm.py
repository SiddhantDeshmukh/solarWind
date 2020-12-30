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

    return output

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
data = dp.omni_preprocess(START_TIME, END_TIME, ['BR'],
    make_tensors=True, split_mini_batches=True)['BR']

print(type(data), type(data['train_in']))

# %%
# Create model
# =========================================================================
model = LSTMNet(1, 20, 1)
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
 
  # Loop over mini-batches
  for i, (train_in, val_in, train_out, val_out) in\
      enumerate(zip(data['train_in'], data['val_in'],
                    data['train_out'], data['val_out'])):
    # Zero param gradients
    optimiser.zero_grad()

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
    print(f"[{epoch + 1}, {i + 1}] - ")
    print(f"Loss (MSE): {running_loss:.3f}")
    print(f"Val. Loss (MSE): {val_loss:3f}")

print("Finished training")

# Save model
MODEL_PATH = '../models/torch_lstm.pth'
torch.save(model.state_dict(), MODEL_PATH)

# %%
# Validate model
# =========================================================================
