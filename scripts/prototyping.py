# %%
# Quick test to see what data are availble in OMNI and to
# compute geoeffectiveness
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import TensorDataset
import torch.optim as optim
import torch.nn as nn
import data_processing as dp
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch


def calculate_geoeffectiveness(wind_density: np.ndarray,
                               hmf_intensity: np.ndarray,
                               wind_speed: np.ndarray,
                               hmf_clock_angle: np.ndarray) -> np.ndarray:
  alpha = 0.5  # empirically determined
  geoffectiveness = wind_density**(2/3 - alpha) * \
      hmf_intensity**(2*alpha) * \
      wind_speed**(7/3 - 2 * alpha) * \
      np.sin(hmf_clock_angle / 2)**4

  return geoffectiveness


START_TIME = datetime(1995, 1, 1)
END_TIME = datetime(2020, 12, 31)

# %%
data, keys = dp.omni_cycle_preprocess(START_TIME, END_TIME,
                                      # auto get ["N", "V", "ABS_B", "HMF_INC"]
                                      get_geoeffectiveness=True,
                                      make_tensors=True, normalise_tensors=True,
                                      normalisation_limits=(-1, 1))

print(data.keys(), keys)
for key in data.keys():
  # Check dimensionality and remove geoeffectiveness from each
  # last index is G by default
  data[key] = data[key][:, :, :keys.index("G")]
  print(f"{key} shape: {data[key].shape}")

# %%

# # %%
# # Plot to make sure the normalisation works, etc
# fig, axes = plt.subplots(1, 3)  # cols: train, val, test

# for i, key in enumerate(train_keys):
#   # Training
#   axes[0].plot(combined_data['train_in'][0, :, i], label=key)

# # Validation
#   axes[1].plot(combined_data['val_in'][0, :, i], label=key)

# # Testing
#   axes[2].plot(combined_data['test_in'][0, :, i], label=key)

#   axes[2].legend()

# plt.show()
# %%
# Create custom TensorDatasets and Loaders for train,val,test
# TensorDataset from combined data
print("Creating Tensor Datasets")
train_set = TensorDataset(
    data['train_in'], data['train_out'])
val_set = TensorDataset(
    data['val_in'], data['val_out'])
test_set = TensorDataset(
    data['test_in'], data['test_out'])

# DataLoaders
batch_size = 32
train_loader = DataLoader(train_set, batch_size=batch_size)
val_loader = DataLoader(val_set, batch_size=batch_size)
test_loader = DataLoader(test_set, batch_size=batch_size)

# %%


class MV_LSTM(nn.Module):
  def __init__(self, n_features, seq_length):
    super(MV_LSTM, self).__init__()
    self.n_features = n_features
    self.seq_length = seq_length
    self.n_hidden = 20  # number of hidden states (cells)
    self.n_layers = 1  # number of stacked LSTM layers

    self.lstm = nn.LSTM(input_size=self.n_features,
                        hidden_size=self.n_hidden,
                        num_layers=self.n_layers,
                        batch_first=True)
    # Output of LSTM is (batch_size, seq_len, num_directions * hidden_size)
    self.linear = nn.Linear(
        self.n_hidden * self.seq_length, 1)  # pass in output size

  def init_hidden(self, batch_size):
    # Initialise hidden state of LSTM layer
    hidden_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
    cell_state = torch.zeros(self.n_layers, batch_size, self.n_hidden)
    self.hidden = (hidden_state, cell_state)

  def forward(self, x):
    batch_size, seq_length, _ = x.size()

    lstm_out, self.hidden = self.lstm(x, self.hidden)
    y = lstm_out.contiguous().view(batch_size, -1)

    return self.linear(y)


model = MV_LSTM(4, 24)
print(model)

loss_func = nn.MSELoss()
optimiser = optim.Adam(model.parameters(), lr=1e-3)
metrics = []


def train(model, device, train_loader, optimiser, loss_func, epoch):
  # Train 'model' and return average training loss
  model.train()
  train_set_size = len(train_loader.dataset)
  num_batches = len(train_loader)
  train_loss = 0.0

  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    batch_size = len(data)
    optimiser.zero_grad()
    model.init_hidden(batch_size)
    output = model(data)
    print(output.shape, target.shape)
    loss = loss_func(output, target)
    train_loss += loss.item()

    loss.backward()
    optimiser.step()

    if batch_idx % (num_batches // 10) == 0:
      # print(f"Train Epoch: {epoch}, batch {batch_idx+1} / {num_batches} "
      #       f"\tLoss: {loss.item():.6f}")
      print(f"Train Epoch: {epoch+1} [{batch_idx * batch_size}/{train_set_size} "
            f"({100. * batch_idx / num_batches:.0f}%)]\tLoss: {loss.item():.2e}")

  avg_train_loss = train_loss / num_batches

  return avg_train_loss


def validate(model, device, val_loader, optimiser, loss_func):
  # Validate 'model' and return average validation loss
  model.eval()
  num_batches = len(val_loader)
  val_loss = 0.0

  for batch_idx, (data, target) in enumerate(val_loader):
    data, target = data.to(device), target.to(device)
    batch_size = len(data)
    optimiser.zero_grad()
    model.init_hidden(batch_size)
    output = model(data)
    loss = loss_func(output, target)
    val_loss += loss.item()

    loss.backward()
    optimiser.step()

  avg_val_loss = val_loss / num_batches
  print(f"Average validation loss: {avg_val_loss:.2e}")

  return avg_val_loss


num_epochs = 1
for epoch in range(num_epochs):
  avg_train_loss = train(model, 'cpu', train_loader,
                         optimiser, loss_func, epoch)

  avg_val_loss = validate(model, "cpu", val_loader, optimiser, loss_func)

  print(f"Epoch {epoch+1}: Avg train loss = {avg_train_loss:.3E}")
  print(f"Epoch {epoch+1}: Avg validation loss = {avg_val_loss:.3E}")

# Test
avg_test_loss = validate(model, "cpu", test_loader, optimiser, loss_func)
print(f"Epoch {epoch+1}: Avg test loss = {avg_test_loss:.3E}")

# Keep track of losses as a list, save that
# Save model as a checkpoint
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimiser.state_dict(),
    'last_train_loss': avg_train_loss,
    'last_val_loss': avg_val_loss,
    'last_test_loss': avg_test_loss
}

# torch.save(checkpoint, './test.pt')

# # %%
# # Load model (test)
# model_load = torch.load('./test.pt')

# %%
# # Quick plotting
# data = dp.get_omni_rtn_data(START_TIME, END_TIME).to_dataframe()
# clock_angle = np.arctan2(-data['BT'], data['BN'])
# data['HMF_theta'] = clock_angle
# data['g'] = calculate_geoeffectiveness(
#     data['N'].values, data['ABS_B'].values, data['V'].values, clock_angle)

# # Plot HMF clock angle on first axis and geoeffectiveness on second
# fig, axes = plt.subplots(1, 2, figsize=(8, 8))

# axes[0].hist(data.index, data['HMF_theta'])
# axes[0].set_xlabel("HMF Clock Angle (rad)")
# axes[0].set_ylabel("Frequency")

# axes[1].plot(data.index, data['g'])
# axes[1].set_xlabel("Time")
# axes[1].set_ylabel("Geoeffectiveness")

# plt.savefig('./g.png', bbox_inches="tight")
