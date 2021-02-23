# %%
# Quick test to see what data are availble in OMNI and to
# compute geoeffectiveness
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

data = dp.omni_preprocess(START_TIME, END_TIME, get_geoeffectiveness=True,
                          make_tensors=True, split_mini_batches=False)

# density, wind speed, HMF intensity, HMF clock angle, geoeffectiveness
train_keys = ["N", "V", "ABS_B", "HMF_INC"]
predict_keys = ["G"]

# Want to predict N, V, ABS_B, HMF_INC, for testing, turn into
# geoeffectiveness and compare to actual values
train_in = torch.cat(([data[key]["train_in"] for key in train_keys]), 2)
train_out = torch.cat(
    ([data[key]["train_out"].unsqueeze(1) for key in train_keys]), 1)
val_in = torch.cat(([data[key]["val_in"] for key in train_keys]), 2)
val_out = torch.cat(([data[key]["val_out"].unsqueeze(1)
                      for key in train_keys]), 1)

print(f"Train in size: {train_in.size()}")
print(f"Train out size: {train_out.size()}")
print(f"Val in size: {val_in.size()}")
print(f"Val out size: {val_out.size()}")

# Split into mini-batches
data = {
    "train_in": train_in,
    "train_out": train_out,
    "val_in": val_in,
    "val_out": val_out,
}

data = dp.split_data_mini_batches(
    data, 32, input_batch_dim=0, output_batch_dim=0,
    input_size=3, output_size=2)

# Just BR for testing!
# data = dp.omni_preprocess(
#     START_TIME, END_TIME, make_tensors=True, split_mini_batches=True)['BR']

# %%


class MV_LSTM(nn.Module):
  def __init__(self, n_features, seq_length):
    super(MV_LSTM, self).__init__()
    self.n_features = n_features
    self.seq_length = seq_length
    self.n_hidden = 32  # number of hidden states (cells)
    self.n_layers = 1  # number of stacked LSTM layers

    self.lstm = nn.LSTM(input_size=self.n_features,
                        hidden_size=self.n_hidden,
                        num_layers=self.n_layers,
                        batch_first=True)
    # Output of LSTM is (batch_size, seq_len, num_directions * hidden_size)
    self.linear = nn.Linear(
        self.n_hidden * self.seq_length, 1)

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


def train(model, device, train_in, train_out, optimiser, loss_func, epoch):
  # Train 'model' on 'train_in'
  model.train()
  batch_size, num_batches = train_in[0].size()[0], len(train_in)
  train_loss = 0.0

  for batch_idx, (data, target) in enumerate(zip(train_in, train_out)):
    data, target = data.to(device), target.to(device)
    optimiser.zero_grad()
    model.init_hidden(batch_size)
    output = model(data)
    loss = loss_func(output, target)
    train_loss += loss.item()

    loss.backward()
    optimiser.step()

    if batch_idx % 100 == 0:
      print(f"Train Epoch: {epoch}, batch {batch_idx+1} / {num_batches} "
            f"\tLoss: {loss.item():.6f}")

  avg_train_loss = train_loss / num_batches

  return avg_train_loss


num_epochs = 5
for epoch in range(num_epochs):
  avg_train_loss = train(model, 'cpu', data["train_in"],
                         data["train_out"], optimiser, loss_func, epoch)

  print(f"Epoch {epoch}: Avg train loss = {avg_train_loss:.3E}")

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
