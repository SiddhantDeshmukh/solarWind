# %%
# Imports
from baseline.baseline_metrics import last_n_steps, mean_n_steps, median_n_steps
from loss_functions import mse, mae
import tensorflow.keras as keras
from models import cnn_1d_model
from datetime import datetime
import data_processing as dapr
import numpy as np

# %%
# Load 1 year of data for quick testing
START_TIME = datetime(1995, 1, 1)
END_TIME = datetime(2019, 12, 31)

# Use only a year of data when using optuna
INPUT_LENGTH = 24
OUTPUT_LENGTH = 24
NUM_FEATURES = 4

data, keys, mean, std = dapr.omni_cycle_preprocess(START_TIME, END_TIME,
                                                   get_geoeffectiveness=True,
                                                   standardise=True)

for key in data.keys():
  # Check dimensionality and remove geoeffectiveness from each
  # last index is G by default
  data[key] = data[key][:, :, :keys.index("G")]
  print(f"{key} shape: {data[key].shape}")
  print(
      f"{key} (Min, Max): ({np.min(data[key], axis=(0, 1))}, {np.max(data[key], axis=(0, 1))})")
  print(f"{key} (Mean, sigma): ({mean}, {std})")

# Remove 'G' mean and std
mean = mean[:-1]
std = std[:-1]

# %%
# Let's fit the 1D CNN

model = cnn_1d_model(INPUT_LENGTH, NUM_FEATURES, OUTPUT_LENGTH)
model.compile(optimizer="adam", loss="mae", metrics=["mse"])
print(model.summary())

model.fit(data['train_in'], data['train_out'], batch_size=32, epochs=30,
          validation_data=(data['val_in'], data['val_out']),
          callbacks=[
    keras.callbacks.EarlyStopping(restore_best_weights=True,
                                  patience=10),
    keras.callbacks.ModelCheckpoint(f"./models/cnn-1d.h5",
                                    save_best_only=True)
])

# Validation
model.evaluate(data['val_in'], data['val_out'])
# %%
# Calculate baseline metric losses
baseline_dict = {
    'last_n': last_n_steps,
    'mean_n': mean_n_steps,
    'median_n': median_n_steps
}
# Add max and min!

baseline_evals = {}
mse_losses = {}
mae_losses = {}


for key, baseline in baseline_dict.items():
  baseline_evals[key] = dapr.unstandardise_array(
      baseline(data['val_in'], n=24, repeated=True), mean, std)
  mse_losses[key] = mse(baseline_evals[key], data['val_out'])
  mae_losses[key] = mae(baseline_evals[key], data['val_out'])

  print(f"{key}: MSE loss = {mse_losses[key]:.4f}")
  print(f"{key}: MAE loss = {mae_losses[key]:.4f}")

# print(data['val_in'][-1])
# print(baseline_evals['last_n'][-1])

# %%
