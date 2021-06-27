# %%
# SVM regressor for OMNI data
from typing import Dict
import data_processing as dapr
from datetime import datetime
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import optuna


# Define objective function for optuna
def objective(trial: optuna.Trial):
  # uses higher script scoppe to get data variables
  kernel = trial.suggest_categorical(
      'kernel', ['linear', 'poly', 'sigmoid', 'rbf'])
  C = trial.suggest_loguniform('C', 1e-3, 1e1)
  gamma = trial.suggest_loguniform('gamma', 1e-3, 1e1)

  svr = svm.SVR(C=C, gamma=gamma, kernel=kernel)
  svr.fit(X_train, y_train)

  return svr.score(X_val, y_val)


# Load data
# START_TIME = datetime(1995, 1, 1)
START_TIME = datetime(2019, 1, 1)
END_TIME = datetime(2019, 12, 31)

# Use only a year of data when using optuna
INPUT_LENGTH = 24
OUTPUT_LENGTH = 24
NUM_FEATURES = 4

data, keys, mean, std = dapr.omni_cycle_preprocess(START_TIME, END_TIME,
                                                   get_geoeffectiveness=True,
                                                   standardise=True)

for key in data.keys():
  # Check dimensionality
  print(f"{key} shape: {data[key].shape}")
  print(f"{key}(Min, Max): ({np.min(data[key], axis=(0, 1))},\
                            {np.max(data[key], axis=(0, 1))})")
  print(f"{key} (Mean, sigma): ({mean}, {std})")


# %%

# Create a study for each feature and perform 50 trials for each
n_trials: int = 150
best_trials: Dict[str, optuna.Trial] = {}
for i, key in enumerate(keys):
  # Slice to be (num_samples, num_features) (2D from 3D)
  # and pick index of feature
  X_train = data['train_in'][:, 0, i].reshape(-1, 1)
  y_train = data['train_out'][:, 0, i]
  X_val = data['val_in'][:, 0, i].reshape(-1, 1)
  y_val = data['val_out'][:, 0, i]
  X_test = data['test_in'][:, 0, i].reshape(-1, 1)
  y_test = data['test_out'][:, 0, i]

  # Create optuna trial
  study_name = f"svr-optuna-{key}"
  study = optuna.create_study(study_name=study_name, direction='maximize')
  study.optimize(objective, n_trials=n_trials)

  # Print Optuna study statistics
  print("\n++++++++++++++++++++++++++++++++++\n")
  print("Study statistics: ")
  print("  Number of finished trials: ", len(study.trials))

  print("Best trial:")
  trial = study.best_trial

  print(f" Trial number: {trial.number}")
  print(f" Loss (trial value): {trial.value}")

  print(" Params:")
  for key, value in trial.params.items():
    print(f"{key}: {value}")

  best_trials[key] = trial

for key, trial in best_trials.items():
  print(f"{key}: Best trial score = {trial}")

# Map G-> G?
# Ensemble learning: train SVM on each parameter separately and then
# combine predictions at the end to classify a storm

# # Plot
# fig, axes = plt.subplots(2, 2)
# for i, key in enumerate(keys):
#   if i >= 4:  # key 5 is Geoeffectiveness, not predicting this atm
#     break

#   x = i // 2
#   y = i % 2

#   # Data
#   axes[x, y].plot(y_val[:24, i], '.', label="Data")

#   # Prediction
#   axes[x, y].plot(prediction[:, i], '-', label="Prediction")

#   # Aesthetics
#   axes[x, y].legend()
#   axes[x, y].set_title(key)

# print(score)
# plt.show()

# %%
