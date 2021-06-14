# %%
# =========================================================================
# Imports
# =========================================================================
import data_processing as dapr
import pandas as pd
import optuna
import mlflow

import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import StepLR
from datetime import datetime

# %%
# =========================================================================
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


# =========================================================================
# Datetime utility functions
# =========================================================================
def datetime_from_cycle(solar_cycles: pd.DataFrame, cycle: int,
                        key='start_min', fmt='%Y-%m-%d'):
  # From the 'solar_cycles' DataFrame, get the 'key' datetime (formatted as
  # %Y-%m-%d by default in the csv) as a datetime
  return datetime.strptime(solar_cycles.loc[cycle][key], fmt)


# %%
# =========================================================================
# Training and validation functions
# =========================================================================
def train(model, device, train_in, train_out, optimizer, loss_function, epoch):
  # Train 'model' on training data
  assert len(train_in) == len(train_out), \
      f"Error: Training input, output have size mismatch {len(train_in), len(train_out)}"

  model.train()
  print(f"Training epoch {epoch}.")
  num_batches = len(train_in)
  train_loss = 0.0

  for batch_idx, (data, target) in enumerate(zip(train_in, train_out)):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = loss_function(output, target)
    train_loss += loss.item()

    loss.backward()
    optimizer.step()

    # Print loss every 10 batches
    if batch_idx % 10 == 0:
      print(f"Train Epoch: {epoch}, batch {batch_idx} / {num_batches} "
            f"\tLoss: {loss.item():.6f}")

  avg_train_loss = train_loss / num_batches
  return avg_train_loss


def validate(model, device, val_in, val_out, loss_function):
  assert len(val_in) == len(val_out), \
      f"Error: Training input, output have size mismatch {len(val_in), len(val_out)}"

  model.eval()
  print("Validating.")
  val_loss = 0.0

  with torch.no_grad():
    for data, target in zip(val_in, val_out):
      data, target = data.to(device), target.to(device)
      output = model(data)

      # Sum of batch loss
      val_loss += loss_function(output, target).item()

  val_loss /= len(val_in)
  print(f"Validation set: Average loss: {val_loss:.4f}")

  return val_loss


# %%
# =========================================================================
# Optuna parameter suggestions
# =========================================================================
optimisers = {
    "Adam": optim.Adam
}


def suggest_hyperparameters(trial: optuna.Trial):
  # learning rate on log scale
  # dropout ratio in range [0.0, 0.9], step 0.1
  # optimizer is categorical
  lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
  # dropout = trial.suggest_float("dropout", 0.0, 0.9, step=0.1)
  optimiser_name = trial.suggest_categorical(
      "optimiser_name", list(optimisers.keys()))

  return lr, optimiser_name

# =========================================================================
# Optuna + MLFlow run (objective function)
# =========================================================================


def objective(trial: optuna.Trial):
  best_val_loss = float('Inf')
  run_name = "lstm-predict-24"

  # Start new MLFlow run
  with mlflow.start_run(run_name=run_name):
    # Hyparam suggestions from optuna; log with MLFlow
    lr, optimiser_name = suggest_hyperparameters(trial)

    # Shouldn't I apply the parameters to the trial first? Not sure if this
    # is done automatically
    mlflow.log_params(trial.params)

    # Use CUDA if GPU available and log device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlflow.log_param("device", device)

    # Initialise network
    model = LSTMNet().to(device)

    # Pick optimiser (from Optuna suggestions)
    optimiser = optimisers[optimiser_name](model.parameters(), lr=lr)

    # LR scheduler
    scheduler = StepLR(optimiser, step_size=1, gamma=0.7)

    # Load data

    # Define loss
    loss_function = nn.MSELoss()

    # Network training + validation
    num_epochs = 30
    for epoch in range(num_epochs):
      avg_train_loss = train(model, device,
                             data['train_in'], data['train_out'],
                             optimiser, loss_function, epoch)
      avg_val_loss = validate(model, device,
                              data['val_in'], data['val_out'],
                              loss_function)

      if avg_val_loss <= best_val_loss:
        best_val_loss = avg_val_loss

      # Log avg train and val loss for current epoch
      mlflow.log_metric('avg_train_losses', avg_train_loss, step=epoch)
      mlflow.log_metric('avg_val_loss', avg_val_loss, step=epoch)

      scheduler.step()

  return best_val_loss


if __name__ == "__main__":
  # %%
  # =========================================================================
  # Data preprocessing
  # =========================================================================
  solar_cycles_csv = '../res/solar_cycles.csv'
  solar_cycles = pd.read_csv(solar_cycles_csv, index_col=0)

  START_TIME = datetime_from_cycle(solar_cycles, 21)  # start cycle 21
  END_TIME = datetime_from_cycle(
      solar_cycles, 24, key='end')  # end cycle 24

  # Get data split into training, validation, testing in 24 hour sections
  # Change this to have cycle 21 and 22 for training, 23 for val, 24 for test
  print("Loading data...")
  data = dapr.omni_cycle_preprocess(START_TIME, END_TIME, ['BR'],
                                    make_tensors=True, split_mini_batches=True)['BR']

  # %%
  # =========================================================================
  # Create model
  # =========================================================================
  print("Creating model")
  model = LSTMNet(4, 20, 24)
  print(model)

  # %%
  # =========================================================================
  # Set up optimiser and loss
  # =========================================================================
  print("Using MSELoss")
  criterion = nn.MSELoss()
  optimiser = optim.RMSprop(model.parameters(), lr=1e-3)
  metrics = []  # how to use mae loss as a metric?

  print("Optuna study:")
  study = optuna.create_study(
      study_name='lstm-mlflow-optuna', direction='minimize')
  study.optimize(objective, n_trials=5)

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
 # Instead of minimising over the 'test loss', minimise over 'validation loss'
 # Recommendation: Have lots of validation sets!
