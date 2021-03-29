# Neural network baseline models with optuna optimisation for hyperparams
from typing import Dict
import tensorflow.keras as keras
from datetime import datetime
import loss_functions
import models
from loss_functions import mse, rmse
import numpy as np

import data_processing as dp
import os
import optuna

# Define hyperparameters for optimisation
optimizers = {
    'Adam': keras.optimizers.Adam,
    'RMSProp': keras.optimizers.RMSprop
}

loss_funcs = {
    'mse': keras.losses.MeanSquaredError(),
    'mae': keras.losses.MeanAbsoluteError()
}


def suggest_hyperparameters(trial: optuna.Trial):
  params = {}
  # learning rate on log scale
  # dropout ratio in range [0.0, 0.9], step 0.1
  params['lr'] = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
  params['dropout'] = trial.suggest_float('dropout', 0.0, 0.9, step=0.1)
  params['optimiser_name'] = trial.suggest_categorical(
      'optimiser_name', list(optimizers.keys()))

  params['batch_size'] = trial.suggest_categorical(
      'batch_size', [32, 64, 128, 256])

  params['loss'] = trial.suggest_categorical('loss', loss_funcs)

  # Layers
  encoder_layer_choices = [[4, 8, 16, 32, 64, 128, 256],
                           [8, 16, 32, 64, 128, 256, 512],
                           [16, 32, 64, 128, 256, 512, 1024]]
  decoder_layer_choices = [[4, 8, 16, 32, 64, 128, 256],
                           [8, 16, 32, 64, 128, 256, 512],
                           [16, 32, 64, 128, 256, 512, 1024]]

  params['num_encoder_layers'] = trial.suggest_int(
      'num_encoder_layers', 1, 3)
  params['encoder_layers'] = trial.suggest_categorical(
      'encoder_layers', encoder_layer_choices)

  params['num_decoder_layers'] = trial.suggest_int(
      'num_decoder_layers', 1, 3)
  params['decoder_layers'] = trial.suggest_categorical(
      'decoder_layers', decoder_layer_choices)

  params['use_attention'] = trial.suggest_categorical(
      'use_attention', [True, False])

  # Input/output length
  params['input_length'] = trial.suggest_categorical(
      'input_length', [12, 24, 36, 48])

  return params


def objective(trial: optuna.Trial):
  # Initialise hyperparameters
  params = suggest_hyperparameters(trial)
  print(params)

  # Create and compile model
  # Currently ignoring 'input_length'
  model = models.lstm_model(input_length=24, output_length=24,
                            num_features=4,
                            num_encoder_layers=params['num_encoder_layers'],
                            encoder_neurons=params['encoder_layers'],
                            num_decoder_layers=params['num_decoder_layers'],
                            decoder_neurons=params['decoder_layers'],
                            use_attention=params['use_attention'])

  optimizer = optimizers[params['optimiser_name']](lr=params['lr'])

  # Use all losses not chosen as metrics
  metrics = list(loss_funcs.keys()).remove(params['loss'])
  model.compile(optimizer=optimizer, loss=params['loss'], metrics=metrics)

  print(model.summary())

  # Fit model
  model.fit(data['train_in'], data['train_out'], batch_size=params['batch_size'],
            epochs=30, validation_data=(data['val_in'], data['val_out']),
            callbacks=[keras.callbacks.EarlyStopping(
                restore_best_weights=True, patience=10),
                keras.callbacks.ModelCheckpoint(
                f'./models/lstm-optuna-{trial.number}.h5', save_best_only=True)
  ],
      verbose=2)

  # Should evaluate on test? Then we need to unstandardise the outputs!
  return model.evaluate(data['val_in'], data['val_out'])


# Load data
START_TIME = datetime(1995, 1, 1)
END_TIME = datetime(2020, 12, 31)

INPUT_LENGTH = 24
OUTPUT_LENGTH = 24
NUM_FEATURES = 4

data, keys, mean, std = dp.omni_cycle_preprocess(START_TIME, END_TIME,
                                                 # auto get ["N", "V", "ABS_B", "HMF_INC"]
                                                 get_geoeffectiveness=True,
                                                 standardise=True)

for key in data.keys():
  # Check dimensionality and remove geoeffectiveness from each
  # last index is G by default
  data[key] = data[key][:, :, :keys.index("G")]
  print(f"{key} shape: {data[key].shape}")
  print(f"{key} (Min, Max): ({np.min(data[key])}, {np.max(data[key])})")

# Optuna study
study = optuna.create_study()
study.optimize(objective, n_trials=5)
