# Neural network baseline models with optuna optimisation for hyperparams
# NOTE: This is outdated and the stacked LSTM does not compile
# Since we moved towards 1D CNN as the main model, we are only using a
# simple LSTM as a baseline now

from typing import Dict
import tensorflow.keras as keras
from datetime import datetime
import models
import numpy as np

import data_processing as dapr
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


# Add functionality to use stacked LSTM vs encoder-decoder!
def suggest_hyperparameters(trial: optuna.Trial):
  params = {}
  # learning rate on log scale
  # dropout ratio in range [0.0, 0.9], step 0.1
  params['lr'] = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
  params['dropout'] = trial.suggest_float('dropout', 0.0, 0.9, step=0.1)
  params['optimiser_name'] = trial.suggest_categorical(
      'optimiser_name', list(optimizers.keys()))

  # params['batch_size'] = trial.suggest_categorical(
  #     'batch_size', [32, 64, 128, 256])
  params['batch_size'] = 32

  params['loss'] = trial.suggest_categorical('loss', loss_funcs)

  params['architecture'] = trial.suggest_categorical('architecture',
                                                     ['stacked', 'encoder-decoder'])

  if params['architecture'] == 'stacked':  # stacked LSTM architecture
    params['num_layers'] = trial.suggest_int('num_layers', 1, 3)
    layer_choices = [
        [4, 8, 16],
        [8, 16, 32],
        [16, 32, 64]
    ]
    params['layers'] = trial.suggest_categorical('layers', layer_choices)

    # Add Dense layers!

  else:  # encoder-decoder architecture
    # Layers - symmetric encoder/decoder architecture
    # Encoder layers decrease!
    encoder_layer_choices = [[16, 8, 4],
                             [32, 16, 8],
                             [64, 32, 16]]

    params['num_encoder_layers'] = trial.suggest_int(
        'num_encoder_layers', 1, 3)
    params['encoder_layers'] = trial.suggest_categorical(
        'encoder_layers', encoder_layer_choices)[:params['num_encoder_layers']]

    params['num_decoder_layers'] = params['num_encoder_layers']
    params['decoder_layers'] = params['encoder_layers'][::-1]

  # Attention vs repeat vector
  params['use_attention'] = trial.suggest_categorical(
      'use_attention', [True, False])

  # Input/output length
  # params['input_length'] = trial.suggest_categorical(
  #     'input_length', [12, 24, 36, 48])
  params['input_length'] = 24

  return params


def objective(trial: optuna.Trial):
  # Initialise hyperparameters
  params = suggest_hyperparameters(trial)
  print(params)

  model_choices = {
      'stacked': models.stacked_lstm,
      'encoder-decoder': models.encoder_decoder_lstm
  }

  # Create and compile model
  # Currently ignoring 'input_length' (see suggest_hyperparameters())
  non_model_param_keys = ['lr', 'loss', 'dropout', 'architecture',
                          'optimiser_name', 'batch_size']
  model_params = {key: val for key,
                  val in params.items() if not key in non_model_param_keys}
  model_params['num_features'] = 4

  model = model_choices[params['architecture']](**model_params)
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


# Load 1 year of data for quick testing
START_TIME = datetime(2019, 1, 1)
END_TIME = datetime(2019, 12, 31)

# Use only a year of data when using optuna
INPUT_LENGTH = 24
OUTPUT_LENGTH = 24
NUM_FEATURES = 4

data, keys, mean, std = dapr.omni_cycle_preprocess(START_TIME, END_TIME,
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
