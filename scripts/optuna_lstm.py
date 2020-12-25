from datetime import datetime
import tensorflow as tf
import tensorflow.keras as keras
import optuna
from typing import List
import numpy as np

from models import create_lstm_model
import data_processing as dp

def objective(trial):
  lstm_layers = trial.suggest_int("lstm_layers", 1, 3)
  layer_choices = [(4, 8, 16, 32, 64, 128, 256),
                   (8, 16, 32, 64, 128, 256, 512),
                   (16, 32, 64, 128, 256, 512, 1024)]
  lstm_layer_choice = trial.suggest_categorical("lstm_layer_choice", layer_choices)
  lstm_layer_choice = lstm_layer_choice[:lstm_layers]

  dense_layers = trial.suggest_int("dense_layers", 1, 2)
  layer_choices = [(1024, 512, 256, 1),
                   (512, 256, 128, 1),
                   (128, 64, 32, 1)]
  dense_layer_choice = trial.suggest_categorical("dense_layer_choice", layer_choices)
  dense_layer_choice = dense_layer_choice[4 - dense_layers:]

  lr = trial.suggest_categorical("lr", (1e-5, 1e-4, 1e-3, 2e-1, 1e-2))
  batch_size = trial.suggest_categorical("batch_size", (32, 64, 128, 256))

  model = create_lstm_model(lstm_layers, lstm_layer_choice, dense_layers, dense_layer_choice)
  
  optimizer = keras.optimizers.Adam(lr=lr)
  model.compile(optimizer=optimizer, loss="mse")
  model.fit(inputs_train, outputs_train,
            validation_data=(inputs_val, outputs_val),
            batch_size=batch_size, epochs=500,
            callbacks=[keras.callbacks.EarlyStopping(restore_best_weights=True, patience=10),
                       keras.callbacks.ModelCheckpoint(f"../models/optuna_{trial.number}.h5", save_best_only=True)],
            verbose=2)

  return model.evaluate(inputs_val, outputs_val)[0]


if __name__ == "__main__":
  # Load in data
  START_TIME = datetime(1995, 1, 1)
  END_TIME = datetime(2018, 2, 28)
  INPUT_LENGTH = 24

  data = dp.get_omni_rtn_data(START_TIME, END_TIME).to_dataframe()
  initial_time_stamp = data.index[0]

  mag_field_strength, bulk_wind_speed = np.array(data["BR"]), np.array(data["V"])

  # Split into 24-hour sections
  mag_field_input, mag_field_output = dp.split_into_24_hour_sections(mag_field_strength)

  # Just using B_R data from here on, need to find a better way to use any OMNI dataset
  # Train/test/validation split
  inputs_train, inputs_val, inputs_test,\
    outputs_train, outputs_val, outputs_test,\
    validation_timestamp, test_timestamp, end_timestamp = \
    dp.split_train_val_test(mag_field_input, mag_field_output,
    start_timestamp=initial_time_stamp)

  # Set up Optuna study
  study = optuna.create_study()
  study.optimize(objective, n_trials=100)
  
