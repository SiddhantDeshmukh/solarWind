#%%
import numpy as np
import heliopy.data.omni as omni
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import astropy.units as u
from datetime import datetime, timedelta
from analogue_ensemble import time_window_to_time_delta
import rnn
from baseline_metrics import naive_forecast_start, naive_forecast_end, \
  mean_forecast, median_forecast
from loss_functions import mse, rmse
import pandas as pd

from rnn import lstm_model

def get_omni_rtn_data(start_time: datetime, end_time: datetime):
  identifier = 'OMNI_COHO1HR_MERGED_MAG_PLASMA'  # COHO 1HR data
  omni_data = omni._omni(start_time, end_time, identifier=identifier, intervals='yearly', warn_missing_units=False)

  return omni_data


def remove_nans_from_data(data: np.array, 
    model_inputs: np.array, model_outputs: np.array):
  nan_check = np.array([data[i:i + 25] for i in range(len(data) - 25 + 1)])
  model_inputs = model_inputs[np.where([~np.any(np.isnan(i)) for i in nan_check])]
  model_outputs = model_outputs[np.where([~np.any(np.isnan(i)) for i in nan_check])]

  print(f"Input shape: {model_inputs.shape}")
  print(f"Output shape: {model_outputs.shape}")

  print(f"Any NanNs? {np.any(np.isnan(model_inputs)) or np.any(np.isnan(model_outputs))}")

  return model_inputs, model_outputs


def split_into_24_hour_sections(data: np.array):
  model_inputs = np.array([data[i:i+24] for i in range(len(data) - 24)])[:, :, np.newaxis]
  model_outputs = np.array(data[24:])

  # Check for and remove NaNs
  model_inputs, model_outputs = remove_nans_from_data(data, model_inputs, model_outputs)

  return model_inputs, model_outputs


def split_train_val_test(input_data: np.array, output_data: np.array,
  start_timestamp=None):
  # Split based on hard-coded indices present in OMNI data
  # Order is train -> validation -> test
  train_idx_end = 134929
  val_size = 44527
  val_idx_end = train_idx_end + val_size

  inputs_train, outputs_train = input_data[:train_idx_end], output_data[:train_idx_end]
  inputs_val, outputs_val = input_data[train_idx_end: val_idx_end], output_data[train_idx_end: val_idx_end]
  inputs_test, outputs_test = input_data[val_idx_end:], output_data[val_idx_end:]


  # If 'start_timestamp' was given, calculate TimeStamps validation and
  # test start
  if start_timestamp is not None:
    # Assuming hourly cadence (OMNI data)
    validation_timestamp = start_timestamp + time_window_to_time_delta(train_idx_end *u.hr)
    test_timestamp = validation_timestamp + time_window_to_time_delta(val_size * u.hr)

    return inputs_train, inputs_val, inputs_test, outputs_train, outputs_val, outputs_test, validation_timestamp, test_timestamp
  
  else:
    return inputs_train, inputs_val, inputs_test, outputs_train, outputs_val, outputs_test


def lstm_input_to_analogue_input(lstm_data: np.array,
  start_timestamp: pd.Timestamp) -> list:
  # 'lstm_data' is a 3D array with dimensions (batch_size, time_step, 1)
  # (1 is for 'univariate')
  # Create a list with entries of size 2 * 'time_step' (equal to
  # 'training_window' + 'forecast_window')
  analogue_input_data = []
  done_slicing = False
  idx = 0
  forecast_time = start_timestamp
  time_step = lstm_data.shape[1]

  while not done_slicing:
    analogue_data = lstm_data[idx:idx + 2]
    forecast_time += time_window_to_time_delta(time_step * u.hr)

    # data for analogue ensemble is a tuple of (forecast_time, data)
    # where the 'data' includes both training and forecast (split it based
    # on forecast_time time stamp)
    analogue_input_data.append((forecast_time, analogue_data.flatten()))
    idx += 2

    if idx >= len(lstm_data):
      done_slicing = True

  return analogue_input_data


# Try out an LSTM
if __name__ == "__main__":
  # In hours for hourly cadence OMNI RTN data
  START_TIME = (datetime(1995, 1, 1))
  END_TIME = (datetime(2018, 2, 28))
  INPUT_LENGTH = 24 

  data = get_omni_rtn_data(START_TIME, END_TIME).to_dataframe()

  initial_time_stamp = data.index[0]
  
  mag_field_strength, bulk_wind_speed = np.array(data["BR"]), np.array(data["V"])

  # Split into 24-hour sections
  mag_field_input, mag_field_output = split_into_24_hour_sections(mag_field_strength)
  wind_speed_input, wind_speed_output = split_into_24_hour_sections(bulk_wind_speed)

  # Just using B_R data from here on, need to find a better way to use any OMNI dataset
  # Train/test/validation split
  inputs_train, inputs_val, inputs_test,\
    outputs_train, outputs_val, outputs_test,\
    validation_timestamp, test_timestamp = \
    split_train_val_test(mag_field_input, mag_field_output,
    start_timestamp=initial_time_stamp)
  
  # LSTM model
  model = rnn.lstm_model()
  print(model.summary())

  print(initial_time_stamp, validation_timestamp, test_timestamp)
  analogue_input_data = lstm_input_to_analogue_input(inputs_train, initial_time_stamp)

  print(len(analogue_input_data), analogue_input_data[0][1].shape)

  # %%
  # Compile model
  model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

  # Fit model
  model.fit(inputs_train, outputs_train, 
    validation_data=(inputs_val, outputs_val),
    batch_size=2048, epochs=30,
    callbacks=keras.callbacks.EarlyStopping(restore_best_weights=True, patience=30)
    )

  # %%
  # Test set evaluation
  model.evaluate(inputs_test, outputs_test)

  # Baselines
  naive_start_test = naive_forecast_start(inputs_test)
  naive_end_test = naive_forecast_end(inputs_test)
  mean_test = mean_forecast(inputs_test)
  median_test = median_forecast(inputs_test)

  # Check MSE and RMSE of each baseline
  mse_naive_start, rmse_naive_start = mse(naive_start_test, outputs_test), rmse(naive_start_test, outputs_test)
  mse_naive_end, rmse_naive_end = mse(naive_end_test, outputs_test), rmse(naive_end_test, outputs_test)
  mse_mean, rmse_mean = mse(mean_test, outputs_test), rmse(mean_test, outputs_test)
  mse_median, rmse_median = mse(median_test, outputs_test), rmse(median_test, outputs_test)


  def print_metrics(baseline: str, mse_value: float, rmse_value: float) -> None:
    print(f"{baseline}: MSE = {mse_value:.3f} \t RMSE = {rmse_value:.3f}")

  # Compare baseline metrics to test set evaluation
  baselines = ["Naive start", "Naive end", "Mean", "Median"]
  mse_metrics = [mse_naive_start, mse_naive_end, mse_mean, mse_median]
  rmse_metrics = [rmse_naive_start, rmse_naive_end, rmse_mean, rmse_median]

  for baseline, mse_metric, rmse_metric in zip(baselines, mse_metrics, rmse_metrics):
    print_metrics(baseline, mse_metric, rmse_metric)
# %%
