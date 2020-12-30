# Functions to load and process data, including support for timestamps.
# Used for setting up LSTM and baseline metrics

# =========================================================================
# Imports
# =========================================================================
from typing import Dict
import numpy as np
import heliopy.data.omni as omni
import astropy.units as u
from astropy.units import Quantity
from datetime import datetime

import torch
import pandas as pd


# =========================================================================
# Data loading
# =========================================================================
def get_omni_rtn_data(start_time: datetime, end_time: datetime):
  identifier = 'OMNI_COHO1HR_MERGED_MAG_PLASMA'  # COHO 1HR data
  omni_data = omni._omni(start_time, end_time, identifier=identifier, intervals='yearly', warn_missing_units=False)

  return omni_data

# =========================================================================
# Data cleaning
# =========================================================================
def remove_nans_from_data(data: np.ndarray, 
    model_inputs: np.ndarray, model_outputs: np.ndarray):
  nan_check = np.array([data[i:i + 25] for i in range(len(data) - 25 + 1)])
  model_inputs = model_inputs[np.where([~np.any(np.isnan(i)) for i in nan_check])]
  model_outputs = model_outputs[np.where([~np.any(np.isnan(i)) for i in nan_check])]

  print(f"Input shape: {model_inputs.shape}")
  print(f"Output shape: {model_outputs.shape}")

  print(f"Any NanNs? {np.any(np.isnan(model_inputs)) or np.any(np.isnan(model_outputs))}")

  return model_inputs, model_outputs

# =========================================================================
# Data preprocessing (mainly splitting)
# =========================================================================
def split_into_24_hour_sections(data: np.ndarray):
  model_inputs = np.array([data[i:i+24] for i in range(len(data) - 24)])[:, :, np.newaxis]
  model_outputs = np.array(data[24:])

  # Check for and remove NaNs
  model_inputs, model_outputs = remove_nans_from_data(data, model_inputs, model_outputs)

  return model_inputs, model_outputs


def split_train_val_test(input_data: np.ndarray, output_data: np.ndarray,
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
    end_timestamp = test_timestamp + time_window_to_time_delta(len(input_data[val_idx_end:]) * u.hr)

    return inputs_train, inputs_val, inputs_test, outputs_train, outputs_val, outputs_test, validation_timestamp, test_timestamp, end_timestamp
  
  else:
    return inputs_train, inputs_val, inputs_test, outputs_train, outputs_val, outputs_test


def omni_preprocess(start_time: datetime, end_time: datetime, keys=["BR"],
    make_tensors=False, split_mini_batches=False):
  # Wrapper function around 'get_omni_rtn_data()' and 'split...()'
  data = get_omni_rtn_data(start_time, end_time).to_dataframe()
  initial_timestamp = data.index[0]
  output_data = {}

  for key in keys:
    array = np.array(data[key])
    arr_input, arr_output = split_into_24_hour_sections(array)
    train_in, val_in, test_in, train_out, val_out, test_out,\
      val_timestamp, test_timestamp, end_timestamp =\
        split_train_val_test(arr_input, arr_output,
                             start_timestamp=initial_timestamp)

    arr_dict = {
      "train_in": train_in, 
      "val_in": val_in,
      "test_in": test_in,
      "train_out": train_out,
      "val_out": val_out,
      "test_out": test_out,
      "val_timestamp": val_timestamp,
      "test_timestamp": test_timestamp,
      "end_timestamp": end_timestamp
    }

    if make_tensors:  # NumPy ndarray -> PyTorch Tensor
      # For LSTM, Torch expects [seq_len, batch_size, num_feat]
      for arr_key in arr_dict.keys():
        if isinstance(arr_dict[arr_key], np.ndarray):
          tensor = torch.from_numpy(arr_dict[arr_key])
          if len(arr_dict[arr_key].shape) == 3:
            tensor = tensor.transpose(0, 1)
          arr_dict[arr_key] = tensor

    if split_mini_batches:
      batch_dim = 1 if make_tensors else 0

      # Write a function to find the number of mini batches based on the
      # ideal mini batch size
      arr_dict = split_data_mini_batches(arr_dict, 4000, input_batch_dim=batch_dim)

    output_data[key] = arr_dict

  return output_data


def split_data_mini_batches(data: Dict, num_mini_batches: int,
  input_batch_dim=0, output_batch_dim=0):
  # For each 3D Tensor, split into mini-batches
  for key in data.keys():
    if isinstance(data[key], torch.Tensor):
      if len(data[key].shape) == 3:  # input data
        mini_batches = torch.split(data[key], num_mini_batches, dim=input_batch_dim)
      
      elif len(data[key].shape) == 1:  # output data
        mini_batches = torch.split(data[key], num_mini_batches, output_batch_dim)
      
      else:
        print(f"Warning: shape of '{key}' is not size 3 (input) or 2 (output).")
        print("Ignoring mini-batches.")
        mini_batches = data[key]
    
      data[key] = mini_batches
  return data

# =========================================================================
# Analogue ensemble conversion (from LSTM)
# =========================================================================
def lstm_3d_to_analogue_input(lstm_data: np.ndarray,
    start_timestamp: pd.Timestamp) -> list:
  # 'lstm_data' is a 3D array with dimensions (batch_size, time_step, 1)
  # (1 is for 'univariate')
  # Create a dict of length 'batch_size' containing key-value pairs of
  # 'timestamp': 'value' by flattening data and adding timestamps
  analogue_input_data = {}
  
  # First element of each sequence to uniquely flatten data into single
  # sequence
  flat_data = lstm_data[:, 0, 0]  

  timestamps = []
  timestamp = start_timestamp

  for i in range(len(flat_data)):
    if i > 0:
      timestamp += pd.Timedelta(1, 'hr')

    timestamps.append(timestamp)

  analogue_input_data = {timestamp: value for (timestamp, value) in zip(timestamps, flat_data)}
  assert len(list(analogue_input_data.keys())) == len(flat_data) 
  
  return analogue_input_data


def lstm_2d_to_analogue_input(lstm_data: np.ndarray,
    start_timestamp: pd.Timestamp) -> list:
  # 'lstm_data' is a 2D array with dimensions (time_step, 1), i.e.,
  # 'batch_size' has already been sliced and this is a single batch
  # (1 is for 'univariate')
  # Create a dict of length 'batch_size' containing key-value pairs of
  # 'timestamp': 'value' by flattening data and adding timestamps
  analogue_input_data = {}
  
  flat_data = lstm_data[:, 0]  

  timestamps = []
  timestamp = start_timestamp

  for i in range(len(flat_data)):
    if i > 0:
      timestamp += pd.Timedelta(1, 'hr')

    timestamps.append(timestamp)

  analogue_input_data = {timestamp: value for (timestamp, value) in zip(timestamps, flat_data)}
  assert len(list(analogue_input_data.keys())) == len(flat_data) 
  
  return analogue_input_data

# =========================================================================
# Timestamping
# =========================================================================
def add_timestamps_to_data(data: np.ndarray, start_timestamp: pd.Timestamp):
  # 'data' is a 1D array of values. Adds timestamps incrementing upwards
  # by 1 hour per value
  timestamped_data = {}
  for i in range(len(data)):
    timestamp = start_timestamp + pd.Timedelta(i, unit='hr')
    timestamped_data[timestamp] = data[i]

  return timestamped_data
  
# =========================================================================
# Time conversion utility functions
# =========================================================================
def time_window_to_time_delta(time_window: Quantity) -> pd.Timedelta:
  value, unit = time_window.value, str(time_window.unit)

  # More generally, translate Astropy's units into Pandas'
  if unit == 'h':
    unit = 'hr'
    
  time_delta = pd.Timedelta(value, unit=unit)
  
  return time_delta

