# Functions to load and process data, including support for timestamps.
# Used for setting up LSTM and baseline metrics

# =========================================================================
# Imports
# =========================================================================
from typing import Dict
import numpy as np
import heliopy.data.omni as omni
from astropy.units import Quantity
from datetime import datetime, timedelta

import torch
import pandas as pd


# =========================================================================
# Datetime utility functions
# =========================================================================
def datetime_from_cycle(solar_cycles: pd.DataFrame, cycle: int,
                        key='start_min', fmt='%Y-%m-%d'):
  # From the 'solar_cycles' DataFrame, get the 'key' datetime (formatted as
  # %Y-%m-%d by default in the csv) as a datetime
  return datetime.strptime(solar_cycles.loc[cycle][key], fmt)


# =========================================================================
# Data loading
# =========================================================================
def get_omni_rtn_data(start_time: datetime, end_time: datetime):
  identifier = 'OMNI_COHO1HR_MERGED_MAG_PLASMA'  # COHO 1HR data

  # Change 'start' and 'end' times to include an extra hour since the
  # getter is exclusive of edges
  omni_data = omni._omni(start_time - timedelta(hours=1), end_time + timedelta(hours=1),
                         identifier=identifier, intervals='yearly', warn_missing_units=False)

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
def load_solar_cycles_df():
  solar_cycles_csv = '../res/solar_cycles.csv'
  solar_cycles_df = pd.read_csv(solar_cycles_csv, index_col=0)

  return solar_cycles_df

def split_into_n_hour_sections(data: np.ndarray, n=24):
  model_inputs = np.array([data[i:i+n] for i in range(len(data) - n)])[:, :, np.newaxis]
  model_outputs = np.array(data[n:])

  # Check for and remove NaNs
  model_inputs, model_outputs = remove_nans_from_data(data, model_inputs, model_outputs)

  return model_inputs, model_outputs

def get_cycle_idx(data: pd.Series, cycles_df: pd.DataFrame, cycle: int):
  # Get start and end indices from 'cycles_df' by slicing the
  # DateTime-index Series 'data'
  cycle_start = datetime_from_cycle(cycles_df, cycle)
  cycle_end = datetime_from_cycle(cycles_df, cycle, key='end')

  idx_start = data.index.searchsorted(cycle_start)
  idx_end = data.index.searchsorted(cycle_end)

  return idx_start, idx_end


def slice_data_ranges(input_data: np.ndarray, output_data: np.ndarray,
                      idx_ranges=[(0, 1)]):
  # Slice the data based on index ranges, then concatenate together
  # e.g. idx ranges [(0, 2), (4, 5)] will slice '...data[0:2]' then
  # '...data[4:5]' and concatenate these together to return
  sliced_input = []
  sliced_output = []

  for idx_range in idx_ranges:
    idx_start, idx_end = idx_range
    # print(idx_start, idx_end, idx_range)
    sliced_input.extend(input_data[idx_start: idx_end])
    sliced_output.extend(output_data[idx_start: idx_end])
  
  return np.array(sliced_input), np.array(sliced_output)


# Deprecated
def split_train_val_test(input_data: np.ndarray, output_data: np.ndarray,
  start_timestamp=None, train_idx_ranges=[(0, 134929)],
  val_idx_ranges=[(134929, 179456)], test_idx_ranges=[(179456, -1)]):
  # Split based on hard-coded indices present in OMNI data
  # Order is train -> validation -> test
  inputs_train, outputs_train = slice_data_ranges(input_data, output_data, train_idx_ranges)
  inputs_val, outputs_val = slice_data_ranges(input_data, output_data, val_idx_ranges)
  inputs_test, outputs_test = slice_data_ranges(input_data, output_data, test_idx_ranges)

  return inputs_train, inputs_val, inputs_test, outputs_train, outputs_val, outputs_test
  
  # The following code is unused for the LSTM and is solely for the
  # 'analogue_ensemble'
  # If 'start_timestamp' was given, calculate TimeStamps validation and
  # test start
  # if start_timestamp is not None:
  #   # Assuming hourly cadence (OMNI data)
  #   validation_timestamp = start_timestamp + time_window_to_time_delta(train_idx_end *u.hr)
  #   test_timestamp = validation_timestamp + time_window_to_time_delta((val_idx_end - val_idx_start) * u.hr)
  #   end_timestamp = test_timestamp + time_window_to_time_delta(len(input_data[val_idx_end:]) * u.hr)

  #   return inputs_train, inputs_val, inputs_test, outputs_train, outputs_val, outputs_test, validation_timestamp, test_timestamp, end_timestamp
  
  # else:
  #   return inputs_train, inputs_val, inputs_test, outputs_train, outputs_val, outputs_test


def omni_preprocess(start_time: datetime, end_time: datetime, keys=["BR"],
    make_tensors=False, split_mini_batches=False, n_hours=24,
    train_cycles=[21, 23], val_cycles=[22], test_cycles=[24]):
  # Wrapper function around 'get_omni_rtn_data()' and 'split...()'
  # 'n_hours' is number of hours to split data into (default 24)
  # By default, splits into cycles 21, 23 for training, 22 for validation
  # and 24 for testing
  # Note that 'start_time' and 'end_time' must span all cycles (this is
  # not checked below), e.g. for the default case which includes cycles
  # 21-24, the minimum time window is from the start of cycle 21 to the
  # end of cycle 24
  data = get_omni_rtn_data(start_time, end_time).to_dataframe()
  output_data = {}

  # Load solar cycles DF
  cycles_df = load_solar_cycles_df()
  
  # Determine index ranges for train, val and test data from cycles
  train_idx_ranges = [get_cycle_idx(data[keys[0]], cycles_df, train_cycle) for train_cycle in train_cycles]
  val_idx_ranges = [get_cycle_idx(data[keys[0]], cycles_df, val_cycle) for val_cycle in val_cycles]
  test_idx_ranges = [get_cycle_idx(data[keys[0]], cycles_df, test_cycle) for test_cycle in test_cycles]

  for key in keys:
    array = np.array(data[key])
    arr_input, arr_output = split_into_n_hour_sections(array, n_hours)
    # train_in, val_in, test_in, train_out, val_out, test_out,\
    #   val_timestamp, test_timestamp, end_timestamp =\
    #     split_train_val_test(arr_input, arr_output,
    #                          start_timestamp=initial_timestamp)

    # Slice train, val, test
    train_in, train_out = slice_data_ranges(arr_input, arr_output, train_idx_ranges)
    val_in, val_out = slice_data_ranges(arr_input, arr_output, val_idx_ranges)
    test_in, test_out = slice_data_ranges(arr_input, arr_output, test_idx_ranges)

    arr_dict = {
      "train_in": train_in, 
      "val_in": val_in,
      "test_in": test_in,
      "train_out": train_out,
      "val_out": val_out,
      "test_out": test_out,
      # "val_timestamp": val_timestamp,
      # "test_timestamp": test_timestamp,
      # "end_timestamp": end_timestamp
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
      arr_dict = split_data_mini_batches(arr_dict, 2048, input_batch_dim=batch_dim)

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

