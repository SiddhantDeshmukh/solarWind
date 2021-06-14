# Functions to load and process data, including support for timestamps.
# Used for setting up LSTM and baseline metrics

# =========================================================================
# Imports
# =========================================================================
from typing import Dict, List, Tuple
import numpy as np
import heliopy.data.omni as omni
from astropy.units import Quantity
from datetime import datetime

import torch
from torch.functional import Tensor
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from torch.utils.data.dataset import Dataset


# =========================================================================
# Classes
# =========================================================================
class OMNIDataset(Dataset):
  def __init__(self,
               start_time: datetime,
               end_time: datetime,
               keys: List) -> Dataset:
    # Get OMNI data between 'start_time' - 'end_time' and create a Dataset
    omni_data = get_omni_rtn_data(start_time, end_time).to_dataframe()

    # output tensor is 2D - time-series in '0', features in '1'
    output_tensor = torch.from_numpy(
        omni_data[keys[0].values]).unsqueeze(1)
    if len(keys) > 1:
      for i in range(1, len(keys)):
        key_tensor = torch.from_numpy(
            omni_data[keys[i].values]).unsqueeze(1)
        output_tensor = torch.cat((output_tensor, key_tensor), 1)

    self.data = output_tensor

# =========================================================================
# Normalisation
# =========================================================================


def normalise_tensor(tensor: Tensor, limits: Tuple):
  # Applies min-max normalisation
  # Takes a PyTorch Tensor and a tuple of (left_limit, right_limit)
  # to linearly normalise the data
  tensor_min, tensor_max = torch.min(tensor), torch.max(tensor)
  lower_lim, upper_lim = limits
  scaled_tensor = lower_lim + \
      ((tensor - tensor_min) * (upper_lim - lower_lim)) / \
      (tensor_max - tensor_min)

  return scaled_tensor


def standardise_array(array: np.ndarray, std_array: np.ndarray):
  # Normalises data to have a mean of 0 and a variance of 1
  # 'array' is the array to normalise, 'std_array' is the array to
  # use for mean and std_dev
  mean, std = np.nanmean(std_array, axis=0), \
      np.nanstd(std_array, axis=0)
  normalised_array = (array - mean) / std

  return normalised_array, mean, std


def unstandardise_array(normalised_array: np.ndarray, mean: float, std: float):
  # Given a mean and variance, inverts standardisation performed by
  # 'standardise_array()'
  array = normalised_array * std + mean

  return array

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

# =========================================================================
# Quantity calculations
# =========================================================================


def calculate_geoeffectiveness(wind_density: np.ndarray,
                               hmf_intensity: np.ndarray,
                               wind_speed: np.ndarray,
                               hmf_clock_angle: np.ndarray,
                               norm_angle=False) -> np.ndarray:
  # 'norm_angle' refers to the domain of 'hmf_clock_angle'.
  # False: theta ~ [-pi, pi] (standard)
  # True: theta ~ [0, 2pi] (new normalisation, subtract pi to revert)
  alpha = 0.5  # empirically determined

  # Revert domain to [-pi, pi] from [0, 2pi]
  if norm_angle:
    hmf_clock_angle -= np.pi

  geoffectiveness = wind_density**(2/3 - alpha) * \
      hmf_intensity**(2*alpha) * \
      wind_speed**(7/3 - 2 * alpha) * \
      np.sin(hmf_clock_angle / 2)**4

  return geoffectiveness
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
  # Download OMNI COHO1HR data
  identifier = 'OMNI_COHO1HR_MERGED_MAG_PLASMA'  # COHO 1HR data

  # Change 'start' and 'end' times to include an extra hour since the
  # getter is exclusive of edges
  omni_data = omni._omni(start_time, end_time,
                         identifier=identifier, intervals='yearly',
                         warn_missing_units=False)

  return omni_data


def load_solar_cycles_df():
  solar_cycles_csv = '../res/solar_cycles.csv'
  solar_cycles_df = pd.read_csv(solar_cycles_csv, index_col=0)

  return solar_cycles_df

# =========================================================================
# Data cleaning
# =========================================================================


def get_nan_idxs(data: np.ndarray, n=24):
  # 'n' corresponds to length of slice, default 24 (hourly cadence)
  # same 'n' as in 'split_into_n_hour_sections()'
  idx_edge = n + 1  # check 'n' elements forwards
  nan_check = np.array([data[i:i + idx_edge]
                        for i in range(len(data) - idx_edge + 1)])

  # Remove 'nan' indices
  nan_idxs = np.where([np.any(np.isnan(i)) for i in nan_check])[0]

  return nan_idxs.tolist()


def remove_nans_from_data(data: np.ndarray,
                          model_inputs: np.ndarray, model_outputs: np.ndarray, n=24):
  # 'n' corresponds to length of slice (default 24, same as in
  # 'split_into_n_hour_sections()')
  idx_edge = n + 1  # want to check 'n' elements forward
  nan_check = np.array([data[i:i + idx_edge]
                        for i in range(len(data) - idx_edge + 1)])
  model_inputs = model_inputs[np.where(
      [~np.any(np.isnan(i)) for i in nan_check])]
  model_outputs = model_outputs[np.where(
      [~np.any(np.isnan(i)) for i in nan_check])]

  print(f"Input shape: {model_inputs.shape}")
  print(f"Output shape: {model_outputs.shape}")

  print(
      f"Any NanNs? {np.any(np.isnan(model_inputs)) or np.any(np.isnan(model_outputs))}")

  return model_inputs, model_outputs


def lstm_prepare_nd(multi_array, input_length, output_length):
  inputs = np.array([multi_array[i:i + input_length]
                     for i in range(len(multi_array) - input_length - output_length + 1)])
  outputs = np.array([multi_array[i + input_length:i + input_length + output_length]
                      for i in range(len(multi_array) - input_length - output_length + 1)])

  nan_check = np.array([multi_array[i:i + input_length + output_length]
                        for i in range(len(multi_array) - input_length - output_length + 1)])

  inputs = inputs[np.where([~np.any(np.isnan(i)) for i in nan_check])]
  outputs = outputs[np.where([~np.any(np.isnan(i)) for i in nan_check])]

  print("Input shape:", inputs.shape)
  print("Output shape:", outputs.shape)
  print("Any Nans?:", np.any(np.isnan(outputs))
        or np.any(np.isnan(inputs)))
  return inputs, outputs

# =========================================================================
# Data splitting
# =========================================================================


def split_into_n_hour_sections(data: np.ndarray, n=24, remove_nans=True):
  model_inputs = np.array([data[i:i+n]
                           for i in range(len(data) - n)])[:, :, np.newaxis]
  model_outputs = np.array(data[n:])[:, np.newaxis]

  # Check for and remove NaNs
  if remove_nans:
    model_inputs, model_outputs = remove_nans_from_data(
        data, model_inputs, model_outputs)

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


def omni_cycle_preprocess(start_time: datetime, end_time: datetime,
                          keys=["BR"], get_geoeffectiveness=False,
                          make_tensors=False, standardise=False,
                          n_hours_in=24):
  # 'normalisation_limits' is tuple (min-max); all keys normalised to this
  data = get_omni_rtn_data(start_time, end_time).to_dataframe()

  # Overwrite keys and get inclination angle, calculate geoeffectiveness
  if get_geoeffectiveness:
    # clock angle
    print("Calculating HMF clock angle...")
    data['HMF_INC'] = np.arctan2(-data['BT'].values, data['BN'].values)
    # geoeffectiveness
    print("Calculating geoeffectiveness...")
    data['G'] = calculate_geoeffectiveness(
        data['N'].values, data['ABS_B'].values,
        data['V'].values, data['HMF_INC'].values)

    keys = ["N", "ABS_B", "V", "HMF_INC", "G"]

  data = data[keys]

  # Just divide 60/20/20 train/val/test
  train_end_idx = int(0.6 * len(data))
  val_end_idx = int(0.8 * len(data))

  train_data = data.values[:train_end_idx]
  val_data = data.values[train_end_idx:val_end_idx]
  test_data = data.values[val_end_idx:]

  train_in, train_out = lstm_prepare_nd(
      train_data, n_hours_in, n_hours_in)
  val_in, val_out = lstm_prepare_nd(
      val_data, n_hours_in, n_hours_in)
  test_in, test_out = lstm_prepare_nd(
      test_data, n_hours_in, n_hours_in)

  arr_dict = {
      "train_in": train_in,
      "train_out": train_out,
      "val_in": val_in,
      "val_out": val_out,
      "test_in": test_in,
      "test_out": test_out
  }

  if standardise:
    for key, values in arr_dict.items():
      # Do not standardise 'test'
      # Do not standardise outputs
      if key.endswith('_out'):
        print(f"Not standardising output for {key}, not standardising.")
      else:
        print(f"Standardising {key} to have mean 0, variance 1.")
        arr_dict[key], mean, std = standardise_array(values, train_data)

  if make_tensors:  # NumPy ndarray -> PyTorch Tensor
    # For LSTM, Torch expects [batch_size, seq_len, num_feat]
    # Same as TensorFlow! Make sure to flag 'batch_first=True' in LSTM()
    print("Creating tensors (for PyTorch!)")
    for arr_key in arr_dict.keys():
      if isinstance(arr_dict[arr_key], np.ndarray):
        tensor = torch.from_numpy(arr_dict[arr_key])
        arr_dict[arr_key] = tensor

  if standardise:
    return arr_dict, keys, mean, std
  else:
    return arr_dict, keys


def split_data_mini_batches(data: Dict, mini_batch_size: int,
                            input_size=3, output_size=1,
                            input_batch_dim=0, output_batch_dim=0):
  # For each 3D Tensor, split into mini-batches
  for key in data.keys():
    if isinstance(data[key], torch.Tensor):
      if len(data[key].shape) == input_size:  # input data
        mini_batches = torch.split(
            data[key], mini_batch_size, dim=input_batch_dim)

      elif len(data[key].shape) == output_size:  # output data
        mini_batches = torch.split(
            data[key], mini_batch_size, output_batch_dim)

      else:
        print(
            f"Warning: shape of '{key}' is not size 3 (input) or 1 (output).")
        print("Ignoring mini-batches.")
        mini_batches = data[key]

      # Pad last mini_batch
      mini_batches = pad_sequence(mini_batches).transpose(0, 1)
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

  analogue_input_data = {timestamp: value for (
      timestamp, value) in zip(timestamps, flat_data)}
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

  analogue_input_data = {timestamp: value for (
      timestamp, value) in zip(timestamps, flat_data)}
  assert len(list(analogue_input_data.keys())) == len(flat_data)

  return analogue_input_data
