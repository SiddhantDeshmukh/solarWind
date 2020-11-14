import heliopy.data.omni as omni
import matplotlib.pyplot as plt
import pandas as pd
import astropy.units as u
from astropy.units import Quantity
from datetime import datetime
import numpy as np
from typing import Union, Tuple
import math


# Move to list utilities
def get_middle_idx(list_):
  # Note this always rounds down
  middle_idx = math.trunc(len(list_) / 2)

  return middle_idx


def get_omni_rtn_data(start_time, end_time):
  identifier = 'OMNI_COHO1HR_MERGED_MAG_PLASMA'  # COHO 1HR data
  omni_data = omni._omni(start_time, end_time, identifier=identifier, intervals='yearly', warn_missing_units=False)

  return omni_data


def time_window_to_time_delta(time_window: Quantity) -> pd.Timedelta:
  value, unit = time_window.value, str(time_window.unit)

  # More generally, translate Astropy's units into Pandas'
  if unit == 'h':
    unit = 'hr'
    
  time_delta = pd.Timedelta(value, unit=unit)
  
  return time_delta

def slice_training_data(df: pd.DataFrame, start_time: pd.Timestamp, training_window: Quantity) -> pd.DataFrame:
  # Slice DateTime indexed DataFrame 'df' to a window ['start_time' - 'training_window']  
  training_start_time = start_time - time_window_to_time_delta(training_window)
  sliced_df = df[(df.index > training_start_time) & (df.index < start_time)]

  return sliced_df
  
def slice_df_using_blockout(df: pd.DataFrame, forecast_start_time: pd.Timestamp, 
    block_out_period: Quantity) -> Tuple[pd.DataFrame, pd.DataFrame]:
  # Slice DataFrame 'df' into 'before' and 'after' Dataframes around a 
  # specified block-out period (which is not included in either DataFrame)
  block_out_start = forecast_start_time - time_window_to_time_delta(block_out_period.to(u.s))
  block_out_end = forecast_start_time + time_window_to_time_delta(block_out_period.to(u.s))

  df_before, df_after = df[(df.index < block_out_start)], df[(df.index > block_out_end)]

  return df_before, df_after


def calculate_error_matrix(array1_values: np.array, array2_values: np.array) -> np.array:
  # Calculate error between arrays given a loss metric and create a 2D
  # matrix of error values
  matrix = np.full((len(array1_values), len(array2_values)), np.NaN)

  # Roll data in adjacent columns so that each row represents possible analogue
  for i in range(len(array2_values)):
    matrix[:, i] = np.roll(array1_values, -i)
  
  # Remove rows that do not have data due to the roll
  matrix = matrix[:-(len(array2_values) - 1), :]

  # Compute squared error (should support different loss functions)
  matrix = np.square(matrix - array2_values)
  
  return matrix


def weight_matrix(matrix: np.array, weighting_type: Union[str, np.array]) -> np.array:
  # Takes in 2D error matrix and 1D weighting array and applies weights
  weighted_mean_matrix = None
  if weighting_type:
      if weighting_type == 'linear':
          weight = np.linspace(0, 1, matrix.shape[1]+1)[1:]  #  define array with values for weighting
          weighted_matrix = matrix * np.array(weight)  #  multiple the matrix value by the appropraite weight
          weighted_mean_matrix = np.mean(weighted_matrix, axis=1) # compute the mean of error of each analogue
      
      elif weighting_type == 'quadratic':
          weight = np.linspace(0, 1, matrix.shape[1] + 1)[1:]**2 #define array with values for weighting
          weighted_matrix = matrix * np.array(weight) # multiple the matrix value by the appropraite weight
          weighted_mean_matrix = np.mean(weighted_matrix, axis=1) # compute the mean of error of each analogue
      
      elif weighting_type == 'cubic':
          weight = np.linspace(0, 1, matrix.shape[1] + 1)[1:] ** 3 #define array with values for weighting
          weighted_matrix = matrix * np.array(weight) # multiple the matrix value by the appropraite weight
          weighted_mean_matrix = np.mean(weighted_matrix, axis=1) # compute the mean of error of each analogue
      
      elif isinstance(weighting_type, (list, np.ndarray)): # check if given weighting array is a list or np.ndarray
          weight = weighting_type #define array with values for weighting as the array inputed
          weighted_matrix = matrix * np.array(weight) # multiple the matrix value by the appropraite weight
          weighted_mean_matrix = np.mean(weighted_matrix, axis=1) # compute the mean of error of each analogue
  else:
      weighted_mean_matrix = np.mean(matrix, axis=1) # if no valid weighting rule is specified then compute the unweighted mean error for each analogue
  
  return weighted_mean_matrix


def get_best_analogues(df: pd.DataFrame, error_key: str, num_analogues: int):
  # Get a DataFrame of 'n' analogues with the smallest errors (stored in 'error_column' of 'df')
  analogue_df = df.nsmallest(num_analogues, error_key)  # for DF, specify column
  # analogue_df = df.nsmallest(num_analogues)  # for Series, no column
  # analogue_df = analogue_df.sort_values(by='index')

  return analogue_df

def remove_overlapping_analogues(analogue_df: pd.DataFrame, error_key: str, training_window: Quantity):
  # Find analogues that overlap and remove the one with the highest error
  df = analogue_df.copy(deep=True)
  i = 0

  # Sort through DF to ensure none of analogues overlap in time
  # Since they are indexed in time, if the first analogue does not overlap
  # with the second one, it does not overlap with any
  print(analogue_df)
  # while i < len(df) - 2:
  #   while i < len(df) - 2 and abs(df.index[i + 1] - df.index[i]) < time_window_to_time_delta(training_window):
  #     current_error, next_error = df[error_key].iloc[i], df[error_key].iloc[i + 1]
  #     drop_idx = i if next_error < current_error else i + 1
  #     drop_idx = analogue_df.index[drop_idx]
      

  #     df.drop(drop_idx)

  #     i += 1

  return df


def wmse_n_non_overlapping(num_analogues: int, error_key: str, error_array: np.array, blocked_out_series: pd.Series, training_window: Quantity):
  # Pick best 'num_analogues' analogues that do not overlap
  # Add errors to DataFrame
  blocked_out_df = blocked_out_series.to_frame()
  blocked_out_df[error_key] = error_array

  # Get analogues
  analogue_df = get_best_analogues(blocked_out_df, error_key, num_analogues)
  # analogue_df = remove_overlapping_analogues(analogue_df, error_key, training_window)

  # if len(analogue_df) < num_analogues: # check analogue_df still contains n analogues
  #   mult = 2
  #   while (len(analogue_df) < num_analogues) & (mult < len(blocked_out_df) / training_window.value): #iterate until there are n or more analogues that do not overlap
  #     analogue_df = get_best_analogues(blocked_out_df, num_analogues * mult) # find best n*mult analogues
  #     analogue_df = remove_overlapping_analogues(analogue_df, error_key, training_window) #remove analogues overlapping in time
  #     mult = mult * 2 #increase value of mult

  #   analogue_df = get_best_analogues(analogue_df, error_key, num_analogues) # cut down to n best analogues  

  # elif len(analogue_df) > num_analogues: # if there are too many analogues then choose n best
  #   analogue_df = get_best_analogues(analogue_df, error_key, num_analogues)

  return analogue_df


def round_down(num, divisor):
    '''
    rounds down to the nearest "divisor"
    :param num:
    :param divisor:
    :return:
    '''
    return num - (num%divisor)

def get_analogues(analogue_df: pd.DataFrame, full_data_df: pd.DataFrame, training_window: Quantity, lead_time: Quantity, temporal_resolution: Quantity):
  # Creates 2D array of analogues for 't0 < t <= t0 + lead_time'
  start_time = analogue_df.index - time_window_to_time_delta(training_window)
  end_time = analogue_df.index + time_window_to_time_delta(lead_time)

  # Sample based on temporal resolution
  num_points = int(round_down(training_window, temporal_resolution) / temporal_resolution  + round_down(lead_time, temporal_resolution) / temporal_resolution) #number of 3-hourly datapoints each analogue period contains
  analogue_matrix = np.full((len(analogue_df), num_points), np.NaN)

  for i in range(len(analogue_df)):
    # Slice to start - end indices
    analogue = full_data_df[(full_data_df.index > start_time[i]) & (full_data_df.index <= end_time[i])]
    
    # Resample based on temporal resolution
    # Note resampling Series like this changes the start and end labels, 
    # so the left interval is closed and the right interval is truncated
    # if necessary

    # There must be a better way!
    analogue = analogue.resample(f'{int(temporal_resolution.value)}{temporal_resolution.unit}', closed='left').mean()
    try:
      analogue_matrix[i] = analogue.values
    except ValueError:  # truncate analogue on right interval
      analogue = analogue[:-1]
      try: 
        analogue_matrix[i] = analogue.values
      except ValueError:  # resolution sampling got number of points wrong
        # ignore this analogue
        continue


  return analogue_matrix


def compute_analogue_median(analogue_matrix: np.array):
  analogue_median =  np.nanmedian(analogue_matrix, axis=0)

  return analogue_median


def get_data_after_forecast_time(df: pd.DataFrame, forecast_time: pd.Timestamp, forecast_window: Quantity):
  end_time = (forecast_time + time_window_to_time_delta(forecast_window))

  assert end_time > forecast_time

  mask_before = (df.index > forecast_time)
  mask_after = (df.index <= end_time)
  mask = mask_before & mask_after

  assert len([item for item in mask if item == True]) > 0

  data_after_forecast = df[mask]

  return data_after_forecast


def run_analogue_ensemble(observed_data: pd.Series, 
    forecast_start_time: pd.Timestamp, training_window: Quantity, forecast_window: Quantity, 
    temporal_resolution: Quantity, block_out_period: Quantity, 
    num_analogues: int):
  # Sample windows based on temporal resolution
  training_window = round_down(training_window, temporal_resolution)
  forecast_window = round_down(forecast_window, temporal_resolution)

  # Block out data around forecast time 
  observed_data_before, observed_data_after = slice_df_using_blockout(observed_data, forecast_start_time, block_out_period)

  # Get training window data (data before forecast time within forecast window)
  training_data = slice_training_data(observed_data, forecast_start_time, training_window)

  # Calculate error between training data and all observations
  squared_error_before = calculate_error_matrix(observed_data_before.values, training_data.values)
  squared_error_after = calculate_error_matrix(observed_data_after.values, training_data.values)
  squared_error_matrix = np.concatenate([squared_error_before, squared_error_after])

  # This might be the error in the plot
  observed_data = pd.concat([observed_data_before.iloc[len(training_data) - 1:], observed_data_after.iloc[len(training_data) - 1:]])
  # //

  # Calculate weighted mean of squared error
  wmse = weight_matrix(squared_error_matrix, weighting_type='linear')
  error_key = 'wmse'

  # Get analogues based on closest 'n' matches based on error
  analogue_df = wmse_n_non_overlapping(num_analogues, error_key, wmse, observed_data, training_window)
  analogue_matrix = get_analogues(analogue_df, observed_data, training_window, forecast_window, temporal_resolution)
  analogue_weighted_median = compute_analogue_median(analogue_matrix)

  # Stitch observed data together across training window
  observed_data_forecast = get_data_after_forecast_time(observed_data, forecast_start_time, forecast_window)
  observed_data_ensemble = pd.concat((training_data, observed_data_forecast))

  return analogue_weighted_median, analogue_matrix, observed_data_ensemble


def plot_analogue_ensemble(ax: plt.Axes, analogue_weighted_median: np.array, analogue_matrix: np.array, observed_data: pd.DataFrame, temporal_resolution: Quantity):
  # Refactor to use the 'tools'/'usefulScripts' generalised functions
  
  # Get x-axes to centre data around forecast time  
  analogue_middle_index = get_middle_idx(analogue_weighted_median)
  num_points = len(analogue_weighted_median)
  x_grid = temporal_resolution * np.linspace(-analogue_middle_index+1, num_points - analogue_middle_index, num_points)

  # Analogues
  ax.plot(x_grid, analogue_matrix.T, color="lightgrey")
  ax.plot(x_grid, analogue_weighted_median, color="red", label="Weighted Median Prediction")

  # Observations
  ax.plot(x_grid, observed_data, label="Observation")

  # Aesthetics
  ax.axvline(color='k', ls='--')
  ax.legend()


if __name__ == "__main__":
  start_time = (datetime(2017, 1, 1))
  end_time = (datetime(2018, 2, 28))

  omni_data = get_omni_rtn_data(start_time, end_time)

  # AnEn using pre-loaded data assuming DateTime indices
  # Written for use with CDAWeb OMNI datasets or any Pandas DataFrame
  # with a DateTime index
  start_time = pd.Timestamp(2017, 12, 5, 10, 0)  # read this in
  training_window = 24 * (u.hr)  # in hours, read this in
  forecast_window = 24 * (u.hr)  # in hours, read this in
  temporal_resolution = 2 * (u.hr)  # in hours, read this in
  block_out_period = 1 * (u.hr)
  num_analogues = 10  # Take in as parameter


  # Radial magnetic field strength
  observed_data = omni_data.to_dataframe()
  observed_br_data = observed_data['BR']
  analogue_weighted_median_br, analogue_matrix_br, observed_data_br = run_analogue_ensemble(observed_br_data, start_time, 
      training_window, forecast_window, temporal_resolution, block_out_period, num_analogues)


  # Bulk flow speed
  observed_v_data = observed_data['V']
  analogue_weighted_median_v, analogue_matrix_v, observed_data_v = run_analogue_ensemble(observed_v_data, start_time, 
      training_window, forecast_window, temporal_resolution, block_out_period, num_analogues)

  # Plot predictions
  fig, axes = plt.subplots(1, 2, figsize=(10, 5))
  
  # Radial magnetic field strength
  plot_analogue_ensemble(axes[0], analogue_weighted_median_br, analogue_matrix_br, observed_data_br, temporal_resolution)
  axes[0].set_xlabel(r'Time from $t_0$ (hours)')
  axes[0].set_ylabel(r'$B_r (nT)$')

  # Bulk flow speed
  plot_analogue_ensemble(axes[1], analogue_weighted_median_v, analogue_matrix_v, observed_data_v, temporal_resolution)
  axes[1].set_xlabel(r'Time from $t_0$ (hours)')
  axes[1].set_ylabel(r'$V (km / s)$')

  plt.savefig('./omni_AnEn.svg')
