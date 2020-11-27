from logging import error
import heliopy.data.omni as omni
import matplotlib.pyplot as plt
import pandas as pd
import astropy.units as u
from astropy.units import Quantity
from datetime import datetime
import numpy as np
from typing import Union, Tuple
import math


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

# Move to list utilities
def get_middle_idx(list_):
  # Note this always rounds down
  middle_idx = math.trunc(len(list_) / 2)

  return middle_idx


if __name__ == "__main__":
  # 14 month window for finding analogues
  data_start_time = datetime(2017, 1, 1)
  data_end_time = datetime(2018, 2, 28)

  # Get OMNI data for specified range
  omni_data = get_omni_rtn_data(data_start_time, data_end_time).to_dataframe()

  # Define properties for forecast
  forecast_time = pd.Timestamp(2017, 12, 5, 10, 0)
  training_window = 24 * (u.hr)  # 24 hours before 'forecast_time' 
  forecast_window = 24 * (u.hr)  # 24 hours after 'forecast_time'
  num_analogues = 10  # Number of analogues to find for ensemble
  num_points = int(training_window.value)  # Utility for slicing

  # Test with radial magnetic field strength
  br_data = omni_data['BR']

  # Get the pre-forecast and post-forecast data (include forecast after)
  trend_start_time = forecast_time - time_window_to_time_delta(training_window)
  br_data_before = br_data[br_data.index < trend_start_time]
  br_data_after = br_data[br_data.index >= forecast_time]

  # Get the current trend
  current_trend_mask = (br_data.index >= trend_start_time) & (br_data.index < forecast_time)
  current_br_trend = br_data[current_trend_mask]

  # Get true observations over training and forecast period
  observed_end_time = forecast_time + time_window_to_time_delta(training_window)
  observed_mask  = (br_data.index >= trend_start_time) & (br_data.index < observed_end_time)
  observed_trend = br_data[observed_mask]

  # Calculate MSE for 'before' data, used to find the best analogues
  # i.e. find the analogues most similar to the trend we have just seen,
  # using MSE as a test metric
  error_matrix = np.full((len(br_data_before), len(current_br_trend)), np.NaN)

  # Roll data in adjacent columns so that each row represents an analogue
  for i in range(len(current_br_trend)):
    error_matrix[:, i] = np.roll(br_data_before.values, -i)

  # Remove rows without data (truncated from roll)
  error_matrix = error_matrix[: -(len(current_br_trend) - 1), :]

  # Compute squared error
  error_matrix = np.square(error_matrix - current_br_trend.values)

  # MSE - take mean over analogues
  mse_matrix = np.mean(error_matrix, axis=1)

  # Add errors to DataFrame
  df = br_data_before.to_frame().iloc[: -(len(current_br_trend) -1), :]

  df['mse'] = mse_matrix
  analogue_df = df.nsmallest(num_analogues, 'mse')

  # DateTime indices of best-analogues (start times of each)
  analogue_start_times = analogue_df.index
  
  # Full matrix includes analogues before and after forecast
  analogue_matrix = np.full((len(analogue_df), num_points * 2), np.NaN)
  for i, start_time in enumerate(analogue_start_times):
    end_time = start_time + time_window_to_time_delta(training_window) \
      + time_window_to_time_delta(forecast_window)
    analogue = br_data[start_time: end_time].iloc[:-1]
    
    analogue_matrix[i] = analogue

  # 'Nanmedian' prediction
  analogue_prediction = np.nanmedian(analogue_matrix, axis=0)

  # Plot observed, analogue ensemble and constituent analogues
  fig, axes = plt.subplots(1, 1, figsize=(12, 12))

  # Set up 'x' to be a time axis going from -12 -> 12, with 0 as forecast
  # Make 'before' and 'after' forecast time
  time_axis_before = np.linspace(-24, 0, num=24)
  time_axis_after = np.linspace(0, 24, 24)
  time_axis_full = np.linspace(-24, 24, 48)

  # Plot best analogues
  axes.plot(time_axis_full, analogue_matrix.T, color="lightgrey")

  # Plot observations
  axes.plot(time_axis_full, observed_trend, color='blue', label='Observation')

  # Plot analogue prediction
  axes.plot(time_axis_full, analogue_prediction, color='red', label='Median Prediction')

  # Aesthetics
  axes.axvline(color='k', ls='--')
  axes.legend()

  plt.savefig('./analogue_test.svg')
