import matplotlib.pyplot as plt
import pandas as pd
import astropy.units as u
from astropy.units import Quantity
from datetime import datetime
import numpy as np
from typing import Union
import math


# =========================================================================
# List utility functions
# =========================================================================
# Move to list utilities
def get_middle_idx(list_: list) -> int:
  # Note this always rounds down
  middle_idx = math.trunc(len(list_) / 2)

  return middle_idx

# =========================================================================
# Error matrix functions
# =========================================================================


def create_error_matrix(data_before_forecast: np.array,
                        current_trend: np.array) -> np.array:
  error_matrix = np.full(
      (len(data_before_forecast), len(current_trend)), np.NaN)

  # Roll data in adjacent columns so that each row represents an analogue
  for i in range(len(current_trend)):
    error_matrix[:, i] = np.roll(data_before_forecast.values, -i)

  # Remove rows without data (truncated from roll)
  error_matrix = error_matrix[: -(len(current_trend) - 1), :]

  return error_matrix


def mse_error_matrix(data_before_forecast: np.array,
                     current_trend: np.array) -> np.array:
  # Calculate MSE for 'before' data, used to find the best analogues
  # i.e. find the analogues most similar to the trend we have just seen,
  error_matrix = create_error_matrix(data_before_forecast, current_trend)

  # Compute squared error
  error_matrix = np.square(error_matrix - current_trend.values)

  # MSE - take mean over analogues
  mse_matrix = np.nanmean(error_matrix, axis=1)

  return mse_matrix


def rmse_error_matrix(data_before_forecast: np.array,
                      current_trend: np.array) -> np.array:
  # Calculate RMSE for 'before' data, used to find the best analogues
  # i.e. find the analogues most similar to the trend we have just seen
  # RMSE - take root of mean over analogues
  mse_matrix = mse_error_matrix(data_before_forecast, current_trend)
  rmse_matrix = np.sqrt(mse_matrix)

  return rmse_matrix

# =========================================================================
# Analogue Ensemble functions
# =========================================================================
# Refactor to take in a cost function


def run_analogue_ensemble(data: pd.DataFrame, forecast_time_index: pd.Timestamp,
                          lookback_and_lookforward: int = 24,
                          num_analogues: int = 10) -> Union[np.array, np.array, pd.Series]:
  # Create an analogue ensemble prediction for a given set of data and
  # return the matrix of analogues, the prediction and the observed trend
  # Get the pre-forecast and post-forecast data (include forecast after)
  trend_start_time = forecast_time_index - lookback_and_lookforward
  trend_end_time = forecast_time_index + lookback_and_lookforward

  data_before_forecast = data.iloc[:trend_start_time]

  # Truncate data to be a maximum of 20000 points
  max_points = 20000
  if len(data_before_forecast) > max_points:
    data_before_forecast = data.iloc[trend_start_time -
                                     max_points:trend_start_time]

  current_trend = data[trend_start_time:forecast_time_index +
                       lookback_and_lookforward]
  observed_trend = data[trend_start_time:forecast_time_index +
                        lookback_and_lookforward]

  # Calculate error matrix *!based on key!*
  mse_matrix = mse_error_matrix(data_before_forecast, current_trend)

  # Remove rows from DataFrame that were removed when calculating error
  # matrix (due to the roll)
  df = data_before_forecast.to_frame().iloc[: -(len(current_trend) - 1), :]

  # Add errors to DataFrame
  df['mse'] = mse_matrix
  analogue_df = df.nsmallest(num_analogues, 'mse')

  # DateTime indices of best-analogues (start times of each)
  analogue_start_times = [np.where(data.index == i)[0][0]
                          for i in analogue_df.index]
  # Full matrix includes analogues before and after forecast
  analogue_matrix = np.full(
      (len(analogue_df), lookback_and_lookforward * 2), np.NaN)

  for i, start_time in enumerate(analogue_start_times):
    analogue_matrix[i] = data[start_time:start_time +
                              2 * lookback_and_lookforward]

  # 'Nanmedian' prediction
  analogue_prediction = np.nanmedian(analogue_matrix, axis=0)

  return analogue_matrix, analogue_prediction, observed_trend

# =========================================================================
# Plotting functions
# =========================================================================


def plot_analogue_ensemble(ax: plt.Axes, analogue_matrix: np.array,
                           analogue_prediction: np.array, observed_trend: np.array,
                           xlabel="Time [h]", ylabel="", title=""):
  fig = None
  if ax is None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

  # Set up 'x' to be a time axis going from -12 -> 12, with 0 as forecast
  # Make 'before' and 'after' forecast time
  # Note this is still hard-coded for a 24 hour training + 24 hour forecast
  # period
  time_axis_full = np.linspace(-24, 24, 48)

  # Plot best analogues
  ax.plot(time_axis_full, analogue_matrix.T, color="lightgrey")

  # Plot observations
  ax.plot(time_axis_full, observed_trend,
          color='blue', label='Observation')

  # Plot analogue prediction
  ax.plot(time_axis_full, analogue_prediction,
          color='red', label='Median Prediction')

  # Aesthetics
  ax.axvline(color='k', ls='--')

  ax.set_xlabel(xlabel, fontsize=18)
  ax.set_ylabel(ylabel, fontsize=18)
  ax.set_title(title, fontsize=20)

  ax.legend(fontsize=14)

  return fig, ax


if __name__ == "__main__":
  # 14 month window for finding analogues
  data_start_time = datetime(2017, 1, 1)
  data_end_time = datetime(2018, 2, 28)

  # Get OMNI data for specified range
  omni_data = get_omni_rtn_data(
      data_start_time, data_end_time).to_dataframe()

  # Define properties for forecast
  forecast_time = pd.Timestamp(2017, 12, 5, 10, 0)
  training_window = 24 * (u.hr)  # 24 hours before 'forecast_time'
  forecast_window = 24 * (u.hr)  # 24 hours after 'forecast_time'
  num_analogues = 10  # Number of analogues to find for ensemble

  # Test with radial magnetic field strength
  data = omni_data['BR']

  analogue_matrix, analogue_prediction, observed_trend = \
      run_analogue_ensemble(data, forecast_time, training_window,
                            forecast_window, num_analogues)

  # Plot observed, analogue ensemble and constituent analogues
  fig, axes = plt.subplots(1, 2, figsize=(16, 8))

  # Forecast time stamp as title
  subplots_title = f"Forecast: {forecast_time}"

  plot_analogue_ensemble(axes[0], analogue_matrix, analogue_prediction,
                         observed_trend, ylabel="Radial Magnetic Field Strength [nT]")

  # Test with radial wind speed
  data = omni_data["V"]

  analogue_matrix, analogue_prediction, observed_trend = \
      run_analogue_ensemble(data, forecast_time, training_window,
                            forecast_window, num_analogues)

  # Plot observed, analogue ensemble and constituent analogues
  plot_analogue_ensemble(axes[1], analogue_matrix, analogue_prediction,
                         observed_trend, ylabel=r"Radial Wind Velocity [km s$^{-1}$]")

  plt.suptitle(subplots_title, fontsize=20)
  plt.savefig('./omni_AnEn.svg')
