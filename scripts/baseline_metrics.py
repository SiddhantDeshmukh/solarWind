import numpy as np
import tensorflow.keras as keras


def naive_forecast_start(data: np.array) -> np.array:
  # Choose the first value of each 24-hour period as the forecast
  # Assumes 'data' is a 3D array (as input to TensorFlow)
  start_data = data[:, 0, 0]
  return start_data


def naive_forecast_end(data: np.array) -> np.array:
  # Choose the last value of each 24-hour period as the forecast
  # Assumes 'data' is a 3D array (as input to TensorFlow)
  end_data = data[:, -1, 0]  # Last 24 hour period
  return end_data


def mean_forecast(data: np.array) -> np.array:
  # Choose the mean value of each 24-hour period as the forecast
  # Assumes 'data' is a 3D array (as input to TensorFlow)
  mean = np.mean(data, axis=1).flatten()

  return mean

def median_forecast(data: np.array) -> np.array:
  # Choose the median value of each 24-hour period as the forecast
  # Assumes 'data' is a 3D array (as input to TensorFlow)
  median = np.median(data, axis=1).flatten()

  return median


def fully_connected_nn():
  model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[50, 1]),
    keras.layers.Dense(1)
  ])