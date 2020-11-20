import numpy as np
from .loss_functions import *
import tensorflow.keras as keras


def naive_forecast_start(data: np.array) -> float:
  # Choose the last value and use that as the forecast
  # Assumes 'data' is a 2D array (as input to TensorFlow)
  return data[:, -1]


def naive_forecast_start(data: np.array) -> float:
  # Choose the first value and use that as the forecast
  # Assumes 'data' is a 2D array (as input to TensorFlow)
  return data[:, 0]


def mean_forecast(data: np.array) -> float:
  # Choose the first value and use that as the forecast
  # Assumes 'data' is a 2D array (as input to TensorFlow)
  mean = np.mean(data, axis=1)

  return mean


def fully_connected_nn():
  model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[50, 1]),
    keras.layers.Dense(1)
  ])