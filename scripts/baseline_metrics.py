import numpy as np
from .loss_functions import *
import tensorflow.keras as keras


def naive_forecast(data: np.array) -> float:
  # Choose the last value and use that as the forecast
  # Assumes 'data' is a 2D array (as input to TensorFlow)
  return data[:, -1]


def fully_connected_nn():
  model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[50, 1]),
    keras.layers.Dense(1)
  ])