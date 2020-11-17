import tensorflow.keras as keras
from tensorflow.python.keras.engine import input_spec


def simple_rnn():
  model = keras.models.Sequential([
    keras.layers.SimpleRNN(1, input_shape=[None, 1])
  ])

  return model


def deep_rnn():
  model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.SimpleRNN(20, return_sequences=True),
    keras.layers.SimpleRNN(1)
  ])

  return model


def lstm_model():
  model = keras.models.Sequential([
    keras.layers.LSTM(20, activation="linear", name="lstm_initial", input_shape=[None, 1]),
    keras.layers.Dense(1, name="dense_final", activation="linear")
  ])

  return model


def conv_gru_model():
  # 1D conv layer to process sequences, then GRU cells for time series
  # predictions 
  model = keras.models.Sequential([
    keras.layers.Conv1D(filters=20, kernel_size=4, strides=2, 
      padding="valid", input_shape=[None, 1]),
    keras.layers.GRU(20, return_sequences=True),
    keras.layers.GRU(20, return_sequences=True),
    keras.layers.TimeDistributed()
  ])