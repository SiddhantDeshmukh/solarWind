import tensorflow.keras as keras
from tensorflow.keras import backend as K
from typing import List


class CustomAttention(keras.layers.Layer):
  def __init__(self, return_sequences=True):
    self.return_sequences = return_sequences
    super(CustomAttention, self).__init__()

  def build(self, input_shape):
    self.W = self.add_weight(name="att_weight", shape=(
        input_shape[-1], 1), initializer='normal')
    self.b = self.add_weight(name="att_bias", shape=(
        input_shape[1], 1), initializer='zeros')

    super(CustomAttention, self).build(input_shape)

  def call(self, x):
    e = K.tanh(K.dot(x, self.W) + self.b)
    a = K.softmax(e, axis=1)

    output = x * a

    return output if self.return_sequences else K.sum(output, axis=1)


def simple_rnn(input_length=24, num_features=1, output_length=1):
  model = keras.models.Sequential([
      keras.layers.SimpleRNN(20, input_shape=[input_length, num_features]),
      keras.layers.RepeatVector(output_length),
      keras.layers.TimeDistributed(
          keras.layers.Dense(num_features, activation="linear")
      ),
  ])

  return model


def deep_rnn(input_length=24, num_features=1, output_length=1):
  model = keras.models.Sequential([
      keras.layers.SimpleRNN(20, return_sequences=True,
                             input_shape=[input_length, num_features]),
      keras.layers.RepeatVector(output_length),
      keras.layers.SimpleRNN(20, return_sequences=True),
      keras.layers.TimeDistributed(
          keras.layers.Dense(num_features, activation="linear")
      ),
  ])

  return model


def lstm_model(input_length=24, num_features=1, output_length=1):
  # Multiple parallel input and multi-step output using
  # TimeDistributed Dense layer
  model = keras.models.Sequential([
      keras.layers.LSTM(32, input_shape=[input_length, num_features]),
      keras.layers.RepeatVector(output_length),
      keras.layers.LSTM(32,  return_sequences=True),
      keras.layers.TimeDistributed(
          keras.layers.Dense(num_features, activation="linear")
      ),
  ])

  return model


def lstm_attention_model(input_length=24, num_features=1, output_length=1):
  # Similar to 'lstm_model' but builds an additional attention network
  model = keras.models.Sequential([
      keras.layers.LSTM(32, input_shape=(input_length, num_features),
                        return_sequences=True),
      CustomAttention(return_sequences=True),
      keras.layers.LSTM(32, return_sequences=True),
      keras.layers.TimeDistributed(
          keras.layers.Dense(num_features, activation='linear')
      )
  ])

  return model


def conv_gru_model(input_length=24, num_features=1, output_length=1):
  # 1D conv layer to process sequences, then GRU cells for time series
  # predictions
  model = keras.models.Sequential([
      keras.layers.Conv1D(filters=20, kernel_size=4, strides=2,
                          padding="valid",
                          input_shape=[input_length, num_features]),
      keras.layers.GRU(20, return_sequences=True),
      keras.layers.RepeatVector(output_length),
      keras.layers.GRU(20, return_sequences=True),
      keras.layers.TimeDistributed(
          keras.layers.Dense(num_features, activation="linear")
      ),
  ])

  return model


def create_lstm_model(num_lstm_layers: int,
                      lstm_layer_neurons: List[int],
                      num_dense_layers: int,
                      dense_layer_neurons: List[int],
                      cudnn=False) -> keras.models.Model:
  model = keras.models.Sequential()

  # Add LSTM layers
  for i in range(num_lstm_layers):
    layer = keras.layers.CuDNNLSTM if cudnn else keras.layers.LSTM
    if i == 0:
      # First layer
      if i + 1 != num_lstm_layers:
        # Multiple layer LSTM
        model.add(layer(lstm_layer_neurons[i], input_shape=(
            None, 11), return_sequences=True))
      else:
        # Single layer LSTM
        model.add(layer(lstm_layer_neurons[i], input_shape=(None, 11)))
    elif i + 1 == num_lstm_layers:
      # Last layer
      model.add(layer(lstm_layer_neurons[i]))
    else:
      model.add(layer(lstm_layer_neurons[i], input_shape=(
          None, 11), return_sequences=True))

  # Add Dense layers
  for i in range(num_dense_layers):
    layer = keras.layers.Dense
    if i + 1 == num_dense_layers:
      model.add(layer(dense_layer_neurons[i], activation='linear'))
    else:
      model.add(layer(dense_layer_neurons[i], activation='relu'))

  return model
