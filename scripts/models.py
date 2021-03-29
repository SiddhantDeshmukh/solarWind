import tensorflow.keras as keras
from tensorflow.keras import backend as K
from typing import List

from tensorflow.python.keras.layers.core import RepeatVector


class CustomAttention(keras.layers.Layer):
  def __init__(self, return_sequences=True):
    self.return_sequences = return_sequences
    super(CustomAttention, self).__init__()

  def get_config(self):
    config = super().get_config().copy()
    config.update({'return_sequences': self.return_sequences})

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
  print("Creating simple RNN")
  model = keras.models.Sequential([
      keras.layers.SimpleRNN(20, input_shape=[input_length, num_features]),
      keras.layers.RepeatVector(output_length),
      keras.layers.TimeDistributed(
          keras.layers.Dense(num_features, activation="linear")
      ),
  ])

  return model


def deep_rnn(input_length=24, num_features=1, output_length=1):
  print("Creating deep RNN")
  model = keras.models.Sequential([
      keras.layers.SimpleRNN(20, return_sequences=False,
                             input_shape=[input_length, num_features]),
      keras.layers.RepeatVector(output_length),
      keras.layers.SimpleRNN(20, return_sequences=True),
      keras.layers.TimeDistributed(
          keras.layers.Dense(num_features, activation="linear")
      ),
  ])

  return model


def lstm_model(input_length=24, output_length=1, num_features=1,
               num_encoder_layers=1, encoder_neurons=[32],
               num_decoder_layers=1, decoder_neurons=[32],
               use_attention=False
               ):
  # Multiple parallel input and multi-step output using
  # TimeDistributed Dense layer
  model = keras.models.Sequential()

  # Encoder
  for i in range(num_encoder_layers):
    layer = keras.layers.LSTM  # could later add other layers to be chosen
    if i == 0:
      # First layer
      if i + 1 != num_encoder_layers:
        # Multiple layer LSTM
        model.add(layer(encoder_neurons[i],
                        input_shape=(input_length, num_features),
                        return_sequences=True))
      else:
        # Single layer LSTM
        model.add(layer(encoder_neurons[i],
                        input_shape=(input_length, num_features)))
    elif i + 1 == num_encoder_layers:
      # Last layer
      # Attention needs 'return_sequences=True', repeat vector needs 'False'
      model.add(layer(encoder_neurons[i], return_sequences=use_attention))
    else:
      model.add(layer(encoder_neurons[i], return_sequences=True))

  # Attention or Repeat Vector
  if use_attention:
    model.add(CustomAttention(return_sequences=True))
  else:
    model.add(RepeatVector(output_length))

  # Decoder
  for i in range(num_decoder_layers):
    layer = keras.layers.LSTM
    model.add(layer(decoder_neurons[i], return_sequences=True))

  # Time Distributed layer
  model.add(keras.layers.TimeDistributed(
      keras.layers.Dense(num_features, activation='linear')
  ))

  return model


def lstm_attention_model(input_length=24, num_features=1, output_length=1):
  # Similar to 'lstm_model' but builds an additional attention network
  print("Creating attention LSTM")
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
      keras.layers.GRU(20, return_sequences=False),
      keras.layers.RepeatVector(output_length),
      keras.layers.GRU(20, return_sequences=True),
      keras.layers.TimeDistributed(
          keras.layers.Dense(num_features, activation="linear")
      ),
  ])

  return model
