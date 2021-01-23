import tensorflow.keras as keras
from typing import List
import torch
import torch.nn as nn


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
        model.add(layer(lstm_layer_neurons[i], input_shape=(None, 11), return_sequences=True))
      else:
        # Single layer LSTM
        model.add(layer(lstm_layer_neurons[i], input_shape=(None, 11)))
    elif i + 1 == num_lstm_layers:
      # Last layer
      model.add(layer(lstm_layer_neurons[i]))
    else:
      model.add(layer(lstm_layer_neurons[i], input_shape=(None, 11), return_sequences=True))

  # Add Dense layers
  for i in range(num_dense_layers):
    layer = keras.layers.Dense
    if i + 1 == num_dense_layers:
      model.add(layer(dense_layer_neurons[i], activation='linear'))
    else:
      model.add(layer(dense_layer_neurons[i], activation='relu'))
  
  return model

def pytorch_lstm_model(input_size: int,
                      num_lstm_layers: int,
                      lstm_layer_neurons: List[int],
                      num_dense_layers: int,
                      dense_layer_neurons: List[int]) -> nn.Sequential:
  layers = []

  # Add LSTM layers
  for i in range(num_lstm_layers):
    if i == 0:  # first layer
      layers.append(nn.LSTMCell(input_size, lstm_layer_neurons[i]))

    else:
      layers.append(nn.LSTMCell(lstm_layer_neurons[i - 1], lstm_layer_neurons[i]))
    
  # Add Dense layers
  for i in range(num_dense_layers):
    if i + 1 == num_dense_layers:  # last layer
      layers.append(nn.Linear(lstm_layer_neurons[-1], 1))
    else:
      layers.append(nn.ReLU())

  return nn.Sequential(layers)