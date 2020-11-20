#%%
import numpy as np
import heliopy.data.omni as omni
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import astropy.units as u
from datetime import datetime
import rnn
from baseline_metrics import naive_forecast_start, naive_forecast_end, \
  mean_forecast, median_forecast

#%%
def get_omni_rtn_data(start_time: datetime, end_time: datetime):
  identifier = 'OMNI_COHO1HR_MERGED_MAG_PLASMA'  # COHO 1HR data
  omni_data = omni._omni(start_time, end_time, identifier=identifier, intervals='yearly', warn_missing_units=False)

  return omni_data


def remove_nans_from_data(data: np.array, 
    model_inputs: np.array, model_outputs: np.array):
  nan_check = np.array([data[i:i + 25] for i in range(len(data) - 25 + 1)])
  model_inputs = model_inputs[np.where([~np.any(np.isnan(i)) for i in nan_check])]
  model_outputs = model_outputs[np.where([~np.any(np.isnan(i)) for i in nan_check])]

  print(f"Input shape: {model_inputs.shape}")
  print(f"Output shape: {model_outputs.shape}")

  print(f"Any NanNs? {np.any(np.isnan(model_inputs)) or np.any(np.isnan(model_outputs))}")

  return model_inputs, model_outputs


def split_into_24_hour_sections(data: np.array):
  model_inputs = np.array([data[i:i+24] for i in range(len(data) - 24)])[:, :, np.newaxis]
  model_outputs = np.array(data[24:])

  # Check for and remove NaNs
  model_inputs, model_outputs = remove_nans_from_data(data, model_inputs, model_outputs)

  return model_inputs, model_outputs


def split_train_val_test(input_data: np.array, output_data: np.array):
  # Split based on hard-coded indices present in OMNI data
  # Order is train -> validation -> test
  train_idx_end = 134929
  val_size = 44527
  val_idx_end = train_idx_end + val_size

  inputs_train, outputs_train = input_data[:train_idx_end], output_data[:train_idx_end]
  inputs_val, outputs_val = input_data[train_idx_end: val_idx_end], output_data[train_idx_end: val_idx_end]
  inputs_test, outputs_test = input_data[val_idx_end:], output_data[val_idx_end:]

  return inputs_train, inputs_val, inputs_test, outputs_train, outputs_val, outputs_test


# Try out an LSTM
if __name__ == "__main__":
  # In hours for hourly cadence OMNI RTN data
  START_TIME = (datetime(1995, 1, 1))
  END_TIME = (datetime(2018, 2, 28))
  INPUT_LENGTH = 24 

  data = get_omni_rtn_data(START_TIME, END_TIME).to_dataframe()
  
  mag_field_strength, bulk_wind_speed = np.array(data["BR"]), np.array(data["V"])

  # Split into 24-hour sections
  mag_field_input, mag_field_output = split_into_24_hour_sections(mag_field_strength)
  wind_speed_input, wind_speed_output = split_into_24_hour_sections(bulk_wind_speed)

  # Just using B_R data from here on, need to find a better way to use any OMNI dataset
  # Train/test/validation split
  inputs_train, inputs_val, inputs_test,\
    outputs_train, outputs_val, outputs_test = \
    split_train_val_test(mag_field_input, mag_field_output)
  
  # LSTM model
  model = rnn.lstm_model()
  print(model.summary())
  # %%
  # Compile model
  model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

  # Fit model
  model.fit(inputs_train, outputs_train, 
    validation_data=(inputs_val, outputs_val),
    batch_size=2048, epochs=30,
    callbacks=keras.callbacks.EarlyStopping(restore_best_weights=True, patience=30)
    )

# %%
  # Test set evaluation
  model.evaluate(inputs_test, outputs_test)

  naive_start_test = naive_forecast_start(inputs_test)
  naive_end_test = naive_forecast_end(inputs_test)
  mean_test = mean_forecast(inputs_test)
  median_test = median_forecast(inputs_test)

  print(naive_start_test.shape)
  print(naive_end_test.shape)
  print(mean_test.shape)
  print(median_test.shape)
# %%
