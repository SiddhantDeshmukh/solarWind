#%%
import numpy as np
import tensorflow.keras as keras
import astropy.units as u
from datetime import datetime
from analogue_ensemble import run_analogue_ensemble
import models
from baseline_metrics import naive_forecast_start, naive_forecast_end, \
  mean_forecast, median_forecast, solar_rotation_forecast
from loss_functions import mse, rmse
import pandas as pd

import data_processing as dp


#%%
# Try out an LSTM
if __name__ == "__main__":
  # In hours for hourly cadence OMNI RTN data
  START_TIME = (datetime(1995, 1, 1))
  END_TIME = (datetime(2018, 2, 28))
  INPUT_LENGTH = 24 

  data = dp.get_omni_rtn_data(START_TIME, END_TIME).to_dataframe()

  initial_time_stamp = data.index[0]
  
  mag_field_strength, bulk_wind_speed = np.array(data["BR"]), np.array(data["V"])

  # Split into 24-hour sections
  mag_field_input, mag_field_output = dp.split_into_24_hour_sections(mag_field_strength)
  wind_speed_input, wind_speed_output = dp.split_into_24_hour_sections(bulk_wind_speed)

  # Just using B_R data from here on, need to find a better way to use any OMNI dataset
  # Train/test/validation split
  inputs_train, inputs_val, inputs_test,\
    outputs_train, outputs_val, outputs_test,\
    validation_timestamp, test_timestamp, end_timestamp = \
    dp.split_train_val_test(mag_field_input, mag_field_output,
    start_timestamp=initial_time_stamp)
  
  # LSTM model
  model = models.lstm_model()
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

  # Simple baselines
  naive_start_test = naive_forecast_start(inputs_test)
  naive_end_test = naive_forecast_end(inputs_test)
  mean_test = mean_forecast(inputs_test)
  median_test = median_forecast(inputs_test)
  solar_rotation_test = solar_rotation_forecast(inputs_test)

  # %%
  # Analogue ensemble baseline (run an AnEn for every point, excluding
  # 24 points on each edge)
  # Define properties for forecast
  # Choose forecast time within start/end time randomly
  # Initialise inputs to AnEn
  data_start_time = test_timestamp + pd.Timedelta(1, unit='hr')
  forecast_time = data_start_time + pd.Timedelta(24, unit='hr')
  analogue_inputs = pd.Series(dp.add_timestamps_to_data(inputs_train.flatten()[::INPUT_LENGTH], initial_time_stamp))

  training_window = 24 * (u.hr)  # 24 hours before 'forecast_time' 
  forecast_window = 24 * (u.hr)  # 24 hours after 'forecast_time'
  num_analogues = 10  # Number of analogues to find for ensemble

  # print("Creating dataset...")

  # # Flatten LSTM data to 1D, taking every INPUT_LENGTH point to create a
  # # concurrent sequence, and add timestamps
  # test_inputs = dp.add_timestamps_to_data(inputs_test.flatten()[::INPUT_LENGTH], test_timestamp)
  # analogue_inputs = analogue_inputs.append(pd.Series(test_inputs))

  # # Save to file
  # analogue_inputs.to_csv('../res/br_timeseries.csv')

  # Load BR data as series
  analogue_inputs = pd.read_csv('../res/br_timeseries.csv')

  # Run the analogue ensemble for each point in the test set
  print("Running analogue ensemble...")
  analogue_predictions = pd.Series()

  # %%
  for i in range(len(inputs_test)):
    if i >= len(inputs_test) - 2:
      analogue_prediction = inputs_test[i]
      
    else:
      data_start_time = test_timestamp + pd.Timedelta(i, unit='hr')
      forecast_time = data_start_time + pd.Timedelta(24, unit='hr')

      analogue_matrix, analogue_prediction, observed_trend = \
        run_analogue_ensemble(analogue_inputs, forecast_time, training_window,
        forecast_window, num_analogues)
        
      if (i % ((len(inputs_test) - 2) // 100)) == 0:
        print(f"Done {i} of {len(inputs_test - 2)}")
        
    analogue_predictions = analogue_predictions.append(pd.Series(analogue_prediction))

  # analogue_predictions = np.array(analogue_predictions)

  # Save to file
  analogue_predictions.to_csv('../out/analogue_br_prediction.csv')

  # analogue_predictions = pd.read_csv('../res/br_timeseries.csv')

  # %%
  # Check MSE and RMSE of each baseline
  mse_naive_start, rmse_naive_start = mse(naive_start_test, outputs_test), rmse(naive_start_test, outputs_test)
  mse_naive_end, rmse_naive_end = mse(naive_end_test, outputs_test), rmse(naive_end_test, outputs_test)
  mse_mean, rmse_mean = mse(mean_test, outputs_test), rmse(mean_test, outputs_test)
  mse_median, rmse_median = mse(median_test, outputs_test), rmse(median_test, outputs_test)
  mse_AnEn, rmse_AnEn = mse(analogue_predictions, outputs_test), rmse(analogue_predictions, outputs_test)


  def print_metrics(baseline: str, mse_value: float, rmse_value: float) -> None:
    print(f"{baseline}: MSE = {mse_value:.3f} \t RMSE = {rmse_value:.3f}")

  # Compare baseline metrics to test set evaluation
  baselines = ["Naive start", "Naive end", "Mean", "Median", "Analogue Ensemble"]
  mse_metrics = [mse_naive_start, mse_naive_end, mse_mean, mse_median, mse_AnEn]
  rmse_metrics = [rmse_naive_start, rmse_naive_end, rmse_mean, rmse_median, rmse_AnEn]

  for baseline, mse_metric, rmse_metric in zip(baselines, mse_metrics, rmse_metrics):
    print_metrics(baseline, mse_metric, rmse_metric)
  