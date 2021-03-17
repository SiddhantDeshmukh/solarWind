# %%
import tensorflow.keras as keras
from datetime import datetime
import models
from baseline.baseline_metrics import naive_forecast_start, naive_forecast_end, \
    mean_forecast, median_forecast, solar_rotation_forecast
from loss_functions import mse, rmse
import numpy as np

import data_processing as dp


# %%
if __name__ == "__main__":
  # Input data with maximum non-NaN range
  START_TIME = datetime(1995, 1, 1)
  END_TIME = datetime(2020, 12, 31)

  INPUT_LENGTH = 24
  OUTPUT_LENGTH = 24

  data, keys = dp.omni_cycle_preprocess(START_TIME, END_TIME,
                                        # auto get ["N", "V", "ABS_B", "HMF_INC"]
                                        get_geoeffectiveness=True,
                                        normalise=True)

  for key in data.keys():
    # Check dimensionality and remove geoeffectiveness from each
    # last index is G by default
    data[key] = data[key][:, :, :keys.index("G")]
    print(f"{key} shape: {data[key].shape}")
    print(f"{key} (Min, Max): ({np.min(data[key])}, {np.max(data[key])})")

# %%
  # LSTM model
  parallel_lstm = models.lstm_model(
      input_length=INPUT_LENGTH, num_features=4, output_length=OUTPUT_LENGTH)
  print(parallel_lstm.summary())

  attention_lstm = models.lstm_attention_model(
      input_length=INPUT_LENGTH, num_features=4, output_length=OUTPUT_LENGTH)
  print(attention_lstm.summary())

  # %%
  # Optimizer
  optimizer = keras.optimizers.Adam(lr=1e-3)

  # Compile parallel LSTM model
  parallel_lstm.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

  # Compile attention LSTM model
  attention_lstm.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

  # Fit parallel LSTM model model
  parallel_lstm.fit(data['train_in'], data['train_out'],
                    validation_data=(data['val_in'], data['val_out']),
                    batch_size=32, epochs=30,
                    callbacks=keras.callbacks.EarlyStopping(
                        restore_best_weights=True, patience=10),
                    verbose=2
                    )

  # Fit attention LSTM model
  attention_lstm.fit(data['train_in'], data['train_out'],
                     validation_data=(data['val_in'], data['val_out']),
                     batch_size=32, epochs=30,
                     callbacks=keras.callbacks.EarlyStopping(
                         restore_best_weights=True, patience=10),
                     verbose=2
                     )

  # %%
  # Test set evaluation
  parallel_lstm.evaluate(data['test_in'], data['test_out'])

  # Simple baselines
  naive_start_test = naive_forecast_start(data['test_in'])
  naive_end_test = naive_forecast_end(data['test_in'])
  mean_test = mean_forecast(data['test_in'])
  median_test = median_forecast(data['test_in'])
  solar_rotation_test = solar_rotation_forecast(data['test_in'])

  # %%
  # Check MSE and RMSE of each baseline
  mse_naive_start, rmse_naive_start = mse(
      naive_start_test, data['test_out']), rmse(naive_start_test, data['test_out'])
  mse_naive_end, rmse_naive_end = mse(
      naive_end_test, data['test_out']), rmse(naive_end_test, data['test_out'])
  mse_mean, rmse_mean = mse(
      mean_test, data['test_out']), rmse(mean_test, data['test_out'])
  mse_median, rmse_median = mse(
      median_test, data['test_out']), rmse(median_test, data['test_out'])
  # mse_AnEn, rmse_AnEn = mse(analogue_predictions, outputs_s_termse(analogue_predictions, data['test_out'])

  def print_metrics(baseline: str, mse_value: float, rmse_value: float) -> None:
    print(f"{baseline}: MSE = {mse_value:.3f} \t RMSE = {rmse_value:.3f}")

  # Compare baseline metrics to test set evaluation
  baselines = ["Naive start", "Naive end", "Mean", "Median"]
  mse_metrics = [mse_naive_start, mse_naive_end, mse_mean, mse_median]
  rmse_metrics = [rmse_naive_start, rmse_naive_end, rmse_mean, rmse_median]

  for baseline, mse_metric, rmse_metric in zip(baselines, mse_metrics, rmse_metrics):
    print_metrics(baseline, mse_metric, rmse_metric)
