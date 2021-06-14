# Run the analogue ensemble for solar cycles 21 - 24
# Imports
# =========================================================================
from datetime import datetime
import data_processing as dapr
import analogue_ensemble as ae
import astropy.units as u
from loss_functions import mse
import json
import numpy as np
import pandas as pd


# =========================================================================
# Solar cycle loading
# =========================================================================
solar_cycles_csv = '../res/solar_cycles.csv'
solar_cycles = pd.read_csv(solar_cycles_csv, index_col=0)

# =========================================================================
# Datetime utility functions
# =========================================================================


def datetime_from_cycle(solar_cycles: pd.DataFrame, cycle: int,
                        key='start_min', fmt='%Y-%m-%d'):
  # From the 'solar_cycles' DataFrame, get the 'key' datetime (formatted as
  # %Y-%m-%d by default in the csv) as a datetime
  return datetime.strptime(solar_cycles.loc[cycle][key], fmt)

# =========================================================================
# JSON Encoding
# =========================================================================


def array_to_list(array: np.ndarray):
  # Converts a NumPy array into a list of Python floats
  return list(array.astype(float))


output_predictions = []

# =========================================================================
# Analogue ensemble parameters
# =========================================================================
training_window = 24 * (u.hr)
forecast_window = 24 * (u.hr)
full_window_timedelta = dapr.time_window_to_time_delta(
    training_window + forecast_window)
num_analogues = 10

# =========================================================================
# Running analogue ensemble
# =========================================================================
# Loop over relevant cycles
cycle_nums = [21, 22, 23, 24]
for cycle in cycle_nums:
  print(f"Starting cycle {cycle} predictions...")
  # Data loading
  # Use the previous cycle to find analogues
  search_cycle = cycle - 1
  start_time = datetime_from_cycle(solar_cycles, search_cycle)
  end_time = datetime_from_cycle(solar_cycles, cycle, key='end')
  data = dapr.get_omni_rtn_data(start_time, end_time).to_dataframe()

  # Get the data for just the current cycle to make predictions on it
  cycle_start = datetime_from_cycle(solar_cycles, cycle)
  cycle_data = dapr.get_omni_rtn_data(cycle_start, end_time).to_dataframe()

  # Setup for analogue predictions
  keys = ['BR', 'V']
  timestamps = list(cycle_data[keys[0]].keys())
  br_single_predictions = []
  vr_single_predictions = []

  df_timestamps = []  # for shorter length runs, i.e. breaking out of loop

  # Make a forecast for each timestamp in this solar cycle
  for i, timestamp in enumerate(timestamps):
    # Check bounds - ensure there ar enough points to make a prediction
    if (timestamp < start_time + full_window_timedelta) or \
       (timestamp > end_time - full_window_timedelta):
      continue

    prediction_dict = {}
    prediction_dict['timestamp'] = str(timestamp)

    for key in keys:
      # Run analogue ensemble for 'forecast_time = timestamp'
      analogue_matrix, analogue_prediction, observed_trend =\
          ae.run_analogue_ensemble(data[key], timestamp, training_window,
                                   forecast_window, num_analogues)

      # Save prediction and MSE loss
      prediction_dict[key] = array_to_list(analogue_prediction)
      prediction_dict[f"{key}_MSE"] = mse(
          analogue_prediction, observed_trend)

      if key == 'BR':
        br_single_predictions.append(analogue_prediction[0])

      if key == 'V':
        vr_single_predictions.append(analogue_prediction[0])

    df_timestamps.append(str(timestamp))
    output_predictions.append(prediction_dict)

    if i % (len(timestamps) // 1000) == 0:
      print(f"Cycle {cycle}: Done {i} of {len(timestamps)}.")
      print(f"Cycle start: {cycle_start}")
      print(f"Current timestamp: {timestamp}")
      print(f"Cycle end: {end_time}")

  # Save cycle data
  # JSON of every prediction
  cycle_json = f'../out/an_en_outputs/analogue_predictions_cycle-{cycle}.json'
  cycle_csv = f'../out/an_en_outputs/analogue_single_predictions_cycle-{cycle}.csv'

  print(f"Writing cycle {cycle} predictions.")

  with open(cycle_json, 'w', encoding='utf-8') as outfile:
    json.dump(output_predictions, outfile, indent=2)

  # Single prediction csv from DF
  output_data = {
      'timestamp': df_timestamps,
      'BR': br_single_predictions,
      'VR': vr_single_predictions
  }

  df = pd.DataFrame(output_data)
  df.set_index('timestamp', inplace=True)

  df.to_csv(cycle_csv)

  print(f"Done with cycle {cycle}!")
