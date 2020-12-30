# Run the analogue ensemble for solar cycles 21 - 24
# Imports
# =========================================================================
from datetime import datetime
import analogue_ensemble as ae
import astropy.units as u
from loss_functions import mse
import json
import numpy as np
import pandas as pd


# Solar cycle datetime structs
# =========================================================================
solar_cycles_csv = '../res/solar_cycles.csv'
solar_cycles = pd.read_csv(solar_cycles_csv, index_col=0)

# JSON Encoding
# =========================================================================
def array_to_list(array: np.ndarray):
  # Converts a NumPy array into a list of Python floats
  return list(array.astype(float))

# Data loading
# =========================================================================
# Predictions for cycle 21 - 24; use cycle 20 as extra info to find
# analogues for earlier times
START_TIME = datetime.strptime(solar_cycles.loc[20]['start_min'], '%Y-%m-%d')  # start of cycle 20
FORECAST_START_TIME = datetime.strptime(solar_cycles.loc[21]['start_min'], '%Y-%m-%d')  # start of cycle 21
END_TIME = datetime.strptime(solar_cycles.loc[24]['end'], '%Y-%m-%d')  # end of cycle 24

data = ae.get_omni_rtn_data(START_TIME, END_TIME).to_dataframe()
keys = ['BR', 'V']

data = data[keys]
output_predictions = []

br_single_predictions = []
vr_single_predictions = []

# Analogue ensemble parameters
# =========================================================================
training_window = 24 * (u.hr)
forecast_window = 24 * (u.hr)
full_window_timedelta = ae.time_window_to_time_delta(training_window + forecast_window)
num_analogues = 10

# Running analogue ensemble
# =========================================================================
timestamps = list(data[keys[0]].keys())
df_timestamps = []

for i, timestamp in enumerate(timestamps):
  # Start forecasting at cycle 21
  if timestamp < FORECAST_START_TIME:
    continue

  # Check bounds: Should be at least 'training_window + forecast_window'
  # points on either side to make the ensemble
  if timestamp < (timestamps[0] + full_window_timedelta):
    continue

  if timestamp > (timestamps[-1] - full_window_timedelta):
    continue

  # TESTING
  if timestamp > FORECAST_START_TIME + ae.time_window_to_time_delta(48 * u.hr):
    break
  # END TESTING

  prediction_dict = {}
  prediction_dict['timestamp'] = str(timestamp)

  for key in keys:
    # Run analogue ensemble for 'forecast_time = timestamp'
    analogue_matrix, analogue_prediction, observed_trend =\
      ae.run_analogue_ensemble(data[key], timestamp, training_window,
                              forecast_window, num_analogues)

    # Save prediction and MSE loss
    prediction_dict[key] = array_to_list(analogue_prediction)
    prediction_dict[f"{key}_MSE"] = mse(analogue_prediction, observed_trend)
    
    if key == 'BR':
      br_single_predictions.append(analogue_prediction[0])
    
    if key == 'V':
      vr_single_predictions.append(analogue_prediction[0])

  output_predictions.append(prediction_dict)
  df_timestamps.append(timestamp)

  if i % 100 == 0:
    print(f"Start: {FORECAST_START_TIME}. Now: {timestamp}. End: {END_TIME}.")
    print(f"Done {i} of {len(timestamps)}.")

# Save outputs
# =========================================================================
# JSON of every prediction
with open('../out/analogue_predictions.json', 'w', encoding='utf-8') as outfile:
  json.dump(output_predictions, outfile, indent=2)

# Single prediction csv from DF
output_data = {
  'timestamp': df_timestamps,
  'BR': br_single_predictions,
  'VR': vr_single_predictions
}

df = pd.DataFrame(output_data)
df.set_index('timestamp', inplace=True)

df.to_csv('../out/analogue_single_predictions.csv')
