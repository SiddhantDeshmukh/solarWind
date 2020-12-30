# Let's check NaN values in the solar cycles

# =========================================================================
# Imports
# =========================================================================
import pandas as pd
import data_processing as dp
from datetime import datetime
import numpy as np

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
# Evaluate 'health' of each cycle
# =========================================================================
# For each cycle, count the number of NaN values for each key, the length
# of the cycle (number of hours), and start and end times -> tabulate all
cycle_df = pd.DataFrame()
cycles = [21, 22, 23, 24]

for cycle in cycles:
  cycle_start = datetime_from_cycle(solar_cycles, cycle)
  cycle_end = datetime_from_cycle(solar_cycles, cycle, key='end')
  cycle_data = dp.get_omni_rtn_data(cycle_start, cycle_end).to_dataframe()

  cycle_health = {
    'cycle': cycle,
    'start': str(cycle_start),
    'end': str(cycle_end)
  }

  keys = ['BR', 'V']
  cycle_health['# total'] = len(cycle_data[keys[0]].to_numpy())

  for key in keys:
    array = cycle_data[key].to_numpy()
    nan_vals_key = f'# NaN ({key})'
    clean_vals_key = f'# clean ({key})'

    cycle_health[nan_vals_key] = np.count_nonzero(np.isnan(array))
    cycle_health[clean_vals_key] = cycle_health['# total'] - cycle_health[nan_vals_key]

  cycle_df = cycle_df.append(cycle_health, ignore_index=True)

cycle_df.set_index('cycle', inplace=True)
print(cycle_df)

# Write to file
cycle_df.to_latex('./cycle_checks.tex')

