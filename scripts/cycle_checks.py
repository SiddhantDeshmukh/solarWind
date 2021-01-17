# Let's check NaN values in the solar cycles

# =========================================================================
# Imports
# =========================================================================
import pandas as pd
import data_processing as dp
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

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
# Array utilities
# =========================================================================
def remove_nans(array: np.ndarray) -> np.ndarray:
  return array[~np.isnan(array)]

# =========================================================================
# Evaluate 'health' of each cycle
# =========================================================================
# For each cycle, count the number of NaN values for each key, the length
# of the cycle (number of hours), and start and end times -> tabulate all
cycle_df = pd.DataFrame()
cycles = [21, 22, 23, 24]

output_cols = ['Start', 'End', 'Total Points']

for cycle in cycles:
  cycle_start = datetime_from_cycle(solar_cycles, cycle)
  cycle_end = datetime_from_cycle(solar_cycles, cycle, key='end')
  cycle_data = dp.get_omni_rtn_data(cycle_start, cycle_end).to_dataframe()

  cycle_health = {
    'Cycle': cycle,
    'Start': str(cycle_start.date()),
    'End': str(cycle_end.date())
  }

  keys = ['BR', 'V']
  cycle_health['Total Points'] = len(cycle_data[keys[0]].to_numpy())

  for key in keys:
    array = cycle_data[key].to_numpy()
    nan_vals_key = f'% NaN ({key})'
    cycle_health[nan_vals_key] = (np.count_nonzero(np.isnan(array)) / cycle_health['Total Points']) * 100

    if not nan_vals_key in output_cols:
      output_cols.append(nan_vals_key)

  cycle_df = cycle_df.append(cycle_health, ignore_index=True)

cycle_df.set_index('Cycle', inplace=True)

# Write to file
cycle_df.to_latex('./cycle_checks.tex', columns=output_cols, float_format='%.2f')

# =========================================================================
# Create histograms of each solar cycle
# =========================================================================
keys = ['BR', 'V']
labels = [r"B$_r$ [nT]", r"v$_r$ [km s$^{-1}$]"]

for key, label in zip(keys, labels):
  nrows, ncols = 2, 2
  fig, axes = plt.subplots(nrows, ncols, figsize=(12, 12))
  fig_file = f"../figs/cycle_checks/hist_{key}.png"

  nan_vals_key = f'% NaN ({key})'

  for i in range(nrows):
    for j in range(ncols):
      idx = i * ncols + j
      
      cycle = cycles[idx]
      cycle_start = datetime_from_cycle(solar_cycles, cycle)
      cycle_end = datetime_from_cycle(solar_cycles, cycle, key='end')
      cycle_data = dp.get_omni_rtn_data(cycle_start, cycle_end).to_dataframe()

      data = remove_nans(cycle_data[key])

      axes[i][j].hist(cycle_data[key])
      axes[i][j].set_xlabel(label)
      axes[i][j].set_ylabel("Frequency")
      print(cycle_df.keys())
      axes[i][j].set_title(f"Cycle {cycle}, {cycle_df[cycle][nan_vals_key]}% NaN")

  plt.savefig(fig_file, bbox_inches="tight")


# 2D hist of 'BR' and 'V'
nrows, ncols = 2, 2
fig, axes = plt.subplots(nrows, ncols, figsize=(12, 12))
fig_file = f"../figs/cycle_checks/hist2d_br_v.png"

for i in range(nrows):
  for j in range(ncols):
    idx = i * ncols + j
      
    cycle_start = datetime_from_cycle(solar_cycles, cycles[idx])
    cycle_end = datetime_from_cycle(solar_cycles, cycles[idx], key='end')
    cycle_data = dp.get_omni_rtn_data(cycle_start, cycle_end).to_dataframe()

    br_data = remove_nans(cycle_data['BR'])
    vr_data = remove_nans(cycle_data['V'])

    axes[i][j].hist2d(cycle_data['BR'], cycle_data['V'])
    axes[i][j].set_xlabel(r"B$_r$ [nT]")
    axes[i][j].set_ylabel(r"v$_r$ [km s$^{-1}$]")

plt.savefig(fig_file, bbox_inches="tight")