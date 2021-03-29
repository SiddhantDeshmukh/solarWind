# %%
import matplotlib.pyplot as plt
from data_processing import get_omni_rtn_data, standardise_array
from datetime import datetime

# Check standardisation
# Input data with maximum non-NaN range
START_TIME = datetime(1995, 1, 1)
END_TIME = datetime(2020, 12, 31)

data = get_omni_rtn_data(START_TIME, END_TIME).to_dataframe()
keys = ["N", "V", "ABS_B"]

fig, axes = plt.subplots(2, 3, figsize=(10, 10))
for i, key in enumerate(keys):
  # Normal
  axes[0][i].hist(data[key])
  axes[0][i].set_xlabel(f"{key}, Unstandardised")
  axes[0][i].set_ylabel("Counts")

  # Standardised
  std_data = standardise_array(data[key].values, data[key].values)
  axes[1][i].hist(std_data)
  axes[1][i].set_xlabel(f"{key}, Standardised")
  axes[1][i].set_ylabel("Counts")

# %%
