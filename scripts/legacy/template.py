import pandas as pd
import numpy as np

# Template that loads the data into dictionaries. For each dictionary, the dictionary keys are in the format (without quotes):
# "quantity [units in cgs] (identifier)"
# For Owens' model data, the dictionary keys are thus
# Year (Owens),Radial Wind Velocity [km s^-1] (Owens),Radial Magnetic Field Magnitude [nT] (Owens),Open Flux [Mx] (Owens)
# For OMNI data, change "Owens" to "OMNI Hr" (in this case, you can also load in the "OMNI Daily" data from the other
# OMNI file. "Hr" refers to the 27-day averages being calculated straight from the hourly averages, whereas "Daily"
# refers to them first being binned into 1-day averages and then 27-day averages.

# The keys for the dictionary can be found as the first line of the data file. When getting the data from the dictionary,
# the key does have to match exactly (case-sensitive as well) otherwise it will throw a compiler error.
owens_dict = pd.read_csv('data/models/owens_equatorial.csv').to_dict('list')

# OMNI hourly -> 27-day averaged data
omni_dict = pd.read_csv("data/27_day_avg/OMNI_27day_avg_hr.csv").to_dict("list")

# All other spacecraft data (27-day averaged)
data_dict = pd.read_csv("data/27_day_avg/ALL_SPACECRAFT_27day_avg.csv").to_dict("list")

# Uncomment the following block to see all the keys.# Owens' model data (27-day averaged)

# # All possible Owens dictionary keys
# print(owens_dict.keys())
#
# # All possible OMNI dictionary keys
# print(omni_dict.keys())
#
# # All possible other spacecraft dictionary keys
# print(data_dict.keys())


# An example line of pulling Owens' b_r data and times
owens_br = owens_dict.get("Radial Magnetic Field Magnitude [nT] (Owens)")
owens_time = owens_dict.get("Year (Owens)")


