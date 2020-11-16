import numpy as np
import pandas as pd
import heliopy.data.omni as omni
import matplotlib.pyplot as plt
import tensorflow as tf
import astropy.units as u
import datetime



def get_omni_rtn_data(start_time, end_time):
  identifier = 'OMNI_COHO1HR_MERGED_MAG_PLASMA'  # COHO 1HR data
  omni_data = omni._omni(start_time, end_time, identifier=identifier, intervals='yearly', warn_missing_units=False)

  return omni_data


# Try out an LSTM
if __name__ == "__main__":
  start_time = (datetime(2017, 1, 1))
  end_time = (datetime(2018, 2, 28))

  omni_data = get_omni_rtn_data(start_time, end_time)
  
    
