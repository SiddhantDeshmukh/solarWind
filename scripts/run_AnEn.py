import AnEn_functions as fn
import pandas as pd


fn.plot_analogue_forecast(file_path='../res/aaH_data.txt',
                          t0=pd.Timestamp(2017,12,5, 10, 30),
                          training_window=23,
                          forecast_length=25,
                          weighting_to_recent='quadratic',
                          number_of_analogues=25,
                          block_out=False)
