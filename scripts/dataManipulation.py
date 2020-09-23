import numpy as np
import pandas as pd
import datetime


def convert_to_fltyr(yr, decday, hour):
  fltyr_list = []

  for index, element in enumerate(yr):
    # if str(yr[index]).startswith('9'):
    #     if len(str(int(yr[index]))) > 1:
    #         yr[index] = float('19' + str(yr[index]))
    #     else:
    #           yr[index] = 2009
    # else:
    #   yr[index] = float('200' + str(yr[index]))

    fltyr = yr[index] + decday[index] / 365.25
    fltyr_list.append(fltyr)

  return fltyr_list


def convert_to_fltyr(years, decdays, hours):
  year_flts = []
  
  for year, decday, hour in zip(years, decdays, hours):
    year_flt = year + (decday + (hour / 24)) / 365.25
    year_flts.append(f"{year_flt:.4f}")

  return year_flts


def convert_to_datetimes(years, decimalDays, cadence="hourly"):
  datetimes = []
  lastDay = 1
  for year, day in zip(years, decimalDays):
    if np.isnan(day):
      if cadence == "daily":
        day = lastDay + 1
      else:
        day = lastDay
    else:
      lastDay = day

    datetimes.append(datetime.datetime(year, 1, 1) + datetime.timedelta(int(day) - 1))

  return datetimes


def preprocess_omni_df(df, cadence):
  # Convert Year and DecimalDay into YearFlt
  df['YearFlt'] = convert_to_fltyr(df['Year'].values, df['DecimalDay'].values, df['Hour'].values)
  df = df.astype({'YearFlt': float})

  # Convert Year, DecimalDay, Hour into DateTime
  df['DateTime'] = convert_to_datetimes(df['Year'].values, df['DecimalDay'].values, cadence)

  return df


def preprocess_and_write(omni_file, header_file, omni_rtn_file=None, headers_rtn_file=None, output_suffix="", cadence="hourly"):
  # Read in solar wind data
  nan_values = [9999999.0, 999999.99, 99999.99, 9999.0, 999.99, 999.9, 999, 99.99, 99.9, 99, 9.999]
  df = pd.read_fwf(omni_file, header=None, na_values=nan_values)

  # Read in headers
  headers = []
  with open(header_file, 'r', encoding='utf-8') as infile:
    headers = [line.rstrip("\n") for line in infile]

  # Add headers to data
  df.columns = headers

  # Add 'OMNI_M' data which uses RTN
  if omni_rtn_file:
    df_rtn = pd.read_fwf(omni_rtn_file, header=None, na_values=nan_values)
    headers_rtn = []
    with open(headers_rtn_file, 'r', encoding='utf-8') as infile:
      headers_rtn = [line.rstrip("\n") for line in infile]

    df_rtn.columns = headers_rtn
    df = pd.concat([df, df_rtn])

  # Preprocess DF and add supplementary columns
  df = preprocess_omni_df(df, cadence)

  # Write out human-readable and machine-readable files
  output_pickle = f'../res/OMNI/omni{output_suffix}.pkl'
  output_csv = f'../res/OMNI/omni{output_suffix}.csv'

  df.to_pickle(output_pickle)
  df.to_csv(output_csv, index=False)


def write_ssn_file():
  col_specs = (
    (0, 4), (6, 7), (9, 10),
    (12, 19), (21, 24), (26, 30),
    (34, 35)
  )

  df = pd.read_fwf('../res/ssn/silso_daily.txt', header=None, nan_values=[-1])
  
  headers = []
  with open('../res/ssn/silso_headers.txt', 'r', encoding='utf-8') as infile:
    headers = [line.rstrip("\n") for line in infile]

  df.columns = headers

  # Convert Year, DecimalDay, Hour into DateTime
  datetimes = []
  for year, month, day in zip(df['Year'], df['Month'], df['Day']):
    datetimes.append(datetime.datetime.strptime(f'{year}-{month}-{day}', '%Y-%m-%d').date())

  df['DateTime'] = datetimes

  df.to_pickle('../res/ssn/silso_daily.pkl')
  df.to_csv('../res/ssn/silso_daily.csv', index=False)


if __name__ == "__main__":
  omni_hr_file = '../res/OMNI/omni_hourly.dat'
  omni_day_file = '../res/OMNI/omni_daily.dat'
  omni_yr_file = '../res/OMNI/omni_yearly.dat'
  headers_file = '../res/OMNI/keys.csv'

  omni_hr_rtn_file = '../res/OMNI/omni_m_all_years.dat'
  omni_daily_rtn_file = '../res/OMNI/omni_m_daily.dat'
  headers_rtn_file = '../res/OMNI/keys_m.csv'

  write_ssn_file()
  # preprocess_and_write(omni_hr_file, headers_file, omni_hr_rtn_file, headers_rtn_file, output_suffix="_hourly", cadence="hourly")
  # preprocess_and_write(omni_day_file, headers_file, omni_daily_rtn_file, headers_rtn_file, output_suffix="_daily", cadence="daily")
  # preprocess_and_write(omni_yr_file, headers_file, output_suffix="_yearly", cadence="yearly")