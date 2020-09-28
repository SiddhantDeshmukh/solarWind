import numpy as np
import pandas as pd
import datetime


def decimal_dates_from_dates(dates, hours):
  decimal_years = []
  for date, hour in zip(dates, hours):
    start = datetime.date(date.year, 1, 1).toordinal()
    year_length = datetime.date(date.year+1, 1, 1).toordinal() - start
    decimal_years.append(date.year + float(date.toordinal() - start) / year_length + (hour / 24.))

  return decimal_years


def days_of_year_to_datetimes(years, days_of_year):
  datetimes = []
  for year, day in zip(years, days_of_year):
    # print(year, day)
    datetimes.append(datetime.datetime.strptime(f'{str(year).rjust(4, "0")}-{str(int(day)).rjust(3, "0")}', '%Y-%j'))

  return datetimes


def split_string_date_time_to_datetime(years, months, days, hours=None):
  datetimes = []

  if hours:
    for year, month, day, hour in zip(years, months, days, hours):
      datetimes.append(datetime.datetime.strptime(
        f'{str(year).rjust(4, "0")}-{str(month).rjust(2, "0")}-{str(day).rjust(2, "0")}-{str(hour).rjust(2, "0")}', 
        '%Y-%m-%d-%H'
      ))
  else:
    for year, month, day in zip(years, months, days):
      datetimes.append(datetime.datetime.strptime(
        f'{str(year).rjust(4, "0")}-{str(month).rjust(2, "0")}-{str(day).rjust(2, "0")}', 
        '%Y-%m-%d'
      ))

  return datetimes

def preprocess_omni_df(df):
  # Convert Year, DecimalDay in to DateTime
  df['Date'] = days_of_year_to_datetimes(df['Year'].values, df['DayOfYear'].values)

  # Convert Year, DayOfYear, Hour to DecimalDate
  df['DecimalDate'] = decimal_dates_from_dates(df['Date'], df['Hour'].values)

  return df


def preprocess_and_write(omni_file, header_file, omni_rtn_file=None, headers_rtn_file=None, output_suffix="", cadence="hourly"):
  # Read in solar wind data
  nan_values = [9999999.0, 999999.99, 99999.99, 9999.0, 999.99, 999.9, 999]
  # df = pd.read_fwf(omni_file, header=None, na_values=nan_values)
  df = pd.read_csv(omni_file, header=None, delim_whitespace=True, na_values=nan_values)

  # Read in headers
  headers = []
  with open(header_file, 'r', encoding='utf-8') as infile:
    headers = [line.rstrip("\n") for line in infile]

  # Add headers to data
  df.columns = headers

  # Add 'OMNI_M' data which uses RTN
  if omni_rtn_file:
    df_rtn = pd.read_csv(omni_rtn_file, header=None, delim_whitespace=True, na_values=nan_values)
    headers_rtn = []
    with open(headers_rtn_file, 'r', encoding='utf-8') as infile:
      headers_rtn = [line.rstrip("\n") for line in infile]

    df_rtn.columns = headers_rtn
    df = pd.concat([df, df_rtn])

  # Preprocess DF and add supplementary columns
  df = preprocess_omni_df(df)

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

  # Convert Year, DecimalDay into DateTime
  df['Date'] = split_string_date_time_to_datetime(df['Year'].values, df['Month'].values, df['Day'].values)

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
  preprocess_and_write(omni_hr_file, headers_file, omni_hr_rtn_file, headers_rtn_file, output_suffix="_hourly", cadence="hourly")
  preprocess_and_write(omni_day_file, headers_file, omni_daily_rtn_file, headers_rtn_file, output_suffix="_daily", cadence="daily")
  preprocess_and_write(omni_yr_file, headers_file, output_suffix="_yearly", cadence="yearly")