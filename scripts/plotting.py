import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Constants
hr_in_day = 24
day_in_month = 30
day_in_year = 365.25
hr_in_month = hr_in_day * day_in_month
hr_in_year = hr_in_day * day_in_year


def get_omni_df(output_suffix=""):
  df = pd.read_pickle(f'../res/OMNI/omni{output_suffix}.pkl')
  df = df.set_index("DateTime")
  df.sort_index(ascending=True, inplace=True)

  return df


def plot_key_against_time(ax, df, key, error_key=None, year_range=None, color='k', ls='-', averaging=None):
  return_fig = False

  if ax == None:
    return_fig = True
    fig, ax = plt.subplots()

  # Copy DF for plotting so it is not changed outside of this scope
  if year_range:  # Tuple of [start, end]
    plot_df = df[df['YearFlt'].between(year_range[0], year_range[1])].copy()
  else:
    plot_df = df.copy()

  if averaging:
    plot_df = plot_df.resample(averaging).mean()

  if error_key:
    ax.errorbar(plot_df.index, plot_df[key], yerr=plot_df[error_key].values, color=color, ls=ls)
  else:
    ax.plot(plot_df.index, plot_df[key], color=color, ls=ls)

  if return_fig:
    return fig, ax


def plot_key_histogram(ax, df, key, bins=None, year_range=None, averaging=None):
  return_fig = False

  if ax == None:
    return_fig = True
    fig, ax = plt.subplots()

  # Copy DF for plotting so it is not changed outside of this scope
  if year_range:  # Tuple of [start, end]
    plot_df = df[df['YearFlt'].between(year_range[0], year_range[1])].copy()
  else:
    plot_df = df.copy()

  if averaging:
    plot_df = plot_df.resample(averaging).mean()

  ax.hist(plot_df[key], bins=bins, density=True)
  ax.set_xscale('log')

  if return_fig:
    return fig, ax


if __name__ == "__main__":
  df = get_omni_df("_hourly")

  # fig, ax = plot_key_against_time(None, df, 'Avg_B', year_range=(2010, 2020), averaging=None)

  # fig, ax = plot_key_against_time(None, df, 'Bz_GSE', year_range=(2010, 2020), averaging='2D')
  # plot_key_against_time(ax, df, 'Bx_GSE', year_range=(2010, 2020), color='b', averaging='2D')
  # plot_key_against_time(ax, df, 'By_GSE', year_range=(2010, 2020), color='r', averaging='2D')

  # plt.savefig('../figs/test.svg', bbox_inches="tight")

  # Histograms
  fig, axes = plt.subplots(2, 2)

  # Magnetic field
  magnetic_field_bins = np.arange(np.min(df['Avg_B_RTN']), np.max(df['Avg_B_RTN']) + 0.5, 0.5)  # 0.5 nT bin size
  plot_key_histogram(axes[0][0], df, 'Avg_B_RTN', bins=magnetic_field_bins)

  # Velocity
  velocity_bins = np.arange(np.min(df['PlasmaFlowSpeed']), np.max(df['PlasmaFlowSpeed']) + 10, 10)  # 10 km/s bin size
  plot_key_histogram(axes[0][1], df, 'PlasmaFlowSpeed', bins=velocity_bins)

  # Density
  density_bins = np.arange(np.min(df['ProtonDensity']), np.max(df['ProtonDensity']) + 1, 1)  # 1 cm^-3 nT bin size
  plot_key_histogram(axes[1][0], df, 'ProtonDensity', bins=density_bins)

  # Temperature
  temperature_bins = np.arange(np.min(df['ProtonTemp']), np.max(df['ProtonTemp']) + 10000, 10000)  # 10,000 K bin size
  plot_key_histogram(axes[1][1], df, 'ProtonTemp', bins=temperature_bins)
  
  plt.savefig('../figs/test_hist.svg', bbox_inches="tight")