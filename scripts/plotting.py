from re import error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import utilities as util
import dataframeUtilities as dfut


# Constants
hr_in_day = 24
day_in_month = 30
day_in_year = 365.25
hr_in_month = hr_in_day * day_in_month
hr_in_year = hr_in_day * day_in_year


def plot_correlations(ax, df, keys, year_range=None, cmap='jet', add_text=True):
  fig = None
  if ax == None:
    fig, ax = plt.subplots()

  plot_df = dfut.truncate_year_range(df, year_range) if year_range else df.copy()
  print(plot_df)
  correlation = dfut.get_correlation(plot_df, columns=keys)
  print(correlation)

  # Plot heatmap
  ax.imshow(correlation, cmap=cmap)
  
  # Axes ticks
  ticks = np.arange(len(keys))
  ax.set_xticks(ticks)
  ax.set_yticks(ticks)
  
  # Axes tick labels
  ax.set_xticklabels(keys)
  ax.set_yticklabels(keys)

  # Rotate the tick labels and set their alignment.
  plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
          rotation_mode="anchor")

  # Add text annotations of correlation coefficients to heatmap
  if add_text:
    for i in range(len(keys)):
        for j in range(len(keys)):
            ax.text(j, i, correlation[i, j], ha="center", va="center", color="w")

  return fig, ax  # 'fig' will be None if 'ax' was provided
  

def plot_key(ax, df, x_key, y_key, x_error_key=None, y_error_key=None, color='k', ls='-',
             xlabel=None, ylabel=None, xscale="linear", yscale="linear"):
  fig = None
  if ax == None:
    fig, ax = plt.subplots()

  # Copy DF for plotting so it is not changed outside of this scope
  plot_df = df.copy()

  # Plot data
  if x_error_key or y_error_key:
    ax.errorbar(plot_df[x_key], plot_df[y_key], xerr=plot_df[x_error_key].values, yerr=plot_df[y_error_key].values,
       color=color, ls=ls)
  else:
    ax.plot(plot_df[x_key], plot_df[y_key], color=color, ls=ls)

  # Labels
  if xlabel:
    ax.set_xlabel(xlabel)
  
  if ylabel:
    ax.set_ylabel(ylabel)

  # Set scales
  ax.set_xscale(xscale)
  ax.set_yscale(yscale)

  return fig, ax  # 'fig' will be None if 'ax' was provided


def plot_key_against_time(ax, df, key, error_key=None, year_range=None, color='k', ls='-', averaging=None,
                          xlabel=None, ylabel=None, xscale="linear", yscale="linear"):  
  plot_df = dfut.truncate_year_range(df, year_range) if year_range else df.copy()

  # Average data by resampling DataFrame
  if averaging:
    plot_df = plot_df.resample(averaging).mean()

  fig, ax = plot_key(ax, plot_df, x_key='YearFlt', y_key=key, x_error_key=None, y_error_key=error_key, 
                color=color, ls=ls, xlabel=xlabel, ylabel=ylabel, xscale=xscale, yscale=yscale)

  if fig:
    return fig, ax


def plot_key_histogram(ax, df, key, color='gray', bins=None, year_range=None, averaging=None, 
                       xlabel=None, ylabel=None, xscale="linear", yscale="linear"):
  fig = None
  if ax == None:
    fig, ax = plt.subplots()


  # Copy DF for plotting so it is not changed outside of this scope
  if year_range:  # Tuple of [start, end]
    plot_df = df[df['YearFlt'].between(year_range[0], year_range[1])].copy()
  else:
    plot_df = df.copy()

  if averaging:
    plot_df = plot_df.resample(averaging).mean()

  mean, median, std_dev = np.mean(plot_df[key]), np.nanmedian(plot_df[key]), np.std(plot_df[key])

  ax.hist(plot_df[key], bins=bins, density=True, color=color)

  # Vertical lines for mean, median
  ax.axvline(mean, c='r', ls='--')
  ax.axvline(median, c='b', ls='--')

  # Set scales
  ax.set_xscale(xscale)
  ax.set_yscale(yscale)

  if xlabel:
    ax.set_xlabel(xlabel)

  if ylabel:
    ax.set_ylabel(ylabel)

  return fig, ax  # 'fig' will be None if 'ax' was provided

if __name__ == "__main__":
  df = dfut.concat_dfs(
    dfut.get_df('../res/OMNI/omni_hourly.pkl', index="DateTime"), 
    dfut.get_df('../res/ssn/silso_daily.pkl', index="DateTime")
  )

  # fig, ax = plot_key_against_time(None, df, 'Avg_B', year_range=(2010, 2020), averaging=None)

  # fig, ax = plot_key_against_time(None, df, 'Bz_GSE', year_range=(2010, 2020), averaging='2D')
  # plot_key_against_time(ax, df, 'Bx_GSE', year_range=(2010, 2020), color='b', averaging='2D')
  # plot_key_against_time(ax, df, 'By_GSE', year_range=(2010, 2020), color='r', averaging='2D')

  # plt.savefig('../figs/test.svg', bbox_inches="tight")

  # # Histograms
  # fig, axes = plt.subplots(3, 2, figsize=(12, 16))

  # # Magnetic field
  # magnetic_field_bins = np.arange(np.min(df['Avg_B_RTN']), np.max(df['Avg_B_RTN']) + 0.5, 0.5)  # 0.5 nT bin size
  # plot_key_histogram(axes[0][0], df, 'Avg_B_RTN', bins=magnetic_field_bins, year_range=(1963.9, 2017),
  #  xlabel=r"Avg $B_{RTN}$ [nT]", ylabel="Frequency", xscale="log")

  # # Velocity
  # velocity_bins = np.arange(np.min(df['PlasmaFlowSpeed']), np.max(df['PlasmaFlowSpeed']) + 10, 10)  # 10 km/s bin size
  # plot_key_histogram(axes[0][1], df, 'PlasmaFlowSpeed', bins=velocity_bins, year_range=(1963.9, 2017), 
  #   xlabel=r"Velocity [km s$^{-1}$]", ylabel="Frequency", xscale="log")

  # # Density
  # density_bins = np.arange(np.min(df['ProtonDensity']), np.max(df['ProtonDensity']) + 1, 1)  # 1 cm^-3 bin size
  # plot_key_histogram(axes[1][0], df, 'ProtonDensity', bins=density_bins,  year_range=(1963.9, 2017),
  #   xlabel=r" Proton Density [cm$^{-3}$]", ylabel="Frequency", xscale="log")

  # # Temperature
  # temperature_bins = np.arange(np.min(df['Temperature']), np.max(df['Temperature']) + 10000, 10000)  # 10,000 K bin size
  # plot_key_histogram(axes[1][1], df, 'Temperature', bins=temperature_bins,  year_range=(1965.5, 2017),
  #   xlabel="Proton Temperature [K]", ylabel="Frequency", xscale="log")

  # # Sunspot Number
  # ssn_bins = np.arange(np.min(df['SSN']), np.max(df['SSN']) + 10, 10)
  # plot_key_histogram(axes[2][0], df, 'SSN', bins=ssn_bins, year_range=(1963.9, 2017), 
  #   xlabel="Sunspot Number", ylabel="Frequency")

  # # Sunspot Number vs Time
  # plot_key_against_time(axes[2][1], df, 'SSN', year_range=(1963.9, 2017), xlabel="Date", ylabel="Sunspot Number")
  
  # plt.savefig('../figs/test_hist.svg', bbox_inches="tight")

  # SSN correlations
  fig, axes = plt.subplots(3, 2, figsize=(12, 16))

  # Magnetic field
  plot_correlations(axes[0][0], df, ['Avg_B_RTN', 'SSN'], year_range=(1963.9, 2017))

  # Velocity
  plot_correlations(axes[0][1], df, ['PlasmaFlowSpeed', 'SSN'], year_range=(1963.9, 2017))

  # Density
  plot_correlations(axes[1][0], df, ['ProtonDensity', 'SSN'], year_range=(1963.9, 2017))

  # Temperature
  plot_correlations(axes[1][1], df, ['Temperature', 'SSN'], year_range=(1963.9, 2017))

  # All parameters
  plot_correlations(axes[2][1], df, ['Avg_B_RTN', 'PlasmaFlowSpeed', 'ProtonDensity', 'Temperature', 'SSN'])

  plt.savefig('../figs/test_corr.svg', bbox_inches='tight')