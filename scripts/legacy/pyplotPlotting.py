import numpy as np
import pandas as pd
from tools import utilities as util
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tools.stellarUtilities as stut
import scipy.optimize as opt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib


# Declare files and read in 27-day averaged data as dictionaries
ACE_file = './data/27_day_avg/ACE_27day_avg.csv'
DSCOVR_file = './data/27_day_avg/DSCOVR_27day_avg.csv'
HELIOS1_file = './data/27_day_avg/HELIOS_1_27day_avg.csv'
HELIOS2_file = './data/27_day_avg/HELIOS_2_27day_avg.csv'
STEREO_A_file = './data/27_day_avg/STEREO_A_27day_avg.csv'
STEREO_B_file = './data/27_day_avg/STEREO_B_27day_avg.csv'
sunspot_file = './data/sunspot/dly_sunspot_27day_avg.csv'
ULYSSES_file = './data/27_day_avg/ULYSSES_27day_avg.csv'
ULYSSES_FULL_file = './data/27_day_avg/ULYSSES_FULL_27day_avg.csv'
WIND_file = './data/27_day_avg/WIND_27day_avg.csv'
OMNI_daily_file = "data/27_day_avg/OMNI_27day_avg_daily.csv"
OMNI_hr_file = "data/27_day_avg/OMNI_27day_avg_hr.csv"

Solanki_Vieira_phi_file = './data/models/Solanki_Vieira_open_flux_fix.csv'
owens_interp_file = 'data/models/owens_equat_interp.csv'
owens_orig_file = 'data/models/owens_equatorial.csv'

ACE_dict = pd.read_csv(ACE_file).to_dict('list')
DSCOVR_dict = pd.read_csv(DSCOVR_file).to_dict('list')
HELIOS1_dict = pd.read_csv(HELIOS1_file).to_dict('list')
HELIOS2_dict = pd.read_csv(HELIOS2_file).to_dict('list')
STEREO_A_dict = pd.read_csv(STEREO_A_file).to_dict('list')
STEREO_B_dict = pd.read_csv(STEREO_B_file).to_dict('list')
ULYSSES_dict = pd.read_csv(ULYSSES_file).to_dict('list')
ULYSSES_FULL_dict = pd.read_csv(ULYSSES_FULL_file).to_dict('list')
WIND_dict = pd.read_csv(WIND_file).to_dict('list')
OMNI_daily_dict = pd.read_csv(OMNI_daily_file).to_dict("list")
OMNI_hr_dict = pd.read_csv(OMNI_hr_file).to_dict("list")

sunspot_dict = pd.read_csv(sunspot_file).to_dict('list')
Solanki_Vieira_dict = pd.read_csv(Solanki_Vieira_phi_file).to_dict('list')
owens_interp_dict = pd.read_csv(owens_interp_file).to_dict('list')
owens_orig_dict = pd.read_csv(owens_orig_file).to_dict('list')

data_dict = {**ACE_dict, **DSCOVR_dict, **HELIOS1_dict, **HELIOS2_dict, **STEREO_A_dict, **STEREO_B_dict,
             **ULYSSES_dict, **ULYSSES_FULL_dict, **WIND_dict, **OMNI_daily_dict, **OMNI_hr_dict}

# Add the Cranmer 2017 sunspot vs mdot law, restricting data to years past 1970 (index 1938 in data)
# Plot mdot vs year to see it
cranmer_law_fltyr = np.array([yr for i, yr in enumerate(sunspot_dict.get("Year (SILS)")) if i > 1938])
cranmer_law_mdot = np.array([2.21 * 10 ** 9 * (s + 570) for i, s in enumerate(sunspot_dict.get("Sunspot Number (SILS)")) if i > 1938])

# Add the sunspot data from file with the same constraints as the Cranmer 2017 power law, index-wise
sn_fltyr = cranmer_law_fltyr
sunspot_num = np.array([sn for i, sn in enumerate(sunspot_dict.get("Sunspot Number (SILS)")) if i > 1938])

# Add the Solanki-Vieira model data, restricting to years past 1970 (index 98617 in data)
sv_fltyr = [yr for i, yr in enumerate(Solanki_Vieira_dict.get("Year (Solanki-Vieira)")) if i > 98617]
sv_phi = [10**22 * phi
          for i, phi in enumerate(Solanki_Vieira_dict.get("Open Flux [10^-22 Mx] (Solanki-Vieira)")) if i > 98617]
fit_factor = 1.5
sv_phi = np.array(sv_phi) * fit_factor

# Add the Owens model data, restricting to years past 1960 (index 98617 in interpolated, index 345 in original)
owens_interp_fltyr = sv_fltyr
owens_orig_fltyr = np.array([yr for i, yr in enumerate(owens_orig_dict.get("Year (Owens)"))
                             if i >= 0])

owens_interp_phi = np.array([phi for i, phi in enumerate(owens_interp_dict.get("Open Flux [Mx] (Owens to S-V)")) if i > 98617])
owens_orig_phi = np.array([phi for i, phi in enumerate(owens_orig_dict.get("Open Flux [Mx] (Owens)"))
                           if i >= 0]) / 2

owens_interp_vr = np.array([vr * 10**5 for i, vr in enumerate(
    owens_interp_dict.get("Radial Wind Velocity [km s^-1] (Owens to S-V)")) if i > 98617])
owens_orig_vr = np.array([vr * 10**5 for i, vr in enumerate(owens_orig_dict.get("Radial Wind Velocity [km s^-1] (Owens)"))
                          if i >= 345])

# MCMC parameter fits; of the form mdot = a*(phi**b + v**c (+ sn**d)) *PARETO DISTRIBUTION*
# Scaling factor multiplied to entire function!
# a = 99.73  # original: 99.73 +/- 3.653
# b = 0.432  # original: 0.432 +/- 0.000
# c = 0.665  # original: 0.665 +/- 0.249
# scaling = 1.5

# new MCMC parameters from *LOGNORMAL DISTRIBUTION*
# Scaling factor multiplied to entire function!
a = 55  # original: 49.43 +/- 15.77
b = 0.453  # original: 0.274 +/- 0.024
c = 1.25  # original: 0.549 +/- 0.074
scaling = 1

# newer MCMC parameters for more flexible fit
a1 = 1
a2 = 0.453
a3 = -0.01
a4 = 1.85

# Lambda for 2 parameter mcmc fit (phi and vr)
mcmc_fit = lambda phi, vr: a * (phi**b + vr**c) * (scaling)

mcmc_fit2 = lambda phi, vr: abs(a1 * phi**a2 + a3 * vr**a4)

# Dispersion line using only open flux
disp_line = lambda phi: abs(-1.68 * 10**-11 * phi - 2.50 * 10**11)

# 2 parameter dispersion upper; just a fit of the upper data
disp_upper2d = lambda phi, vr: a * (phi**(b + 0.005) + vr**c)

# 2 parameter dispersion lower; just a fit of the lower data
disp_lower2d = lambda phi, vr: a * (phi**(b - 0.02) + vr**c)


# Plotting starts here
# Get required data from the data dictionary
time_ace = np.array(data_dict.get("Year (ACE)"))
br_ace = np.array(data_dict.get("Radial Magnetic Field Magnitude [nT] (ACE)"))
mdot_ace = np.array(data_dict.get("Mass Loss Rate [g s^-1] (ACE)"))
phi_ace = np.array(data_dict.get("Open Flux [Mx] (ACE)"))
sn_ace = np.array(data_dict.get("Sunspot Number (ACE)"))
vr_ace = np.array(data_dict.get("Radial Wind Velocity [km s^-1] (ACE)")) * 10**5
nmf_ace = np.array(data_dict.get("Normalised Mass Flux [g s^-1] (ACE)"))
torque_ace = np.array(data_dict.get("Solar Torque [erg] (ACE)"))

time_dscovr = np.array(data_dict.get("Year (DSCOVR)"))
br_dscovr = np.array(data_dict.get("Radial Magnetic Field Magnitude [nT] (DSCOVR)"))
mdot_dscovr = np.array(data_dict.get("Mass Loss Rate [g s^-1] (DSCOVR)"))
phi_dscovr = np.array(data_dict.get("Open Flux [Mx] (DSCOVR)"))
sn_dscovr = np.array(data_dict.get("Sunspot Number (DSCOVR)"))
vr_dscovr = np.array(data_dict.get("Radial Wind Velocity [km s^-1] (DSCOVR)")) * 10**5
nmf_dscovr = np.array(data_dict.get("Normalised Mass Flux [g s^-1] (DSCOVR)"))
torque_dscovr = np.array(data_dict.get("Solar Torque [erg] (DSCOVR)"))

time_helios1 = np.array(data_dict.get("Year (Helios 1)"))
br_helios1 = np.array(data_dict.get("Radial Magnetic Field Magnitude [nT] (Helios 1)"))
mdot_helios1 = np.array(data_dict.get("Mass Loss Rate [g s^-1] (Helios 1)"))
phi_helios1 = np.array(data_dict.get("Open Flux [Mx] (Helios 1)"))
sn_helios1 = np.array(data_dict.get("Sunspot Number (Helios 1)"))
vr_helios1 = np.array(data_dict.get("Radial Wind Velocity [km s^-1] (Helios 1)")) * 10**5
nmf_helios1 = np.array(data_dict.get("Normalised Mass Flux [g s^-1] (Helios 1)"))
torque_helios1 = np.array(data_dict.get("Solar Torque [erg] (Helios 1)"))

time_helios2 = np.array(data_dict.get("Year (Helios 2)"))
br_helios2 = np.array(data_dict.get("Radial Magnetic Field Magnitude [nT] (Helios 2)"))
mdot_helios2 = np.array(data_dict.get("Mass Loss Rate [g s^-1] (Helios 2)"))
phi_helios2 = np.array(data_dict.get("Open Flux [Mx] (Helios 2)"))
sn_helios2 = np.array(data_dict.get("Sunspot Number (Helios 2)"))
vr_helios2 = np.array(data_dict.get("Radial Wind Velocity [km s^-1] (Helios 2)")) * 10**5
nmf_helios2 = np.array(data_dict.get("Normalised Mass Flux [g s^-1] (Helios 2)"))
torque_helios2 = np.array(data_dict.get("Solar Torque [erg] (Helios 2)"))

time_stereoa = np.array(data_dict.get("Year (STEREO A)"))
br_stereoa = np.array(data_dict.get("Radial Magnetic Field Magnitude [nT] (STEREO A)"))
mdot_stereoa = np.array(data_dict.get("Mass Loss Rate [g s^-1] (STEREO A)"))
phi_stereoa = np.array(data_dict.get("Open Flux [Mx] (STEREO A)"))
sn_stereoa = np.array(data_dict.get("Sunspot Number (STEREO A)"))
vr_stereoa = np.array(data_dict.get("Radial Wind Velocity [km s^-1] (STEREO A)")) * 10**5
nmf_stereoa = np.array(data_dict.get("Normalised Mass Flux [g s^-1] (STEREO A)"))
torque_stereoa = np.array(data_dict.get("Solar Torque [erg] (STEREO A)"))

time_stereob = np.array(data_dict.get("Year (STEREO B)"))
br_stereob = np.array(data_dict.get("Radial Magnetic Field Magnitude [nT] (STEREO B)"))
mdot_stereob = np.array(data_dict.get("Mass Loss Rate [g s^-1] (STEREO B)"))
phi_stereob = np.array(data_dict.get("Open Flux [Mx] (STEREO B)"))
sn_stereob = np.array(data_dict.get("Sunspot Number (STEREO B)"))
vr_stereob = np.array(data_dict.get("Radial Wind Velocity [km s^-1] (STEREO B)")) * 10**5
nmf_stereob = np.array(data_dict.get("Normalised Mass Flux [g s^-1] (STEREO B)"))
torque_stereob = np.array(data_dict.get("Solar Torque [erg] (STEREO B)"))

time_ulysses = np.array(data_dict.get("Year (Ulysses)"))
br_ulysses = np.array(data_dict.get("Radial Magnetic Field Magnitude [nT] (Ulysses)"))
mdot_ulysses = np.array(data_dict.get("Mass Loss Rate [g s^-1] (Ulysses)"))
phi_ulysses = np.array(data_dict.get("Open Flux [Mx] (Ulysses)"))
sn_ulysses = np.array(data_dict.get("Sunspot Number (Ulysses)"))
vr_ulysses = np.array(data_dict.get("Radial Wind Velocity [km s^-1] (Ulysses)")) * 10**5
nmf_ulysses = np.array(data_dict.get("Normalised Mass Flux [g s^-1] (Ulysses)"))
torque_ulysses = np.array(data_dict.get("Solar Torque [erg] (Ulysses)"))

time_wind = np.array(data_dict.get("Year (WIND)"))
br_wind = np.array(data_dict.get("Radial Magnetic Field Magnitude [nT] (WIND)"))
mdot_wind = np.array(data_dict.get("Mass Loss Rate [g s^-1] (WIND)"))
phi_wind = np.array(data_dict.get("Open Flux [Mx] (WIND)"))
sn_wind = np.array(data_dict.get("Sunspot Number (WIND)"))
vr_wind = np.array(data_dict.get("Radial Wind Velocity [km s^-1] (WIND)")) * 10**5
nmf_wind = np.array(data_dict.get("Normalised Mass Flux [g s^-1] (WIND)"))
torque_wind = np.array(data_dict.get("Solar Torque [erg] (WIND)"))

# Filter out the bad data for OMNI
time_omni_daily = np.array([yr for i, yr in enumerate(data_dict.get("Year (OMNI Daily)"))
                            if data_dict.get("Flags (OMNI Daily)")[i] != 1])
br_omni_daily = np.array([br for i, br in enumerate(data_dict.get("Radial Magnetic Field Magnitude [nT] (OMNI Daily)"))
                            if data_dict.get("Flags (OMNI Daily)")[i] != 1])
mdot_omni_daily = np.array([mdot for i, mdot in enumerate(data_dict.get("Mass Loss Rate [g s^-1] (OMNI Daily)"))
                            if data_dict.get("Flags (OMNI Daily)")[i] != 1])
phi_omni_daily = np.array([phi for i, phi in enumerate(data_dict.get("Open Flux [Mx] (OMNI Daily)"))
                            if data_dict.get("Flags (OMNI Daily)")[i] != 1])
sn_omni_daily = np.array([sn for i, sn in enumerate(data_dict.get("Sunspot Number (OMNI Daily)"))
                            if data_dict.get("Flags (OMNI Daily)")[i] != 1])
vr_omni_daily = np.array([vr for i, vr in enumerate(data_dict.get("Radial Wind Velocity [km s^-1] (OMNI Daily)"))
                            if data_dict.get("Flags (OMNI Daily)")[i] != 1]) * 10**5
nmf_omni_daily = np.array([nmf for i, nmf in enumerate(data_dict.get("Normalised Mass Flux [g s^-1] (OMNI Daily)"))
                            if data_dict.get("Flags (OMNI Daily)")[i] != 1])
torque_omni_daily = np.array([torque for i, torque in enumerate(data_dict.get("Torque [erg] (OMNI Daily)"))
                            if data_dict.get("Flags (OMNI Daily)")[i] != 1])


time_omni_hr = np.array([yr for i, yr in enumerate(data_dict.get("Year (OMNI Hr)"))
                            if data_dict.get("Flags (OMNI Hr)")[i] != 1])
br_omni_hr = np.array([br for i, br in enumerate(data_dict.get("Radial Magnetic Field Magnitude [nT] (OMNI Hr)"))
                            if data_dict.get("Flags (OMNI Hr)")[i] != 1])
mdot_omni_hr = np.array([mdot for i, mdot in enumerate(data_dict.get("Mass Loss Rate [g s^-1] (OMNI Hr)"))
                            if data_dict.get("Flags (OMNI Hr)")[i] != 1])
phi_omni_hr = np.array([phi for i, phi in enumerate(data_dict.get("Open Flux [Mx] (OMNI Hr)"))
                            if data_dict.get("Flags (OMNI Hr)")[i] != 1])
sn_omni_hr = np.array([sn for i, sn in enumerate(data_dict.get("Sunspot Number (OMNI Hr)"))
                            if data_dict.get("Flags (OMNI Hr)")[i] != 1])
vr_omni_hr = np.array([vr for i, vr in enumerate(data_dict.get("Radial Wind Velocity [km s^-1] (OMNI Hr)"))
                            if data_dict.get("Flags (OMNI Hr)")[i] != 1]) * 10**5
nmf_omni_hr = np.array([nmf for i, nmf in enumerate(data_dict.get("Normalised Mass Flux [g s^-1] (OMNI Hr)"))
                            if data_dict.get("Flags (OMNI Hr)")[i] != 1])
torque_omni_hr = np.array([torque for i, torque in enumerate(data_dict.get("Torque [erg] (OMNI Hr)"))
                            if data_dict.get("Flags (OMNI Hr)")[i] != 1])

# Limit data for SILS to 1970 and beyond
time_sils = [yr for i, yr in enumerate(sunspot_dict.get("Year (SILS)")) if i > 1937]
sn_sils = [sn for i, sn in enumerate(sunspot_dict.get("Sunspot Number (SILS)")) if i > 1937]

time_sils = np.array(time_sils)
sn_sils = np.array(sn_sils)

# 3 panel MCMC plot
# f, axarr = plt.subplots(3, sharex=True)
#
# # disp_lines defined directly below are from original Dash plotting, taking the 95th percentile line
# # Define the mdot vs phi fit from ACE as an unrefined law
# # mdot_phi_law_ace = 0.13 * (phi_ace ** 0.568)
# # disp_phi_line_ace = 6.71 * 10 ** (-12) * phi_ace  # 95th percentile
# #
# # f_upper_phi = mdot_phi_law_ace + disp_phi_line_ace
# # f_lower_phi = mdot_phi_law_ace - disp_phi_line_ace
# #
# # # Define the mdot vs sn fit from ACE as an unrefined law, using sunspot numbers and times from SILS
# # mdot_sn_law_ace = 9.9 * 10**11 * (sn_sils**0.075) + 10**11  # Added factor so mdot does not = zero when sn = zero
# # disp_sn_line_ace = 4.03 * 10**9 * sn_sils  # 95th percentile
# #
# # f_upper_sn = mdot_sn_law_ace + disp_sn_line_ace
# # f_lower_sn = mdot_sn_law_ace - disp_sn_line_ace
# #
# # # Define the mdot vs phi fit from ACE using Solanki-Vieira open fluxes
# # mdot_phi_law_sv = 0.13 * (np.array(sv_phi) ** 0.568)
# # disp_phi_line_sv = 6.71 * 10**(-12) * np.array(sv_phi)  # 95th percentile
# #
# # f_upper_phi_sv = mdot_phi_law_sv + disp_phi_line_sv
# # f_lower_phi_sv = mdot_phi_law_sv - disp_phi_line_sv
#
#
# # Plot data on all 3 panels
# time_list = np.array([time_ace, time_dscovr, time_helios1, time_helios2, time_stereoa, time_stereob, time_ulysses, time_wind])
# phi_list = np.array([phi_ace, phi_dscovr, phi_helios1, phi_helios2, phi_stereoa, phi_stereob, phi_ulysses, phi_wind])
# mdot_list = np.array([mdot_ace, mdot_dscovr, mdot_helios1, mdot_helios2, mdot_stereoa, mdot_stereob, mdot_ulysses, mdot_wind])
# colour_list = ['#1f77b4', '#1f77b4', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#bcbd22', '#17becf']
#
# for i in range(0, len(time_list)):
#     axarr[0].scatter(time_list[i], phi_list[i] / (10**22), color=colour_list[i], marker='.', s=1, zorder=2)
#     axarr[1].scatter(time_list[i], mdot_list[i] / (10**12), color=colour_list[i], marker='.', s=1, zorder=2)
#     axarr[2].scatter(time_list[i], stut.calc_torque(
#         mdot_list[i], stut.calc_rel_alfven_rad(stut.calc_wind_magnetisation(phi_list[i], mdot_list[i]))) / (10**30),
#                      color=colour_list[i], marker='.', s=1, zorder=2)
#
# # Open Flux
# axarr[0].plot(owens_orig_fltyr, owens_orig_phi / (10**22), 'b-', zorder=1)
# axarr[0].set_ylabel(r"$\Phi_{open}$ [$x10^{22}$ Mx]")
#
# # Mass Loss Rate
# fit = mcmc_fit(owens_orig_phi, owens_orig_vr) - 0.4 * 10**12  # By eye tweaking
# disp_upper = abs(6.37 * 10**(-12) * owens_orig_phi + 7.8 * 10**11) / 3
# disp_lower = abs(9.65 * 10**(-12) * owens_orig_phi - 1.69 * 10**11) / 3
#
# axarr[1].plot(owens_orig_fltyr, fit / (10**12), color='#ff9900', ls='-', zorder=1)
# axarr[1].plot(owens_orig_fltyr, (fit + disp_upper) / (10**12), color='#ff9900', ls='--', alpha=0.5, zorder=1)
# axarr[1].plot(owens_orig_fltyr, (fit - disp_lower) / (10**12), color='#ff9900', ls='--', alpha=0.5, zorder=1)
#
# axarr[1].set_ylabel(r"$\dot{M}$ [$x10^{12}$ g s$^{-1}$]")
#
# # Torque
# windmag = stut.calc_wind_magnetisation(owens_orig_phi, fit)
# windmag_upper = stut.calc_wind_magnetisation(owens_orig_phi, fit + disp_upper)
# windmag_lower = stut.calc_wind_magnetisation(owens_orig_phi, fit - disp_lower)
#
# relalfrad = stut.calc_rel_alfven_rad(windmag)
# relalfrad_upper = stut.calc_rel_alfven_rad(windmag_upper)
# relalfrad_lower = stut.calc_rel_alfven_rad(windmag_lower)
#
# torque = stut.calc_torque(fit, relalfrad)
# torque_upper = stut.calc_torque(fit + disp_upper, relalfrad_upper)
# torque_lower = stut.calc_torque(fit - disp_lower, relalfrad_lower)
#
# axarr[2].plot(owens_orig_fltyr, torque / (10**30), color='#ff2b4b', ls='-', zorder=1)
# axarr[2].plot(owens_orig_fltyr, torque_upper / (10**30), color='#ff2b4b', ls='-', alpha=0.5, zorder=1)
# axarr[2].plot(owens_orig_fltyr, torque_lower / (10**30), color='#ff2b4b', ls='-', alpha=0.5, zorder=1)
#
# axarr[2].set_ylabel(r"Torque [$x10^{30}$ erg]")
# # plt.plot(owens_orig_fltyr, disp_upper, color='#ff1400', label="Upper Dispersion")
# # plt.plot(owens_orig_fltyr, disp_lower, color='#2ca02c', label="Lower Dispersion")
# # plt.fill_between(owens_orig_fltyr, fit + disp, fit - disp, '#006600', alpha=0.5, zorder=1)
#
# # util.get_2d_dispersion(phi_ace, vr_ace, mdot_ace, mcmc_fit(phi_ace, vr_ace))
#
# plt.xlabel("Year")

# Plot the MCMC models individually (they're mdot vs (phi, vr) fits so best to check all spacecraft individually
# plt.plot(time_ace, mcmc_fit(phi_ace, vr_ace), 'b')
# f_upper, f_lower = util.get_dispersion(phi_ace, mdot_ace, mcmc_fit(phi_ace, vr_ace), 95)
# plt.fill_between(time_ace, f_upper, f_lower, color='cyan', alpha=0.9, zorder=0)

# plt.plot(time_dscovr, mcmc_fit(phi_dscovr, vr_dscovr), 'b')
# f_upper, f_lower = util.get_dispersion(phi_dscovr, mdot_dscovr, mcmc_fit(phi_dscovr, vr_dscovr), 95)
# plt.fill_between(time_dscovr, f_upper, f_lower, color='cyan', alpha=0.9, zorder=0)
#
# plt.plot(time_helios1, mcmc_fit(phi_helios1, vr_helios1), 'b')
# f_upper, f_lower = util.get_dispersion(phi_helios1, mdot_helios1, mcmc_fit(phi_helios1, vr_helios1), 95)
# plt.fill_between(time_helios1, f_upper, f_lower, color='cyan', alpha=0.9, zorder=0)
#
# plt.plot(time_helios2, mcmc_fit(phi_helios2, vr_helios2), 'b')
# f_upper, f_lower = util.get_dispersion(phi_helios2, mdot_helios2, mcmc_fit(phi_helios2, vr_helios2), 95)
# plt.fill_between(time_helios2, f_upper, f_lower, color='cyan', alpha=0.9, zorder=0)
#
# plt.plot(time_stereoa, mcmc_fit(phi_stereoa, vr_stereoa), 'b')
# f_upper, f_lower = util.get_dispersion(phi_stereoa, mdot_stereoa, mcmc_fit(phi_stereoa, vr_stereoa), 95)
# plt.fill_between(time_stereoa, f_upper, f_lower, color='cyan', alpha=0.9, zorder=0)
#
# plt.plot(time_stereob, mcmc_fit(phi_stereob, vr_stereob), 'b')
# f_upper, f_lower = util.get_dispersion(phi_stereob, mdot_stereob, mcmc_fit(phi_stereob, vr_stereob), 95)
# plt.fill_between(time_stereob, f_upper, f_lower, color='cyan', alpha=0.9, zorder=0)
#
# plt.plot(time_ulysses, mcmc_fit(phi_ulysses, vr_ulysses), 'b')
# f_upper, f_lower = util.get_dispersion(phi_ulysses, mdot_ulysses, mcmc_fit(phi_ulysses, vr_ulysses), 95)
# plt.fill_between(time_ulysses, f_upper, f_lower, color='cyan', alpha=0.9, zorder=0)
#
# plt.plot(time_wind, mcmc_fit(phi_wind, vr_wind), 'b')
# f_upper, f_lower = util.get_dispersion(phi_wind, mdot_wind, mcmc_fit(phi_wind, vr_wind), 95)
# plt.fill_between(time_wind, f_upper, f_lower, color='cyan', alpha=0.9, zorder=0)

# Plot ACE phi model and dispersion lines
# plt.plot(time_ace, mdot_phi_law_ace, 'r-', label='ACE Open Flux Model', zorder=0)
# plt.fill_between(time_ace, f_upper_phi, f_lower_phi, color='lightcoral', alpha=0.5, zorder=0)
# plt.plot(time_ace, f_upper_phi, color='orange', ls='-')
# plt.plot(time_ace, f_lower_phi, color='orange', ls='-')

# Plot ACE phi model and dispersion lines using Solanki-Vieira times and data
# plt.plot(sv_fltyr, mdot_phi_law_sv, color='#0000ff', ls='-',  label="ACE (Solanki-Vieira 2010) Open Flux Model ("
#                                                 + str(fit_factor) + " scaling)", zorder=0)
# plt.fill_between(sv_fltyr, f_upper_phi_sv, f_lower_phi_sv, color='lime', alpha=0.5, zorder=0)
# plt.plot(sv_fltyr, f_upper_phi_sv, color='lime', ls='-')
# plt.plot(sv_fltyr, f_lower_phi_sv, color='lime', ls='-')

# Plot ACE SN model and dispersion lines
# plt.plot(time_sils, mdot_sn_law_ace, 'b-', label='ACE Sunspot Number Model', zorder=0)
# plt.fill_between(time_sils, f_upper_sn, f_lower_sn, color='cyan', alpha=0.5, zorder=0)
# plt.plot(time_ace, f_upper_sn, 'c-')
# plt.plot(time_ace, f_lower_sn, 'c-')

# Cranmer power law
# plt.plot(cranmer_law_fltyr, cranmer_law_mdot, color='magenta', ls='-', label='Cranmer 2017 Power Law', zorder=0)

# Use the open flux dispersion as the overall dispersion since it is dominant source of error
# Plot MCMC models using interpolated Owens vr and S-V phi data
# fit = mcmc_fit(sv_phi, owens_interp_vr)
# disp = disp_line(sv_phi) / 3
#
# plt.plot(sv_fltyr, fit, color='#0099cc', ls='-', label="MCMC lognormal fit with Owens vr & S-V phi")
# plt.plot(sv_fltyr, fit + disp, color='#0099cc', ls='--', alpha=0.5)
# plt.plot(sv_fltyr, fit - disp, color='#0099cc', ls='--', alpha=0.5)
# plt.fill_between(sv_fltyr, fit + disp, fit - disp, '#0099cc', alpha=0.5, zorder=1)

# Plot MCMC models using interpolated Owens vr and phi data
# fit = mcmc_fit(owens_interp_phi, owens_interp_vr)
# disp = disp_line(owens_interp_phi) / 3
#
# plt.plot(owens_interp_fltyr, fit, color='#00e600', ls='-',
#          label="MCMC lognormal fit with interpolated Owens vr and phi")
# plt.plot(owens_interp_fltyr, fit + disp, color='#00e600', ls='--', alpha=0.5)
# plt.plot(owens_interp_fltyr, fit - disp, color='#00e600', ls='--', alpha=0.5)
# plt.fill_between(owens_interp_fltyr, fit + disp, fit - disp, '#00e600', alpha=0.5, zorder=1)

# plt.legend(bbox_to_anchor=(0.35, -0.3), loc=6, ncol=2, mode="expand", borderaxespad=0., frameon=False)
# plt.tight_layout(pad=7)

# Refine Adam's fit, splitting the data into slow and fast wind regimes
# First define the fit function, x_inputs[0] == vr, x_inputs[1] == phi_open
fitfunc = lambda x_inputs, a_prime, b_prime: (a_prime * x_inputs[0] + b_prime) * x_inputs[1]

# Split data into slow and fast where 500 km/s is the cutoff (data is in cm/s so multiplied by 10**5)
input_slow = [(vr_ace[i], phi_ace[i]) for i in range(0, len(vr_ace)) if vr_ace[i] <= 500e5]
nmf_slow = [nmf_ace[i] for i in range(0, len(vr_ace)) if vr_ace[i] <= 500e5]

input_fast = [(vr_ace[i], phi_ace[i]) for i in range(0, len(vr_ace)) if vr_ace[i] > 500e5]
nmf_fast = [nmf_ace[i] for i in range(0, len(vr_ace)) if vr_ace[i] > 500e5]


# Function to add spacecraft data to the existing ACE dataset quickly
def add_data(slow_data, slow_nmf, fast_data, fast_nmf, vr_data, phi_data, nmf_data):
    for i in range(0, len(vr_data)):
        if vr_data[i] <= 500e5:
            slow_data.append((vr_data[i], phi_data[i]))
            slow_nmf.append(nmf_data[i])
        else:
            fast_data.append((vr_data[i], phi_data[i]))
            fast_nmf.append(nmf_data[i])

    return slow_data, fast_data


# Add all spacecraft data
input_slow, input_fast = add_data(input_slow, nmf_slow, input_fast, nmf_fast, vr_dscovr, phi_dscovr, nmf_dscovr)
input_slow, input_fast = add_data(input_slow, nmf_slow, input_fast, nmf_fast, vr_helios1, phi_helios1, nmf_helios1)
input_slow, input_fast = add_data(input_slow, nmf_slow, input_fast, nmf_fast, vr_helios2, phi_helios2, nmf_helios2)
input_slow, input_fast = add_data(input_slow, nmf_slow, input_fast, nmf_fast, vr_stereoa, phi_stereoa, nmf_stereoa)
input_slow, input_fast = add_data(input_slow, nmf_slow, input_fast, nmf_fast, vr_stereob, phi_stereob, nmf_stereob)
input_slow, input_fast = add_data(input_slow, nmf_slow, input_fast, nmf_fast, vr_ulysses, phi_ulysses, nmf_ulysses)
input_slow, input_fast = add_data(input_slow, nmf_slow, input_fast, nmf_fast, vr_wind, phi_wind, nmf_wind)
# input_slow, input_fast = add_data(input_slow, nmf_slow, input_fast, nmf_fast, vr_omni_daily, phi_omni_daily, nmf_omni_daily)
input_slow, input_fast = add_data(input_slow, nmf_slow, input_fast, nmf_fast, vr_omni_hr, phi_omni_hr, nmf_omni_hr)

# Format arrays
input_slow = np.array(input_slow).T
nmf_slow = np.array(nmf_slow)

input_fast = np.array(input_fast).T
nmf_fast = np.array(nmf_fast)

# initial guesses from Adam's fit
a_p_slow = 1e-15
b_p_slow = 9.5e-13
a_p_fast = -2.5e-15
b_p_fast = 2.875e-12

p0_slow = [a_p_slow, b_p_slow]
p0_fast = [a_p_fast, b_p_fast]

# the fit
params_slow, cov_slow = opt.curve_fit(fitfunc, input_slow, nmf_slow, p0=p0_slow)
params_fast, cov_fast = opt.curve_fit(fitfunc, input_fast, nmf_fast, p0=p0_fast)

# slapdash lambda for graphing the straight line
# fitfunc2 = lambda const_phi, vr, a, b: const_phi * (a*np.array(vr) + b)

# plt.figure()
# plt.scatter(input_slow[0], nmf_slow, c=input_slow[1], s=20, cmap='plasma')
# plt.scatter(input_fast[0], nmf_fast, c=input_fast[1], s=20, cmap='plasma')
#
# plt.plot(sorted(input_slow[0]), fitfunc2(6.e22, sorted(input_slow[0]), params_slow[0], params_slow[1]),
#          c='#371b99', ls='-', zorder=1, label="Low Trend")
# plt.plot(sorted(input_fast[0]), fitfunc2(6.e22, sorted(input_fast[0]), params_fast[0], params_fast[1]),
#          c='#371b99', ls='-', zorder=1)
#
# plt.plot(sorted(input_slow[0]), fitfunc2(8.e22, sorted(input_slow[0]), params_slow[0], params_slow[1]),
#          c='#dc5e65', ls='-', zorder=1, label="Middle Trend")
# plt.plot(sorted(input_fast[0]), fitfunc2(8.e22, sorted(input_fast[0]), params_fast[0], params_fast[1]),
#          c='#dc5e65', ls='-', zorder=1)
#
# plt.plot(sorted(input_slow[0]), fitfunc2(12.e22, sorted(input_slow[0]), params_slow[0], params_slow[1]),
#          c='#fcc029', ls='-', zorder=1, label="High Trend")
# plt.plot(sorted(input_fast[0]), fitfunc2(12.e22, sorted(input_fast[0]), params_fast[0], params_fast[1]),
#          c='#fcc029', ls='-', zorder=1)
# # plt.plot(sorted(input_slow[0]), fitfunc2(sorted(input_slow[0]), p0_slow[0], p0_slow[1]) / 2e4, 'g-', zorder=1,
# #          label="Adam's Original Trend")
# # plt.plot(sorted(input_fast[0]), fitfunc2(sorted(input_fast[0]), p0_fast[0], p0_fast[1]) / 1e5 + 2e11, 'g-', zorder=1)
#
# plt.xlabel(r"Radial Wind Velocity [cm s$^{-1}$]")
# plt.ylabel(r"Normalised Mass Flux [g s$^{-1}$]")
#
# plt.colorbar(label="Open Flux [Mx]")
# plt.legend()

# 5 panel plot
# fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=5, ncols=1, figsize=(15, 10), sharex=True)

# # x-axis is time for all of these; just grab br from the Owens dictionary if needed; all plots are only equatorial data
# # Plot data on all 5 panels
# time_list = np.array([time_ace, time_dscovr, time_helios1, time_helios2, time_stereoa, time_stereob, time_ulysses, time_wind])
br_list = np.array([br_ace, br_dscovr, br_helios1, br_helios2, br_stereoa, br_stereob, br_ulysses, br_wind])
vr_list = np.array([vr_ace, vr_dscovr, vr_helios1, vr_helios2, vr_stereoa, vr_stereob, vr_ulysses, vr_wind])
phi_list = np.array([phi_ace, phi_dscovr, phi_helios1, phi_helios2, phi_stereoa, phi_stereob, phi_ulysses, phi_wind])
mdot_list = np.array([mdot_ace, mdot_dscovr, mdot_helios1, mdot_helios2, mdot_stereoa, mdot_stereob, mdot_ulysses, mdot_wind])
# torque_list = np.array([torque_ace, torque_dscovr, torque_helios1, torque_helios2, torque_stereoa, torque_stereob,
#                         torque_ulysses, torque_wind])
# colour_list = ['#1f77b4', '#1f77b4', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#bcbd22', '#17becf']
#
# for i in range(0, len(time_list)):
#     ax2.scatter(time_list[i], phi_list[i] / (1e22), color=colour_list[i], marker='.', s=1, zorder=2)
#     ax4.scatter(time_list[i], mdot_list[i] / (1e12), color=colour_list[i], marker='.', s=1, zorder=2)
#     ax5.scatter(time_list[i], torque_list[i] / (1e30), color=colour_list[i], marker='.', s=1, zorder=2)
#
# avg_spacecraft_torque = np.mean(np.array([item for sublist in torque_list for item in sublist]))
#
# # Plot OMNI data for open flux, mass loss rate and torque
# # ax2.scatter(time_omni_daily, phi_omni_daily / (1e22), c="#33cc33", marker='.', s=1, zorder=2)
# ax2.scatter(time_omni_hr, phi_omni_hr / (1e22), c="#cc00ff", marker='.', s=1, zorder=2)
#
# # ax4.scatter(time_omni_daily, mdot_omni_daily / (1e12), c="#33cc33", marker='.', s=1, zorder=2)
# ax4.scatter(time_omni_hr, mdot_omni_hr / (1e12), c="#cc00ff", marker='.', s=1, zorder=2)
#
# # ax5.scatter(time_omni_daily, torque_omni_daily / (1e30), c="#33cc33", marker='.', s=1, zorder=2)
# ax5.scatter(time_omni_hr, torque_omni_hr / (1e30), c="#cc00ff", marker='.', s=1, zorder=2)

# # First axes: br
owens_full_mag_file = "data/models/owens_full_mag.txt"
owens_mag_array = np.genfromtxt(owens_full_mag_file)
# # Restrict to years past 1960 (index 345)
owens_br_array = np.array([mag for i, mag in enumerate(owens_mag_array[:, 1:]) if i >= 0]) / 1e5  # divide owens br by 10**5 for scaling
owens_lat = np.arange(-87.5, 92.5, 5)
time, lat = np.meshgrid(owens_orig_fltyr, owens_lat)
# #
# # ax1_a = ax1.pcolormesh(time, lat, owens_br_array.T, cmap="seismic")
# #
# # divider1 = make_axes_locatable(ax1)
# # cax1 = divider1.append_axes("right", size="0.5%", pad=0.05)
# # plt.colorbar(ax1_a, cax=cax1, orientation='vertical', label=r'B$_{r}$ [G]')
# # ax1.set_ylabel(r'Latitude')
# #
# # Second axes: open flux
# # # Calculate the spherical integral for open flux using delta theta and delta phi
# owens_open_f = np.array([2. * np.pi * np.sum(np.abs(owens_br_array[i, :] * (1.5e13)**2 * np.cos(owens_lat * np.pi / 180)
#                                            * np.sin(0.5 * 5 * np.pi / 180)))
#                 for i in range(0, len(owens_br_array[:, 0]))])
#
# # Assume isotropy and calculate the open flux from equatorial; I think 18th column is -2.5 or 2.5 degrees (close enough)
# # It's 4pi in the surface integral formula; 2pi here because we divide owens br by 2
# owens_equat_open_f = np.array(2. * np.pi * (1.5e13)**2 * np.abs(owens_br_array[:, 18]))
#
# ax2.plot(owens_orig_fltyr, owens_open_f / (1e22), 'k-', label="Owens Data")
# ax2.plot(owens_orig_fltyr, owens_equat_open_f / (1e22), c='grey', ls='-', label="Equatorial Owens Data")
#
# ax2.set_ylabel(r'Open Flux [x $10^{22}$ Mx]')
# ax2.legend()
#
# # Third axes: mass flux
# owens_full_vr_file = "data/models/owens_full_vr.txt"
# owens_wind_array = np.genfromtxt(owens_full_vr_file)
# owens_vr_array = np.array([vr for i, vr in enumerate(owens_wind_array[:, 1:]) if i >= 0]) * 1e5
#
# owens_avg_vr = [np.mean(owens_vr_array[i, :]) * np.cos(owens_lat * np.pi / 180)
#                 for i in range(0, len(owens_vr_array[0, :]))]
# #
# # Array to reference to see if the wind is designated as 'fast' or 'slow'
# owens_slowfast = np.empty_like(owens_vr_array)
#
# for i in range(0, len(owens_vr_array[:, 0])):
#     for j in range(0, len(owens_vr_array[i, :])):
#         if owens_vr_array[i][j] > 500:  # Fast wind
#             owens_slowfast[i, j] = 1
#         else:  # Slow wind
#             owens_slowfast[i, j] = 0

# owens_massflux = np.empty_like(owens_vr_array)
#
#
# Adam's massflux function without open flux
# def mass_flux_r2(type, vr, openF):
#     if type == 0:  # Slow wind
#         mass_flux = openF * (params_slow[0] * vr + params_slow[1])
#     elif type == 1:  # Fast wind
#         mass_flux = openF * (params_fast[0] * vr + params_fast[1])
#     else:  # error
#         mass_flux = 0
#     return mass_flux
# #
#
# for i in range(0, len(owens_massflux[:, 0])):
#     for j in range(0, len(owens_massflux[i, :])):
#         owens_massflux[i, j] = mass_flux_r2(owens_slowfast[i, j], owens_vr_array[i, j], owens_open_f[i])
#
# ax3_a = ax3.pcolormesh(time, lat, owens_massflux.T / (1e12), cmap='jet')
#
# divider3 = make_axes_locatable(ax3)
# cax3 = divider3.append_axes("right", "0.5%", pad=0.05)
# plt.colorbar(ax3_a, cax=cax3, orientation='vertical', label=r'Mass Flux [x $10^{12}$ g s$^{-1}$]')
# ax3.set_ylabel('Latitude')
#
# # Fourth axes: Mass loss rate
# owens_mdot = np.array([np.sum(4.0 * np.pi * np.cos(owens_lat * np.pi / 180) * np.sin(np.pi / 180 * 5/2.) * owens_massflux[i, :])
#               for i in range(0, len(owens_massflux[:, 0]))])
#
# # 18th index should be equatorial (or near enough, I think it's -2.5 or 2.5 degrees)
# owens_equat_mdot = np.array([4. * np.pi * owens_massflux[i, 18] for i in range(0, len(owens_massflux[:, 0]))])
#
# ax4.plot(owens_orig_fltyr, owens_mdot / (1e12), 'k-', label="Fitted Owens Data")
# ax4.plot(owens_orig_fltyr, owens_equat_mdot / (1e12), c='grey', ls='-', label="Equatorial Fitted Owens Data")
#
# ax4.set_ylabel(r'$\dot{M}$ [x $10^{12}$ g s$^{-1}$]')
# ax4.legend()
#
# # Fifth axes: Torque
# owens_torque = 2.3e30 * (np.asarray(owens_mdot) / 1e12)**0.26 * (np.asarray(owens_open_f) / 8e22)**1.48
#
# ax5.plot(owens_orig_fltyr, owens_torque / (1e30), 'k-', label="Owens Data")
#
# ax5.axhline(y=np.mean(owens_torque / (1e30)), color="grey", ls='-', label="Owens Average")
# ax5.axhline(y=avg_spacecraft_torque / (1e30), color="grey", ls='--', label="Spacecraft Average")
#
# ax5.errorbar([1983], [3.16], yerr=[[0.65],[0.65]], fmt='o', c='red', lw=3, zorder=0)  # Pizzo errorbar
# ax5.scatter([1983], [3.16], color='red', edgecolor='k', s=50, label="Pizzo et al. (1983)")  # Pizzo calc average
# ax5.scatter([1999], [2.1], color='orange', edgecolor='k', s=50, zorder=5, label="Li (1999)")  # Li calc average
#
# ax5.set_ylabel(r'Torque [x $10^{30}$ erg]')
# ax5.legend()
#
# ax1.set_xlim(np.min(owens_orig_fltyr), np.max(owens_orig_fltyr))
# plt.rcParams.update({'font.size': 6})
# plt.tight_layout()

# 3 panel model comparison plot
# fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)

time_list = np.array([time_ace, time_dscovr, time_helios1, time_helios2, time_stereoa, time_stereob, time_ulysses, time_wind])
br_list = np.array([br_ace, br_dscovr, br_helios1, br_helios2, br_stereoa, br_stereob, br_ulysses, br_wind])
vr_list = np.array([vr_ace, vr_dscovr, vr_helios1, vr_helios2, vr_stereoa, vr_stereob, vr_ulysses, vr_wind])
phi_list = np.array([phi_ace, phi_dscovr, phi_helios1, phi_helios2, phi_stereoa, phi_stereob, phi_ulysses, phi_wind])
mdot_list = np.array([mdot_ace, mdot_dscovr, mdot_helios1, mdot_helios2, mdot_stereoa, mdot_stereob, mdot_ulysses, mdot_wind])
torque_list = np.array([torque_ace, torque_dscovr, torque_helios1, torque_helios2, torque_stereoa, torque_stereob,
                        torque_ulysses, torque_wind])
colour_list = ['#1f77b4', '#ff7f0e', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#bcbd22', '#17becf']

# for i in range(0, len(time_list)):
#     ax1.scatter(time_list[i], phi_list[i] / (1e22), color=colour_list[i], marker='.', s=1, zorder=1)
#     ax2.scatter(time_list[i], mdot_list[i] / (1e12), color=colour_list[i], marker='.', s=1, zorder=1)
#     ax3.scatter(time_list[i], torque_list[i] / (1e30), color=colour_list[i], marker='.', s=1, zorder=1)
#
# avg_spacecraft_torque = np.mean(np.array([item for sublist in torque_list for item in sublist]))
#
# # Plot OMNI data for open flux, mass loss rate and torque
# ax1.scatter(time_omni_daily, phi_omni_daily / (1e22), c="#006600", marker='.', s=1, zorder=1)
# ax1.scatter(time_omni_hr, phi_omni_hr / (1e22), c="#cc00ff", marker='.', s=1, zorder=1)
#
# ax2.scatter(time_omni_daily, mdot_omni_daily / (1e12), c="#006600", marker='.', s=1, zorder=1)
# ax2.scatter(time_omni_hr, mdot_omni_hr / (1e12), c="#cc00ff", marker='.', s=1, zorder=1)
#
# ax3.scatter(time_omni_daily, torque_omni_daily / (1e30), c="#006600", marker='.', s=1, zorder=1)
# ax3.scatter(time_omni_hr, torque_omni_hr / (1e30), c="#cc00ff", marker='.', s=1, zorder=1)
#
# # First axes: open flux (Owens for both; MCMC uses the equatorial)
# ax1.plot(owens_orig_fltyr, owens_open_f / (1e22), c='#0000b3', ls='-', label="Owens Data")
# ax1.plot(owens_orig_fltyr, owens_equat_open_f / (1e22), c='#00a3cc', ls='-', alpha=0.7, label="Equatorial Owens Data")
#
# ax1.set_ylabel(r'Open Flux [x $10^{22}$ Mx]')
# ax1.legend()
#
# # Mass Loss Rate
# fit = mcmc_fit(owens_orig_phi, owens_orig_vr) - 0.4 * 10**12  # By eye tweaking
# disp_upper = abs(6.37 * 10**(-12) * owens_orig_phi + 7.8 * 10**11) / 3
# disp_lower = abs(9.65 * 10**(-12) * owens_orig_phi - 1.69 * 10**11) / 3
#
#
# fit2 = mcmc_fit2(owens_orig_phi, owens_orig_vr)
# input2 = np.array((phi_omni_daily, vr_omni_daily))
#
# fitfunc2 = lambda x_inputs, a1, a2, a3, a4: a1 * np.array(x_inputs[0])**a2 + a3 * np.array(x_inputs[1])**a4
#
# params, cov = opt.curve_fit(f=fitfunc2, xdata=input2, ydata=mdot_omni_daily, maxfev=10000)


#
# ax2.plot(owens_orig_fltyr, fit / (10**12), color='#cc0000', ls='-', zorder=1, label="MCMC fit")
# ax2.plot(owens_orig_fltyr, (fit + disp_upper) / (10**12), color='#cc0000', ls='--', alpha=0.5, zorder=1)
# ax2.plot(owens_orig_fltyr, (fit - disp_lower) / (10**12), color='#cc0000', ls='--', alpha=0.5, zorder=1)
#
# owens_mdot = np.array([np.sum(4.0 * np.pi * np.cos(owens_lat * np.pi / 180) * np.sin(np.pi / 180 * 5/2.) * owens_massflux[i, :])
#               for i in range(0, len(owens_massflux[:, 0]))])
#
# # 18th index should be equatorial (or near enough, I think it's -2.5 or 2.5 degrees)
# owens_equat_mdot = np.array([4. * np.pi * owens_massflux[i, 18] for i in range(0, len(owens_massflux[:, 0]))])
#
# ax2.plot(owens_orig_fltyr, owens_mdot / (1e12), color='#0000b3', ls='-', label="Mass Flux Fit")
# ax2.plot(owens_orig_fltyr, owens_equat_mdot / (1e12), color='#00a3cc', ls='-', alpha=0.7, label="Equatorial Mass Flux Fit")
#
# ax2.set_ylabel(r'$\dot{M}$ [x $10^{12}$ g s$^{-1}$]')
# ax2.legend()
#
# # Torque
# torque = np.array(stut.calc_solar_torque(fit, owens_open_f))
# torque_upper = np.array(stut.calc_solar_torque(fit + disp_upper, owens_open_f))
# torque_lower = np.array(stut.calc_solar_torque(fit - disp_lower, owens_open_f))
#
# ax3.plot(owens_orig_fltyr, torque / (10**30), color='#cc0000', ls='-', label="MCMC fit", zorder=1)
# ax3.plot(owens_orig_fltyr, torque_upper / (10**30), color='#cc0000', ls='-', alpha=0.5, zorder=1)
# ax3.plot(owens_orig_fltyr, torque_lower / (10**30), color='#cc0000', ls='-', alpha=0.5, zorder=1)
#
# owens_torque = 2.3e30 * (np.asarray(owens_mdot) / 1e12)**0.26 * (np.asarray(owens_open_f) / 8e22)**1.48
#
# ax3.plot(owens_orig_fltyr, owens_torque / (1e30), color='#0000b3', ls='-', label="Mass Flux Fit", zorder=1)
#
# ax3.axhline(y=np.mean(torque / (1e30)), color="grey", ls='-.', label="MCMC Fit Average")
# ax3.axhline(y=np.mean(owens_torque / (1e30)), color="grey", ls='-', label="Mass Flux Fit Average")
# ax3.axhline(y=avg_spacecraft_torque / (1e30), color="grey", ls='--', label="Spacecraft Average")
#
# ax3.errorbar([1983], [3.16], yerr=[[0.65],[0.65]], fmt='o', c='red', lw=3, zorder=0)  # Pizzo errorbar
# ax3.scatter([1983], [3.16], color='red', edgecolor='k', s=50, label="Pizzo et al. (1983)")  # Pizzo calc average
# ax3.scatter([1999], [2.1], color='orange', edgecolor='k', s=50, zorder=5, label="Li (1999)")  # Li calc average
#
# ax3.set_ylabel(r'Torque [x $10^{30}$ erg]')
# ax3.legend()
#
# ax1.set_xlim(np.min(owens_orig_fltyr), np.max(owens_orig_fltyr))
# plt.rcParams.update({'font.size': 6})

# # Test figure to compare spacecraft and OMNI
# fig_test, (ax_br, ax_phi, ax_vr, ax_mdot, ax_torque) = plt.subplots(5, sharex=True)
#
# for i in range(0, len(time_list)):
#     ax_br.scatter(time_list[i], br_list[i], c=colour_list[i], s=1)
#     ax_phi.scatter(time_list[i], phi_list[i], c=colour_list[i], s=1)
#     ax_vr.scatter(time_list[i], vr_list[i], c=colour_list[i], s=1)
#     ax_mdot.scatter(time_list[i], mdot_list[i], c=colour_list[i], s=1)
#     ax_torque.scatter(time_list[i], torque_list[i], c=colour_list[i], s=1)
#
# # ax_br.scatter(time_omni_daily, br_omni_daily, c="#33cc33", s=1)
# ax_br.scatter(time_omni_hr, br_omni_hr, c="#cc00ff", s=1)
# ax_br.set_ylabel(r"|B$_r$| [nT]")
#
# # ax_phi.scatter(time_omni_daily, phi_omni_daily, c="#33cc33", s=1)
# ax_phi.scatter(time_omni_hr, phi_omni_hr, c="#cc00ff", s=1)
# ax_phi.set_ylabel(r"Open Flux [Mx]")
#
# # ax_vr.scatter(time_omni_daily, vr_omni_daily, c="#33cc33", s=1)
# ax_vr.scatter(time_omni_hr, vr_omni_hr, c="#cc00ff", s=1)
# ax_vr.set_ylabel(r"v$_r$ [km s$^{-1}$]")
#
# # ax_mdot.scatter(time_omni_daily, mdot_omni_daily, c="#33cc33", s=1)
# ax_mdot.scatter(time_omni_hr, mdot_omni_hr, c="#cc00ff", s=1)
# ax_mdot.set_ylabel(r"$\dot{M}$ [g s$^{-1}$]")
#
#
# # ax_torque.scatter(time_omni_daily, torque_omni_daily, c="#33cc33", s=1)
# ax_torque.scatter(time_omni_hr, torque_omni_hr, c="#cc00ff", s=1)
# ax_torque.set_ylabel(r"Torque [erg]")
#
# ax_torque.set_xlabel("Year")
#
# plt.tight_layout()

# Single axis plots for presentation
plt.figure()

# Spacecraft data
for i in range(0, len(time_list)):
    plt.scatter(time_list[i], phi_list[i] / 1e22, c=colour_list[i], s=1)


plt.scatter(time_omni_hr, phi_omni_hr / 1e22, c="#cc00ff", s=1)

# plt.plot(owens_orig_fltyr, fit / 1e12, color='#cc0000', ls='-', zorder=1, label="Initial fit")
# # plt.plot(owens_orig_fltyr, fit2 / 1e12, color="#005ce6", ls='-', zorder=1, label="Second Fit")
# plt.plot(owens_orig_fltyr, fitfunc2((owens_orig_phi, owens_orig_vr), params[0], params[1], params[2], params[3])
#          / 1e12, color="#00cc00", ls='-', zorder=1, label="Curve Fit")
#
# print("a1 = " + "%.3f" % params[0])
# print("a2 = " + "%.3f" % params[1])
# print("a3 = " + "%.3f" % params[2])
# print("a4 = " + "%.3f" % params[3])
#
# print("\nMdot = " + "%.3f" % params[0] + " * Phi_open^("  + "%.3f" % params[1] + ") + " +
#       "%.3f" % params[2] + " * v_r^(" + "%.3f" % params[3] + ")")
#
# # plt.plot(owens_orig_fltyr, (fit + disp_upper) / (10**12), color='#cc0000', ls='--', alpha=0.5, zorder=1)
# # plt.plot(owens_orig_fltyr, (fit - disp_lower) / (10**12), color='#cc0000', ls='--', alpha=0.5, zorder=1)
#
# plt.xlabel("Year")
# plt.ylabel(r"Mass Loss Rate [x $10^{12}$ g s$^{-1}$]")
# plt.legend()

owens_open_f = np.array([2. * np.pi * np.sum(np.abs(owens_br_array[i, :] * (1.5e13)**2 * np.cos(owens_lat * np.pi / 180)
                                           * np.sin(0.5 * 5 * np.pi / 180)))
                for i in range(0, len(owens_br_array[:, 0]))])

# Assume isotropy and calculate the open flux from equatorial; I think 18th column is -2.5 or 2.5 degrees (close enough)
# It's 4pi in the surface integral formula; 2pi here because we divide owens br by 2
owens_equat_open_f = np.array(2. * np.pi * (1.5e13)**2 * np.abs(owens_br_array[:, 18]))

plt.plot(owens_orig_fltyr, owens_open_f / (1e22), 'k-', label="Full Latitudinal Data")
plt.plot(owens_orig_fltyr, owens_equat_open_f / (1e22), c='grey', ls='-', label="Equatorial Data")

plt.xlabel("Year")
plt.ylabel(r'Open Flux [x $10^{22}$ Mx]')

plt.legend()
plt.show()
