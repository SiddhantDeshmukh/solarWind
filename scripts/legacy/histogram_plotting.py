import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tools.utilities as util
from collections import namedtuple
from tools import stellarUtilities as stut

data_dict = pd.read_csv("data/27_day_avg/ALL_SPACECRAFT_27day_avg.csv").to_dict(orient="list")
owens_orig_dict = pd.read_csv('data/models/owens_equatorial.csv').to_dict('list')
omni_dict = pd.read_csv("data/27_day_avg/OMNI_27day_avg_hr.csv").to_dict("list")

spacecraft_identifiers = ["(ACE)", "(DSCOVR)", "(Helios 1)", "(Helios 2)", "(STEREO A)", "(STEREO B)",
                          "(Ulysses)", "(WIND)"]

key_mdot = "Mass Loss Rate [g s^-1] "
key_time = "Year "

colour_list = ['#1f77b4', '#ff7f0e', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#bcbd22', '#17becf']

# Scaling factor multiplied to entire function!
a = 55  # original: 49.43 +/- 15.77
b = 0.453  # original: 0.274 +/- 0.024
c = 1.25  # original: 0.549 +/- 0.074
scaling = 1

# Lambda for 2 parameter mcmc fit (phi and vr)
mcmc_fit = lambda phi, vr: a * (phi**b + vr**c) * (scaling)

# For histogram; all in one list since we don't need to differentiate by spacecraft here
mdot_list = []
time_list = []

# Read in the Owens data here
owens_time = np.array([time for i, time in enumerate(owens_orig_dict.get("Year (Owens)")) if i >= 348])

owens_orig_phi = np.array([phi for i, phi in enumerate(owens_orig_dict.get("Open Flux [Mx] (Owens)"))
                           if i >= 348]) / 2

owens_orig_vr = np.array([vr * 10**5 for i, vr in enumerate(owens_orig_dict.get("Radial Wind Velocity [km s^-1] (Owens)"))
                          if i >= 348])

owens_dist = np.array([1] * len(owens_time))

owens_mdot = mcmc_fit(owens_orig_phi, owens_orig_vr)

individual_mdot_list = [[] for _ in range(0, len(spacecraft_identifiers))]  # Ordered like spacecraft identifiers (alphabetically)
individual_time_list = [[] for _ in range(0, len(spacecraft_identifiers))]

for i, identifier in enumerate(spacecraft_identifiers):
    for j in range(0, len(data_dict.get(key_mdot + identifier))):
        if not np.isnan(data_dict.get(key_mdot + identifier)[j]):  # Check applies to all values for a given spacecraft
            mdot_list.append(data_dict.get(key_mdot + identifier)[j])
            time_list.append(data_dict.get(key_time + identifier)[j])
            individual_mdot_list[i].append(data_dict.get(key_mdot + identifier)[j] / 1e12)
            individual_time_list[i].append(data_dict.get(key_time + identifier)[j])

omni_mdot_list = np.array([mdot for i, mdot in enumerate(omni_dict.get("Mass Loss Rate [g s^-1] (OMNI Hr)"))])
omni_time_list = np.array([time for i, time in enumerate(omni_dict.get("Year (OMNI Hr)"))])
omni_phi_list = np.array([phi for i, phi in enumerate(omni_dict.get("Open Flux [Mx] (OMNI Hr)"))])
omni_vr_list = np.array([vr for i , vr in enumerate(omni_dict.get("Radial Wind Velocity [km s^-1] (OMNI Hr)"))])

data_array = np.array([time_list, mdot_list]).T
data_array = sorted(data_array, key=lambda x: x[0])

owens_data = np.array((owens_time, owens_mdot)).T
owens_data = sorted(owens_data, key=lambda x: x[0])

omni_data_array = np.array([omni_time_list, omni_mdot_list]).T
omni_data_array = sorted(omni_data_array, key=lambda x: x[0])

# Data split by solar cycle
cycle21_data = np.array([(data[0], data[1]) for data in data_array if data[0] >= 1976.25 and data[0] < 1986.75]).T
cycle22_data = np.array([(data[0], data[1]) for data in data_array if data[0] >= 1986.75 and data[0] < 1996.67]).T
cycle23_data = np.array([(data[0], data[1]) for data in data_array if data[0] >= 1996.67 and data[0] < 2008.9]).T
cycle24_data = np.array([(data[0], data[1]) for data in data_array if data[0] >= 2008.9 and data[0] < 2018.5]).T

# Owens data split by solar cycle
owens_cycle21_data = np.array([(data[0], data[1]) for data in owens_data if data[0] >= 1976.25 and data[0] < 1986.75]).T
owens_cycle22_data = np.array([(data[0], data[1]) for data in owens_data if data[0] >= 1986.75 and data[0] < 1996.67]).T
owens_cycle23_data = np.array([(data[0], data[1]) for data in owens_data if data[0] >= 1996.67 and data[0] < 2008.9]).T
owens_cycle24_data = np.array([(data[0], data[1]) for data in owens_data if data[0] >= 2008.9 and data[0] < 2018.5]).T

# OMNI data split by solar cycle
omni_cycle21_data = np.array([(data[0], data[1]) for data in omni_data_array if data[0] >= 1976.25 and data[0] < 1986.75]).T
omni_cycle22_data = np.array([(data[0], data[1]) for data in omni_data_array if data[0] >= 1986.75 and data[0] < 1996.67]).T
omni_cycle23_data = np.array([(data[0], data[1]) for data in omni_data_array if data[0] >= 1996.67 and data[0] < 2008.9]).T
omni_cycle24_data = np.array([(data[0], data[1]) for data in omni_data_array if data[0] >= 2008.9 and data[0] < 2018.5]).T

# Scale for histogram
mdot_list = np.array(mdot_list) / 1e12
owens_mdot = np.array(owens_mdot) / 1e12
omni_mdot_list = np.array(omni_mdot_list) / 1e12

omni_phi_list = np.array(omni_phi_list) / 1e22

# Time independent percentiles
perc_25 = np.percentile(mdot_list, 25)
perc_50 = np.percentile(mdot_list, 50)
perc_90 = np.percentile(mdot_list, 90)

# Time independent percentiles for Owens
owens_perc_25 = np.percentile(owens_mdot, 25)
owens_perc_50 = np.percentile(owens_mdot, 50)
owens_perc_90 = np.percentile(owens_mdot, 90)

# Time independent percentiles for OMNI
omni_perc_25 = np.percentile(omni_mdot_list, 25)
omni_perc_50 = np.percentile(omni_mdot_list, 50)
omni_perc_90 = np.percentile(omni_mdot_list, 90)

# Percentiles for each solar cycle
SolarCycle = namedtuple("SolarCycle",
                        ["cycle_number", "cycle_start", "cycle_end", "perc_25", "perc_50", "perc_90", "mean"])

cycle21 = SolarCycle(21, 1976.25, 1986.75, np.percentile(cycle21_data[1], 25), np.percentile(cycle21_data[1], 50),
                     np.percentile(cycle21_data[1], 90), np.mean(cycle21_data[1]))
cycle22 = SolarCycle(22, 1986.75, 1996.67, np.percentile(cycle22_data[1], 25), np.percentile(cycle22_data[1], 50),
                     np.percentile(cycle22_data[1], 90), np.mean(cycle22_data[1]))
cycle23 = SolarCycle(23, 1996.67, 2008.9, np.percentile(cycle23_data[1], 25), np.percentile(cycle23_data[1], 50),
                     np.percentile(cycle23_data[1], 90), np.mean(cycle23_data[1]))
cycle24 = SolarCycle(24, 2008.9, 2018.5, np.percentile(cycle24_data[1], 25), np.percentile(cycle24_data[1], 50),
                     np.percentile(cycle24_data[1], 90), np.mean(cycle24_data[1]))

# Percentiles for each solar cycle for owens data
owens_cycle21 = SolarCycle(21, 1976.25, 1986.75, np.percentile(owens_cycle21_data[1], 25), np.percentile(owens_cycle21_data[1], 50),
                     np.percentile(owens_cycle21_data[1], 90), np.mean(owens_cycle21_data[1]))
owens_cycle22 = SolarCycle(22, 1986.75, 1996.67, np.percentile(owens_cycle22_data[1], 25), np.percentile(owens_cycle22_data[1], 50),
                     np.percentile(owens_cycle22_data[1], 90), np.mean(owens_cycle22_data[1]))
owens_cycle23 = SolarCycle(23, 1996.67, 2008.9, np.percentile(owens_cycle23_data[1], 25), np.percentile(owens_cycle23_data[1], 50),
                     np.percentile(owens_cycle23_data[1], 90), np.mean(owens_cycle23_data[1]))
owens_cycle24 = SolarCycle(24, 2008.9, 2018.5, np.percentile(owens_cycle24_data[1], 25), np.percentile(owens_cycle24_data[1], 50),
                     np.percentile(owens_cycle24_data[1], 90), np.mean(owens_cycle24_data[1]))

# Percentiles for each solar cycle for OMNI data
omni_cycle21 = SolarCycle(21, 1976.25, 1986.75, np.percentile(omni_cycle21_data[1], 25), np.percentile(omni_cycle21_data[1], 50),
                     np.percentile(omni_cycle21_data[1], 90), np.mean(omni_cycle21_data[1]))
omni_cycle22 = SolarCycle(22, 1986.75, 1996.67, np.percentile(omni_cycle22_data[1], 25), np.percentile(omni_cycle22_data[1], 50),
                     np.percentile(omni_cycle22_data[1], 90), np.mean(omni_cycle22_data[1]))
omni_cycle23 = SolarCycle(23, 1996.67, 2008.9, np.percentile(omni_cycle23_data[1], 25), np.percentile(omni_cycle23_data[1], 50),
                     np.percentile(omni_cycle23_data[1], 90), np.mean(omni_cycle21_data[1]))
omni_cycle24 = SolarCycle(24, 2008.9, 2018.5, np.percentile(omni_cycle24_data[1], 25), np.percentile(omni_cycle24_data[1], 50),
                     np.percentile(omni_cycle24_data[1], 90), np.mean(omni_cycle24_data[1]))

cycle_list = [cycle21, cycle22, cycle23, cycle24]
owens_cycle_list = [owens_cycle21, owens_cycle22, owens_cycle23, owens_cycle24]
omni_cycle_list = [omni_cycle21, omni_cycle22, omni_cycle23, omni_cycle24]


text_colours = ["#ac00e6", "#0066ff", "#2eb82e"]
owens_colours = ["#a31aff", "#0099ff", "#33cc33"]
omni_colours = ["#a31aff", "#0099ff", "#33cc33"]

bins = [0., 0.5, 0.75, 1., 1.25, 1.5, 1.75, 2., 2.25, 2.5, 3.]

# 1D
# Plot cumulative histogram
plt.figure()
y, x, _ = plt.hist(mdot_list, bins=bins, histtype="step", stacked=True, fill=False, color="black", label="All spacecraft")

# Plot individual histograms
for i, data in enumerate(individual_mdot_list):
    plt.hist(data, bins=bins, histtype="step", stacked=True, fill=False, color=colour_list[i],
             label=util.strip_identifier(spacecraft_identifiers[i], False))

# Plot percentile lines and add text
plt.vlines([perc_25, perc_50, perc_90], ymin=0, ymax=y.max(), colors=text_colours)

plt.text(perc_25, y.max(), "%.3f" % perc_25, rotation=0, horizontalalignment="center", verticalalignment="bottom",
         color=text_colours[0])
plt.text(perc_25, y.max() + 20, "25th", rotation=0, horizontalalignment="center", verticalalignment="bottom",
         color=text_colours[0])
plt.text(perc_50, y.max(), "%.3f" % perc_50, rotation=0, horizontalalignment="center", verticalalignment="bottom",
         color=text_colours[1])
plt.text(perc_50, y.max() + 20, "50th", rotation=0, horizontalalignment="center", verticalalignment="bottom",
         color=text_colours[1])
plt.text(perc_90, y.max(), "%.3f" % perc_90, rotation=0, horizontalalignment="center", verticalalignment="bottom",
         color=text_colours[2])
plt.text(perc_90, y.max() + 20, "90th", rotation=0, horizontalalignment="center", verticalalignment="bottom",
         color=text_colours[2])

plt.xlabel(r"Mass Loss Rate [x $10^{12}$ g s$^{-1}$]")
plt.ylabel("Frequency")

# Owens 1D histogram
# plt.figure()
# yp, xp, _ = plt.hist(owens_mdot, bins=5, histtype="step", stacked=True, fill=False, color="#ff3333")
#
#
# plt.vlines([owens_perc_25, owens_perc_50, owens_perc_90], ymin=0, ymax=yp.max(), colors=text_colours)
# plt.text(owens_perc_25, yp.max(), "%.3f" % owens_perc_25, rotation=0, horizontalalignment="center", verticalalignment="bottom",
#          color=owens_colours[0])
# plt.text(owens_perc_25, yp.max() + 5, "25th", rotation=0, horizontalalignment="center", verticalalignment="bottom",
#          color=owens_colours[0])
# plt.text(owens_perc_50, yp.max(), "%.3f" % owens_perc_50, rotation=0, horizontalalignment="center", verticalalignment="bottom",
#          color=owens_colours[1])
# plt.text(owens_perc_50, yp.max() + 5, "50th", rotation=0, horizontalalignment="center", verticalalignment="bottom",
#          color=owens_colours[1])
# plt.text(owens_perc_90, yp.max(), "%.3f" % owens_perc_90, rotation=0, horizontalalignment="center", verticalalignment="bottom",
#          color=owens_colours[2])
# plt.text(owens_perc_90, yp.max() + 5, "90th", rotation=0, horizontalalignment="center", verticalalignment="bottom",
#          color=owens_colours[2])
#
#
# plt.xlabel(r"Mass Loss Rate [x $10^{12}$ g s$^{-1}$]")
# plt.ylabel("Frequency")
plt.legend()

# Plot 1D OMNI histogram
# plt.figure()

# yp2, xp2, _ = plt.hist(omni_mdot_list, bins=bins, histtype="step", stacked=True, fill=False, color="grey", label="OMNI")


# plt.vlines([omni_perc_25, omni_perc_50, omni_perc_90], ymin=0, ymax=yp2.max(), colors=text_colours)
# plt.text(omni_perc_25, yp2.max(), "%.3f" % omni_perc_25, rotation=0, horizontalalignment="center", verticalalignment="bottom",
#          color=omni_colours[0])
# plt.text(omni_perc_25, yp2.max() + 25, "25th", rotation=0, horizontalalignment="center", verticalalignment="bottom",
#          color=omni_colours[0])
# plt.text(omni_perc_50, yp2.max(), "%.3f" % omni_perc_50, rotation=0, horizontalalignment="center", verticalalignment="bottom",
#          color=omni_colours[1])
# plt.text(omni_perc_50, yp2.max() + 25, "50th", rotation=0, horizontalalignment="center", verticalalignment="bottom",
#          color=omni_colours[1])
# plt.text(omni_perc_90, yp2.max(), "%.3f" % omni_perc_90, rotation=0, horizontalalignment="center", verticalalignment="bottom",
#          color=omni_colours[2])
# plt.text(omni_perc_90, yp2.max() + 25, "90th", rotation=0, horizontalalignment="center", verticalalignment="bottom",
#          color=omni_colours[2])


# plt.xlabel(r"Mass Loss Rate [x $10^{12}$ g s$^{-1}$]")
# plt.ylabel("Frequency")
# plt.yscale('log')
# plt.legend(bbox_to_anchor=(0.69, 0.4))

# 2D histogram for Mdot and time
# plt.figure()
#
# plt.hist2d(time_list, mdot_list, range=[[1963, 2018], [0, 3.]], bins=[[1976.25, 1986.75, 1996.67, 2008.9, 2018.5], 10],
#            cmap="Reds", zorder=1)
#
# # Add percentile lines & to each solar cycle
#
# for i, cycle in enumerate(cycle_list):
#     plt.hlines([cycle.perc_25 / 1e12, cycle.perc_50 / 1e12, cycle.perc_90 / 1e12], cycle.cycle_start, cycle.cycle_end,
#                colors=text_colours, zorder=2)
#     plt.text((cycle.cycle_start + cycle.cycle_end) / 2, cycle.perc_25 / 1e12, "%.3f" % (cycle.perc_25 / 1e12), color=text_colours[0],
#              horizontalalignment="center", verticalalignment="bottom", zorder=3)
#     plt.text((cycle.cycle_start + cycle.cycle_end) / 2, cycle.perc_50 / 1e12, "%.3f" % (cycle.perc_50 / 1e12), color=text_colours[1],
#              horizontalalignment="center", verticalalignment="bottom", zorder=3)
#     plt.text((cycle.cycle_start + cycle.cycle_end) / 2, cycle.perc_90 / 1e12, "%.3f" % (cycle.perc_90 / 1e12), color=text_colours[2],
#              horizontalalignment="center", verticalalignment="bottom", zorder=3)
#
# plt.text(2020, cycle24.perc_25, "25th", color=text_colours[0], fontsize=8,
#          horizontalalignment="center", verticalalignment="bottom", zorder=3)
# plt.text(2020, cycle24.perc_50, "50th", color=text_colours[1], fontsize=8,
#          horizontalalignment="center", verticalalignment="bottom", zorder=3)
# plt.text(2020, cycle24.perc_90, "90th", color=text_colours[2], fontsize=8,
#          horizontalalignment="center", verticalalignment="bottom", zorder=3)
#
# # Add text for solar cycles and divider lines
# plt.vlines([cycle21.cycle_end, cycle22.cycle_end, cycle23.cycle_end], ymin=0, ymax=3., color="grey")
#
# plt.text((cycle21.cycle_start + cycle21.cycle_end) / 2, 3., "Cycle 21", color="#595959",
#          horizontalalignment="center", verticalalignment="bottom", zorder=3)
# plt.text((cycle21.cycle_start + cycle21.cycle_end) / 2, 3.2, "Mean: " + "%.3f" % (cycle21.mean/1e12), color="#4d4dff",
#          horizontalalignment="center", verticalalignment="bottom")
#
# plt.text((cycle22.cycle_start + cycle22.cycle_end) / 2, 3., "Cycle 22", color="#595959",
#          horizontalalignment="center", verticalalignment="bottom", zorder=3)
# plt.text((cycle22.cycle_start + cycle22.cycle_end) / 2, 3.2, "Mean: " + "%.3f" % (cycle22.mean/1e12), color="#4d4dff",
#          horizontalalignment="center", verticalalignment="bottom")
#
# plt.text((cycle23.cycle_start + cycle23.cycle_end) / 2, 3., "Cycle 23", color="#595959",
#          horizontalalignment="center", verticalalignment="bottom", zorder=3)
# plt.text((cycle23.cycle_start + cycle23.cycle_end) / 2, 3.2, "Mean: " + "%.3f" % (cycle23.mean/1e12), color="#4d4dff",
#          horizontalalignment="center", verticalalignment="bottom")
#
# plt.text((cycle24.cycle_start + cycle24.cycle_end) / 2, 3., "Cycle 24", color="#595959",
#          horizontalalignment="center", verticalalignment="bottom", zorder=3)
# plt.text((cycle24.cycle_start + cycle24.cycle_end) / 2, 3.2, "Mean: " + "%.3f" % (cycle24.mean/1e12), color="#4d4dff",
#          horizontalalignment="center", verticalalignment="bottom")
#
# plt.colorbar(label="Frequency")
# plt.xlabel("Year")
# plt.ylabel(r"Mass Loss Rate [x $10^{12}$ g s$^{-1}$]")
#
#
# # 2D histogram for Owens data
# plt.figure()
#
# plt.hist2d(owens_time, owens_mdot, range=[[1963, 2018], [0, 3.]], bins=[[1976.25, 1986.75, 1996.67, 2008.9, 2018.5], 5],
#            cmap="Reds", zorder=1)
#
# for i, cycle in enumerate(owens_cycle_list):
#     plt.hlines([cycle.perc_25 / 1e12, cycle.perc_50 / 1e12, cycle.perc_90 / 1e12], cycle.cycle_start, cycle.cycle_end,
#                colors=owens_colours, zorder=2)
#     plt.text((cycle.cycle_start + cycle.cycle_end) / 2, cycle.perc_25 / 1e12, "%.3f" % (cycle.perc_25 / 1e12), color=owens_colours[0],
#              horizontalalignment="center", verticalalignment="top", zorder=3)
#     plt.text((cycle.cycle_start + cycle.cycle_end) / 2, cycle.perc_50 / 1e12, "%.3f" % (cycle.perc_50 / 1e12), color=owens_colours[1],
#              horizontalalignment="center", verticalalignment="bottom", zorder=3)
#     plt.text((cycle.cycle_start + cycle.cycle_end) / 2, cycle.perc_90 / 1e12, "%.3f" % (cycle.perc_90 / 1e12), color=owens_colours[2],
#              horizontalalignment="center", verticalalignment="bottom", zorder=3)
#
# plt.text(2020, owens_cycle24.perc_25, "25th", color=owens_colours[0], fontsize=8,
#          horizontalalignment="center", verticalalignment="bottom", zorder=3)
# plt.text(2020, owens_cycle24.perc_50, "50th", color=owens_colours[1], fontsize=8,
#          horizontalalignment="center", verticalalignment="bottom", zorder=3)
# plt.text(2020, owens_cycle24.perc_90, "90th", color=owens_colours[2], fontsize=8,
#          horizontalalignment="center", verticalalignment="bottom", zorder=3)
#
# # Add text for solar cycles and divider lines
# plt.vlines([owens_cycle21.cycle_end, owens_cycle22.cycle_end, owens_cycle23.cycle_end], ymin=0, ymax=3., color="grey")
# plt.text((owens_cycle21.cycle_start + owens_cycle21.cycle_end) / 2, 3., "Cycle 21", color="#595959",
#          horizontalalignment="center", verticalalignment="bottom", zorder=3)
# plt.text((owens_cycle21.cycle_start + owens_cycle21.cycle_end) / 2, 3.2, "Mean: " + "%.3f" % (owens_cycle21.mean/1e12), color="#4d4dff",
#          horizontalalignment="center", verticalalignment="bottom")
#
# plt.text((owens_cycle22.cycle_start + owens_cycle22.cycle_end) / 2, 3., "Cycle 22", color="#595959",
#          horizontalalignment="center", verticalalignment="bottom", zorder=3)
# plt.text((owens_cycle22.cycle_start + owens_cycle22.cycle_end) / 2, 3.2, "Mean: " + "%.3f" % (owens_cycle22.mean/1e12), color="#4d4dff",
#          horizontalalignment="center", verticalalignment="bottom")
#
# plt.text((owens_cycle23.cycle_start + owens_cycle23.cycle_end) / 2, 3., "Cycle 23", color="#595959",
#          horizontalalignment="center", verticalalignment="bottom", zorder=3)
# plt.text((owens_cycle23.cycle_start + owens_cycle23.cycle_end) / 2, 3.2, "Mean: " + "%.3f" % (owens_cycle23.mean/1e12), color="#4d4dff",
#          horizontalalignment="center", verticalalignment="bottom")
#
# plt.text((owens_cycle24.cycle_start + owens_cycle24.cycle_end) / 2, 3., "Cycle 24", color="#595959",
#          horizontalalignment="center", verticalalignment="bottom", zorder=3)
# plt.text((owens_cycle24.cycle_start + owens_cycle24.cycle_end) / 2, 3.2, "Mean: " + "%.3f" % (owens_cycle24.mean/1e12), color="#4d4dff",
#          horizontalalignment="center", verticalalignment="bottom")
#
# plt.colorbar(label="Frequency")
# plt.xlabel("Year")
# plt.ylabel(r"Mass Loss Rate [x $10^{12}$ g s$^{-1}$]")
#
# # Plot 2D OMNI histogram
# plt.figure()
#
# plt.hist2d(omni_time_list, omni_mdot_list, range=[[1963, 2018], [0, 3.]],
#            bins=[[1976.25, 1986.75, 1996.67, 2008.9, 2018.5], 10], cmap="Reds", zorder=1,)
#
# for i, cycle in enumerate(omni_cycle_list):
#     plt.hlines([cycle.perc_25 / 1e12, cycle.perc_50 / 1e12, cycle.perc_90 / 1e12], cycle.cycle_start, cycle.cycle_end,
#                colors=omni_colours, zorder=2)
#     plt.text((cycle.cycle_start + cycle.cycle_end) / 2, cycle.perc_25 / 1e12, "%.3f" % (cycle.perc_25 / 1e12), color=omni_colours[0],
#              horizontalalignment="center", verticalalignment="top", zorder=3)
#     plt.text((cycle.cycle_start + cycle.cycle_end) / 2, cycle.perc_50 / 1e12, "%.3f" % (cycle.perc_50 / 1e12), color=omni_colours[1],
#              horizontalalignment="center", verticalalignment="bottom", zorder=3)
#     plt.text((cycle.cycle_start + cycle.cycle_end) / 2, cycle.perc_90 / 1e12, "%.3f" % (cycle.perc_90 / 1e12), color=omni_colours[2],
#              horizontalalignment="center", verticalalignment="bottom", zorder=3)
#
# plt.text(2020, omni_cycle24.perc_25, "25th", color=omni_colours[0], fontsize=8,
#          horizontalalignment="center", verticalalignment="bottom", zorder=3)
# plt.text(2020, omni_cycle24.perc_50, "50th", color=omni_colours[1], fontsize=8,
#          horizontalalignment="center", verticalalignment="bottom", zorder=3)
# plt.text(2020, omni_cycle24.perc_90, "90th", color=omni_colours[2], fontsize=8,
#          horizontalalignment="center", verticalalignment="bottom", zorder=3)
#
# # Add text for solar cycles and divider lines
# plt.vlines([omni_cycle21.cycle_end, omni_cycle22.cycle_end, omni_cycle23.cycle_end], ymin=0, ymax=3.0, color="grey")
# plt.text((omni_cycle21.cycle_start + omni_cycle21.cycle_end) / 2, 3., "Cycle 21", color="#595959",
#          horizontalalignment="center", verticalalignment="bottom", zorder=3)
# plt.text((omni_cycle21.cycle_start + omni_cycle21.cycle_end) / 2, 3.2, "Mean: " + "%.3f" % (omni_cycle21.mean/1e12), color="#4d4dff",
#          horizontalalignment="center", verticalalignment="bottom")
#
# plt.text((omni_cycle22.cycle_start + omni_cycle22.cycle_end) / 2, 3., "Cycle 22", color="#595959",
#          horizontalalignment="center", verticalalignment="bottom", zorder=3)
# plt.text((omni_cycle22.cycle_start + omni_cycle22.cycle_end) / 2, 3.2, "Mean: " + "%.3f" % (omni_cycle22.mean/1e12), color="#4d4dff",
#          horizontalalignment="center", verticalalignment="bottom")
#
# plt.text((omni_cycle23.cycle_start + omni_cycle23.cycle_end) / 2, 3., "Cycle 23", color="#595959",
#          horizontalalignment="center", verticalalignment="bottom", zorder=3)
# plt.text((omni_cycle23.cycle_start + omni_cycle23.cycle_end) / 2, 3.2, "Mean: " + "%.3f" % (omni_cycle23.mean/1e12), color="#4d4dff",
#          horizontalalignment="center", verticalalignment="bottom")
#
# plt.text((omni_cycle24.cycle_start + omni_cycle24.cycle_end) / 2, 3., "Cycle 24", color="#595959",
#          horizontalalignment="center", verticalalignment="bottom", zorder=3)
# plt.text((omni_cycle24.cycle_start + omni_cycle24.cycle_end) / 2, 3.2, "Mean: " + "%.3f" % (omni_cycle24.mean/1e12), color="#4d4dff",
#          horizontalalignment="center", verticalalignment="bottom")
#
# plt.colorbar(label="Frequency")
# plt.xlabel("Year")
# plt.ylabel(r"Mass Loss Rate [x $10^{12}$ g s$^{-1}$]")

# Test histograms of mdot vs phi and mdot vs vr
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)

ax1.hist2d(omni_phi_list, omni_mdot_list, bins=[[2.5, 5., 7.5, 10., 12.5, 15., 17.5], 10])

# fig.colorbar(ax=ax1, label="Frequency")
ax1.set_xlabel("Open Flux")
ax1.set_ylabel("Mass Loss Rate")

ax2.hist2d(omni_vr_list, omni_mdot_list, bins=[[250, 300, 400, 500, 550, 650, 700, 750], 10])

# fig.colorbar(ax=ax2, label="Frequency")
ax2.set_xlabel("Radial Wind Velocity")
ax2.set_ylabel("Mass Loss Rate")

ax3.hist2d(omni_vr_list, omni_phi_list, bins=[[250, 300, 400, 500, 550, 650, 700, 750],
                                              [2.5, 5., 7.5, 10., 12.5, 15., 17.5]])
ax3.set_xlabel("Radial Wind Velocity")
ax3.set_ylabel("Open Flux")

ax4.hist2d((omni_phi_list * omni_vr_list), omni_mdot_list,  bins=[[0, 2000, 4000, 6000, 8000, 10000], 10])
ax4.set_xlabel("vr * phi")
ax4.set_ylabel("Mass Loss Rate")

plt.show()
