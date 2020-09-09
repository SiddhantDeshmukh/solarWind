import tools.stellarUtilities as stut
import tools.utilities as util
import pandas as pd
from scipy import optimize as opt
import matplotlib.pyplot as plt
from collections import namedtuple
import numpy as np


class SolarCycle:
    def __init__(self, cycle_num, cycle_data):
        self.__cycle_num = cycle_num
        self.__cycle_data = cycle_data
        self.__cycle_start = cycle_data[0][0]
        self.__cycle_end = cycle_data[-1][0]
        self.__cycle_max = util.get_minmax_val_idx(cycle_data[1], True)
        self.__cycle_min = util.get_minmax_val_idx(cycle_data[1], False)
        self.__cycle_rise_turn = util.get_median(cycle_data[0], trunc_start=0, trunc_end=self.__cycle_max[0])
        self.__cycle_fall_turn = util.get_median(cycle_data[0], trunc_start=self.__cycle_max[0], trunc_end=-1)

        # From initial min to first inflexion
        self.__lower_times = cycle_data[0][0: self.__cycle_rise_turn[0]]

        br = cycle_data[1]
        lower_br = br[0: self.__cycle_rise_turn[0]]
        upper_br = br[self.__cycle_rise_turn[0]: self.__cycle_fall_turn[0]]
        lower_br = np.append(lower_br, br[self.__cycle_fall_turn[0]: -1])

        self.lower_br = lower_br
        self.upper_br = upper_br
        self.__lower_vals = np.array([])
        for i in range(0, self.__cycle_rise_turn[0]):
            for j in range(1, len(cycle_data)):
                self.__lower_vals = np.append(self.__lower_vals, cycle_data[j][i])

        # Between inflexions
        self.__upper_times = cycle_data[0][self.__cycle_rise_turn[0]: self.__cycle_fall_turn[0]]
        self.__upper_vals = np.array([])
        for i in range(self.__cycle_rise_turn[0], self.__cycle_fall_turn[0]):
            for j in range(1, len(cycle_data)):
                self.__upper_vals = np.append(self.__upper_vals, cycle_data[j][i])
        # From second inflexion to end
        self.__lower_times = np.append(self.__lower_times, self.__cycle_data[0][self.__cycle_fall_turn[0]: -1], axis=0)
        for i in range(self.__cycle_rise_turn[0], len(self.__cycle_data)):
            for j in range(1, len(cycle_data)):
                self.__lower_vals = np.append(self.__lower_vals, cycle_data[j][i])

        self.__lower_data = np.array([self.__lower_times, self.__lower_vals]).T
        self.__upper_data = np.array([self.__upper_times, self.__upper_vals]).T

    def get_cycle_num(self):
        return self.__cycle_num

    def get_cycle_data(self):
        return self.__cycle_data

    def get_cycle_start(self):
        return self.__cycle_start

    def get_cycle_end(self):
        return self.__cycle_end

    def get_cycle_max(self):
        return self.__cycle_max

    def get_cycle_min(self):
        return self.__cycle_min

    def get_cycle_rise_turn(self):
        return self.__cycle_rise_turn

    def get_cycle_fall_turn(self):
        return self.__cycle_fall_turn

    def get_lower_times(self):
        return self.__lower_times

    def get_lower_vals(self):
        return self.__lower_vals

    def get_upper_times(self):
        return self.__upper_times

    def get_upper_vals(self):
        return self.__upper_vals

    def get_lower_data(self):
        return self.__lower_data

    def get_upper_data(self):
        return self.__upper_data


owens_dict = pd.read_csv('data/models/owens_equatorial.csv').to_dict('list')
omni_dict = pd.read_csv("data/27_day_avg/OMNI_27day_avg_hr.csv").to_dict("list")

# Owens data
owens_phi = owens_dict.get("Open Flux [Mx] (Owens)")
owens_fltyr = owens_dict.get("Year (Owens)")

# cycles 22, 23, 24 to see if it gets the data properly
omni_time = [yr for i, yr in enumerate(omni_dict.get("Year (OMNI Hr)"))
             if omni_dict.get("Open Flux [Mx] (OMNI Hr)")[i] > 0]
omni_br = [br for i, br in enumerate(omni_dict.get("Radial Magnetic Field Magnitude [nT] (OMNI Hr)"))
           if omni_dict.get("Open Flux [Mx] (OMNI Hr)")[i] > 0]
omni_phi = [phi for i, phi in enumerate(omni_dict.get("Open Flux [Mx] (OMNI Hr)"))
            if omni_dict.get("Open Flux [Mx] (OMNI Hr)")[i] > 0]
omni_mdot = [mdot for i, mdot in enumerate(omni_dict.get("Mass Loss Rate [g s^-1] (OMNI Hr)"))
             if omni_dict.get("Open Flux [Mx] (OMNI Hr)")[i] > 0]

data_array = np.array([omni_time, omni_br, omni_phi, omni_mdot]).T
data_array = sorted(data_array, key=lambda x: x[0])

cycle21_data = np.array([(data[0], data[1], data[2], data[3]) for data in data_array if 1976.25 <= data[0] < 1986.75]).T
cycle22_data = np.array([(data[0], data[1], data[2], data[3]) for data in data_array if 1986.25 <= data[0] < 1996.67]).T
cycle23_data = np.array([(data[0], data[1], data[2], data[3]) for data in data_array if 1996.67 <= data[0] < 2008.9]).T
cycle24_data = np.array([(data[0], data[1], data[2], data[3]) for data in data_array if 2008.9 <= data[0] < 2018.5]).T

cycle21 = SolarCycle(21, cycle21_data)
cycle22 = SolarCycle(22, cycle22_data)
cycle23 = SolarCycle(23, cycle23_data)
cycle24 = SolarCycle(24, cycle24_data)

cycle_list = [cycle21, cycle22, cycle23, cycle24]

# lower_data = np.array([])
#
# for cycle in cycle_list:
#     lower_data = np.append(lower_data, cycle.get_lower_data())
#
# lower_time = []
# lower_phi = []
# lower_mdot = []
#
# for i in range(1, len(lower_data), 2):
#     lower_time.append(lower_data[i-1])
#     lower_phi.append(lower_data[i][1])
#     lower_mdot.append(lower_data[i][2])
#
# # Flatten lists (they're stacked by solar cycle) and remove the erroneous "zeroes"
# lower_phi = [phi for sublist in lower_phi for phi in sublist if phi > 0]
# lower_mdot = [mdot for sublist in lower_mdot for mdot in sublist if mdot > 0]
#
# # Fit the lower data (in particular the mdot to the phi)
# minfit = lambda x, a1, a2: a1 * x**a2
#
# params, _ = opt.curve_fit(minfit, lower_phi, lower_mdot, maxfev=10000)
#
# print("Mdot = " + str(params[0]) + " * phi^(" + str(params[1]) + ")")

# plt.figure()
# # Solar cycle colour; 21, 22, 23, 24
# colour_list = ["#cc0000", "#993399", "#1a53ff", "#008000"]
#
# # for i, cycle in enumerate(cycle_list):
# #     plt.scatter(cycle.get_cycle_data()[2] / 1e22, cycle.get_cycle_data()[3] / 1e12, color=colour_list[i], marker='.',
# #                 label="Solar Cycle 2" + str(i+1))
#
# plt.scatter(np.array(omni_phi) / 1e23, np.array(omni_mdot) / 1e12, c='r', marker='.')
# plt.plot(np.array(omni_phi) / 1e23, np.array(minfit(omni_phi, params[0], params[1])) / 1e12, 'b-')
#
# plt.loglog()
# plt.xlabel(r"$\Phi$$_{open}$ [x$10^{22}$ Mx]")
# plt.ylabel(r"$\dot{M}$ [x$10^{12}$ g s$^{-1}$]")

plt.figure()
for cycle in cycle_list:
    plt.plot(cycle.get_lower_times(), cycle.lower_br, 'r.')
    plt.plot(cycle.get_upper_times(), cycle.upper_br, 'b.')

plt.xlabel("Year")
plt.ylabel(r"B$_r$ [nT]")

# plt.legend()
plt.show()
