from tools import utilities as util
import csv
import numpy as np
import pandas as pd
import tools.stellarUtilities as stut
import pandas as pd


s_in_day = 24 * 3600
ms_in_day = s_in_day * 10**3


# This function seems unnecessarily complicated and niche
def interpolate_spacedata(plasma_times, plasma_data, mag_times, mag_data, plasma_to_mag):
    if plasma_to_mag:  # Interpolate plasma data onto the mag times
        plasma_fits = util.determine_fits(plasma_times, plasma_data)
        fitted_plasma_data = util.evaluate_fits(mag_times, plasma_fits)

        return fitted_plasma_data
    else:  # Interpolate mag data onto the plasma times
        mag_fits = util.determine_fits(mag_times, mag_data)
        fitted_mag_data = util.evaluate_fits(plasma_times, mag_fits)

        return fitted_mag_data


def interpolate_data(interp_times, initial_times, initial_data_list):
    fits = util.determine_fits(initial_times, initial_data_list)
    fitted_data = util.evaluate_fits(interp_times, fits)

    return fitted_data


# Fix times of data into float years
def write_fltyr_data(init_file, fin_file):
    full_data = np.loadtxt(init_file, skiprows=0)
    orig_time = full_data[:, 0:3]
    # other_data = full_data[:, 5]
    br = full_data[:, 12]
    prot_dens = full_data[:, 23]
    vr = full_data[:, 24]
    sn = full_data[:, 39]
    fltyrs = []

    for entry in orig_time:
        # Format day of year into 3 digits
        if len(str(int(entry[1]))) is 1:
            doy = '00' + str(int(entry[1]))
            # print(doy, "single digit")
        elif len(str(int(entry[1]))) is 2:
            doy = '0' + str(int(entry[1]))
            # print(doy, "double digit")
        else:
            doy = str(int(entry[1]))
            # print(doy, "triple digit")
        # if len(str(int(entry[1]))) is 1:
        #     month = '0' + str(int(entry[1]))
        # else:
        #     month = str(int(entry[1]))
        #
        # if len(str(int(entry[2]))) is 1:
        #     day = '0' + str(int(entry[2]))
        # else:
        #     day = str(int(entry[2]))

        # Format hour into 2 digits
        if len(str(int(entry[2]))) < 2:
            hr = '0' + str(int(entry[2]))
        else:
            hr = str((int(entry[2])))

        # mins = str(int(entry[4]))
        # time = str(int(entry[0])) + month + day
        time = str(int(entry[0])) + str(doy) + str(hr)
        print(time)
        fltyr = util.fltyr_from_string(time)
        fltyrs.append(fltyr)
    key_list = ["Year (OMNI)", "Radial Magnetic Field [nT] (OMNI)", "Proton Density [cm^-3] (OMNI)",
                "Radial Wind Velocity [km s^-1] (OMNI)", "Sunspot Number (OMNI)", "Flags (OMNI)"]
    value_list = [fltyrs, br, prot_dens, vr, sn]
    flags = [0] * len(fltyrs)

    # Filter data to remove bad values
    for i in range(0, len(value_list)):
        for j in range(0, len(value_list[i])):
            if value_list[i][j] == 999.9 or value_list[i][j] == 9999 or value_list[i][j] == 999:
                flags[j] = 1  # Flag the value as bad

    value_list.append(flags)

    # Create the dictionary and save to file
    omni_dict = {key: value for key, value in zip(key_list, value_list)}
    pd.DataFrame.from_dict(omni_dict, orient="columns").to_csv(fin_file, index=False)


OMNI_hour_avg_file = "data/OMNI/omni2_all_years.dat"
write_fltyr_data(OMNI_hour_avg_file, "data/OMNI/omni2_hour_avg.csv")
