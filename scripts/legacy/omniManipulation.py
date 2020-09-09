import numpy as np
import pandas as pd
import tools.utilities as util
import tools.stellarUtilities as stut

# OMNI_hr_file = "data/OMNI/OMNI_hr_avg.csv"
OMNI_27day_hourly_file = "data/OMNI/OMNI_27day_hravg.csv"
OMNI_27day_daily_file = "data/OMNI/OMNI_27day_dayavg.csv"

# OMNI_hr_dict = pd.read_csv(OMNI_hr_file).to_dict("list")
OMNI_27day_hr_dict = pd.read_csv(OMNI_27day_hourly_file).to_dict("list")
OMNI_27day_daily_dict = pd.read_csv(OMNI_27day_daily_file).to_dict("list")

# OMNI_orig_hravg_file = "data/OMNI/omni2_hour_avg.csv"
# OMNI_orig_dict = pd.read_csv(OMNI_orig_hravg_file).to_dict("list")

# Add the distance and heliospheric latitude
# OMNI_orig_dict["Distance [AU] (OMNI)"] = [1] * len(OMNI_orig_dict.get("Year (OMNI)"))
# OMNI_orig_dict["Heliospheric Latitude (OMNI)"] = [0] * len(OMNI_orig_dict.get("Year (OMNI)"))

# # Just check how many good and bad data points there are in the hourly averaged set
# bad_points = 0
# good_points = 0
# flags = OMNI_hr_dict.get("Flags (OMNI)")
#
# for i in range(0, len(flags)):
#     if flags[i] == 1:
#         bad_points += 1
#     else:
#         good_points += 1
#
# print("Total: " + str(len(flags)) + ", Bad: " + str(bad_points) + ", Good: " + str(good_points))


def bin_data(bin_size, data_dict, key_list, outfile_name):
    bin_size *= 24
    bin_size = int(bin_size)
    full_data = []

    flags = data_dict.get("Flags (OMNI)")

    for key in key_list:
        print(key, len(data_dict.get(key)))
        if key == "Radial Magnetic Field [nT] (OMNI)":
            full_data.append(np.abs(np.array(data_dict.get(key))))
        else:
            full_data.append(np.array(data_dict.get(key)))

    avgd_data = [[] for _ in range(0, len(full_data))]
    print(len(full_data), len(avgd_data))

    # Bin OMNI data
    for index, element in enumerate(full_data):
        if index is not 0:
            avgd_data[0], avgd_data[index], avgd_flags = util.binned_average_flag(full_data[0], full_data[index], bin_size, flags)

    key_list.append("Flags (OMNI)")
    print(avgd_flags)
    avgd_data.append(avgd_flags)

    avgd_dict = {key: value for key, value in zip(key_list, avgd_data)}
    pd.DataFrame.from_dict(avgd_dict, orient="columns").to_csv(outfile_name, index=False)


key_list = ["Year (OMNI)", "Distance [AU] (OMNI)", "Heliospheric Latitude (OMNI)", "Radial Magnetic Field [nT] (OMNI)",
            "Proton Density [cm^-3] (OMNI)", "Radial Wind Velocity [km s^-1] (OMNI)", "Sunspot Number (OMNI)"]

# # Need to bin twice, once for hourly -> 27-day, once for hourly -> daily (save) -> 27-day
# bin_data(27, OMNI_hr_dict, key_list, "data/OMNI/OMNI_27day_hravg.csv")
# bin_data(1, OMNI_hr_dict, key_list, "data/OMNI/OMNI_1day_avg.csv")

# OMNI_day_file = "data/OMNI/OMNI_1day_avg.csv"
# OMNI_day_dict = pd.read_csv(OMNI_day_file).to_dict("list")
#
# bin_data(27 / 24, OMNI_day_dict, key_list, "data/OMNI/OMNI_27day_dayavg.csv")

# Add mass density to the 27-day averages
m_p = 1.67e-24  # Mass of proton in grams
OMNI_27day_daily_dict["Mass Density [g cm^-3] (OMNI Daily)"] = m_p * np.array(
    OMNI_27day_daily_dict.get("Proton Density [cm^-3] (OMNI Daily)"))

OMNI_27day_hr_dict["Mass Density [g cm^-3] (OMNI Hr)"] = m_p * np.array(
    OMNI_27day_hr_dict.get("Proton Density [cm^-3] (OMNI Hr)"))

# Add mass flux
OMNI_27day_daily_dict["Mass Flux [g cm^-2 s^-1] (OMNI Daily)"] = np.array(
    OMNI_27day_daily_dict.get("Mass Density [g cm^-3] (OMNI Daily)")) * np.array(
    OMNI_27day_daily_dict.get("Radial Wind Velocity [km s^-1] (OMNI Daily)")) * 1e5

OMNI_27day_hr_dict["Mass Flux [g cm^-2 s^-1] (OMNI Hr)"] = np.array(
    OMNI_27day_hr_dict.get("Mass Density [g cm^-3] (OMNI Hr)")) * np.array(
    OMNI_27day_hr_dict.get("Radial Wind Velocity [km s^-1] (OMNI Hr)")) * 1e5

# Add normalised mass flux
OMNI_27day_daily_dict["Normalised Mass Flux [g s^-1] (OMNI Daily)"] = np.array(
    OMNI_27day_daily_dict.get("Mass Flux [g cm^-2 s^-1] (OMNI Daily)")) * (np.array(
    OMNI_27day_daily_dict.get("Distance [AU] (OMNI Daily)")) * 1.5e13)**2

OMNI_27day_hr_dict["Normalised Mass Flux [g s^-1] (OMNI Hr)"] = np.array(
    OMNI_27day_hr_dict.get("Mass Flux [g cm^-2 s^-1] (OMNI Hr)")) * (np.array(
    OMNI_27day_hr_dict.get("Distance [AU] (OMNI Hr)")) * 1.5e13)**2

# Add open flux
OMNI_27day_daily_dict["Open Flux [Mx] (OMNI Daily)"] = stut.calc_open_flux(OMNI_27day_daily_dict.get("Distance [AU] (OMNI Daily)"),
                                               OMNI_27day_daily_dict.get("Radial Magnetic Field Magnitude [nT] (OMNI Daily)"))
OMNI_27day_hr_dict["Open Flux [Mx] (OMNI Hr)"] = stut.calc_open_flux(OMNI_27day_hr_dict.get("Distance [AU] (OMNI Hr)"),
                                               OMNI_27day_hr_dict.get("Radial Magnetic Field Magnitude [nT] (OMNI Hr)"))

# Add mass loss rate
OMNI_27day_daily_dict["Mass Loss Rate [g s^-1] (OMNI Daily)"] = stut.calc_mass_loss_rate(
    OMNI_27day_daily_dict.get("Distance [AU] (OMNI Daily)"),
    OMNI_27day_daily_dict.get("Radial Wind Velocity [km s^-1] (OMNI Daily)"),
    OMNI_27day_daily_dict.get("Mass Density [g cm^-3] (OMNI Daily)"))

OMNI_27day_hr_dict["Mass Loss Rate [g s^-1] (OMNI Hr)"] = stut.calc_mass_loss_rate(
    OMNI_27day_hr_dict.get("Distance [AU] (OMNI Hr)"),
    OMNI_27day_hr_dict.get("Radial Wind Velocity [km s^-1] (OMNI Hr)"),
    OMNI_27day_hr_dict.get("Mass Density [g cm^-3] (OMNI Hr)"))

# Add wind magnetisation
OMNI_27day_daily_dict["Wind Magnetisation (OMNI Daily)"] = stut.calc_wind_magnetisation(
    OMNI_27day_daily_dict.get("Open Flux [Mx] (OMNI Daily)"), OMNI_27day_daily_dict.get("Mass Loss Rate [g s^-1] (OMNI Daily)"))

OMNI_27day_hr_dict["Wind Magnetisation (OMNI Hr)"] = stut.calc_wind_magnetisation(
    OMNI_27day_hr_dict.get("Open Flux [Mx] (OMNI Hr)"), OMNI_27day_hr_dict.get("Mass Loss Rate [g s^-1] (OMNI Hr)"))

# Add relative alfven radius
OMNI_27day_daily_dict["Relative Alfven Radius [Solar Units] (OMNI Daily)"] = stut.calc_rel_alfven_rad(
    OMNI_27day_daily_dict.get("Wind Magnetisation (OMNI Daily)"))

OMNI_27day_hr_dict["Relative Alfven Radius [Solar Units] (OMNI Hr)"] = stut.calc_rel_alfven_rad(
    OMNI_27day_hr_dict.get("Wind Magnetisation (OMNI Hr)"))

# Add torque (THERE IS A BUG SOMEWHERE IN THE TORQUE CALCULATION, USE SOLAR TORQUE INSTEAD)
OMNI_27day_daily_dict["Torque [erg] (OMNI Daily)"] = stut.calc_torque(
    OMNI_27day_daily_dict.get("Mass Loss Rate [g s^-1] (OMNI Daily)"),
    OMNI_27day_daily_dict.get("Relative Alfven Radius [Solar Units] (OMNI Daily)"))

OMNI_27day_hr_dict["Torque [erg] (OMNI Hr)"] = stut.calc_torque(
    OMNI_27day_hr_dict.get("Mass Loss Rate [g s^-1] (OMNI Hr)"),
    OMNI_27day_hr_dict.get("Relative Alfven Radius [Solar Units] (OMNI Hr)"))

# Add solar torque
OMNI_27day_daily_dict["Solar Torque [erg] (OMNI Daily)"] = stut.calc_solar_torque(
    OMNI_27day_daily_dict.get("Mass Loss Rate [g s^-1] (OMNI Daily)"),
    OMNI_27day_daily_dict.get("Open Flux [Mx] (OMNI Daily)"))

OMNI_27day_hr_dict["Solar Torque [erg] (OMNI Hr)"] = stut.calc_solar_torque(
    OMNI_27day_hr_dict.get("Mass Loss Rate [g s^-1] (OMNI Hr)"),
    OMNI_27day_hr_dict.get("Open Flux [Mx] (OMNI Hr)"))

# Write to file
pd.DataFrame.from_dict(OMNI_27day_daily_dict, orient="columns").to_csv("data/OMNI/OMNI_27day_avg_daily.csv", index=False)
pd.DataFrame.from_dict(OMNI_27day_hr_dict, orient="columns").to_csv("data/OMNI/OMNI_27day_avg_hr.csv", index=False)
