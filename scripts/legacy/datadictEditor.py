import numpy as np
from tools import utilities as util
from tools import stellarUtilities as stut
import pandas as pd

# Add mass flux to each spacecraft's data
ACE_file = '/Users/sdeshmukh/PycharmProjects/SummerResearchSun/data/27_day_avg/ACE_27day_avg.csv'
DSCOVR_file = '/Users/sdeshmukh/PycharmProjects/SummerResearchSun/data/27_day_avg/DSCOVR_27day_avg.csv'
HELIOS1_file = '/Users/sdeshmukh/PycharmProjects/SummerResearchSun/data/27_day_avg/HELIOS_1_27day_avg.csv'
HELIOS2_file = '/Users/sdeshmukh/PycharmProjects/SummerResearchSun/data/27_day_avg/HELIOS_2_27day_avg.csv'
STEREO_A_file = '/Users/sdeshmukh/PycharmProjects/SummerResearchSun/data/27_day_avg/STEREO_A_27day_avg.csv'
STEREO_B_file = '/Users/sdeshmukh/PycharmProjects/SummerResearchSun/data/27_day_avg/STEREO_B_27day_avg.csv'
ULYSSES_file = '/Users/sdeshmukh/PycharmProjects/SummerResearchSun/data/27_day_avg/ULYSSES_27day_avg.csv'
ULYSSES_FULL_file = '/Users/sdeshmukh/PycharmProjects/SummerResearchSun/data/27_day_avg/ULYSSES_FULL_27day_avg.csv'
WIND_file = '/Users/sdeshmukh/PycharmProjects/SummerResearchSun/data/27_day_avg/WIND_27day_avg.csv'

OMNI_daily_file = "/Users/sdeshmukh/PycharmProjects/SummerResearchSun/data/27_day_avg/OMNI_27day_avg_daily.csv"
OMNI_hr_file = "/Users/sdeshmukh/PycharmProjects/SummerResearchSun/data/27_day_avg/OMNI_27day_avg_hr.csv"

ACE_dict = pd.read_csv(ACE_file).to_dict('list')
DSCOVR_dict = pd.read_csv(DSCOVR_file).to_dict('list')
HELIOS1_dict = pd.read_csv(HELIOS1_file).to_dict('list')
HELIOS2_dict = pd.read_csv(HELIOS2_file).to_dict('list')
STEREO_A_dict = pd.read_csv(STEREO_A_file).to_dict('list')
STEREO_B_dict = pd.read_csv(STEREO_B_file).to_dict('list')
ULYSSES_dict = pd.read_csv(ULYSSES_file).to_dict('list')
ULYSSES_FULL_dict = pd.read_csv(ULYSSES_FULL_file).to_dict('list')
WIND_dict = pd.read_csv(WIND_file).to_dict('list')

OMNI_daily_dict = pd.read_csv(OMNI_daily_file).to_dict('list')
OMNI_hr_dict = pd.read_csv(OMNI_hr_file).to_dict('list')

dummypath = '/Users/sdeshmukh/PycharmProjects/SummerResearchSun/tools/dummydata/'

key1 = "Open Flux [Mx]"
key2 = "Mass Loss Rate [g s^-1]"
key3 = "Radial Wind Velocity [km s^-1]"

spacecraft_identifiers = [" (ACE)", " (DSCOVR)", " (Helios 1)", " (Helios 2)", " (STEREO A)", " (STEREO B)",
                          " (Ulysses)", " (Ulysses Full)", " (WIND)", " (OMNI)", " (OMNI)"]

dict_list = [ACE_dict, DSCOVR_dict, HELIOS1_dict, HELIOS2_dict, STEREO_A_dict, STEREO_B_dict, ULYSSES_dict,
              ULYSSES_FULL_dict, WIND_dict, OMNI_daily_dict, OMNI_hr_dict]

OMNI_daily_dict["Solar Torque [erg] (OMNI)"] = stut.calc_solar_torque(OMNI_daily_dict.get(key2 + " (OMNI)"),
                                                                      OMNI_daily_dict.get(key1 + " (OMNI)"))

OMNI_hr_dict["Solar Torque [erg] (OMNI)"] = stut.calc_solar_torque(OMNI_hr_dict.get(key2 + " (OMNI)"),
                                                                   OMNI_hr_dict.get(key1 + " (OMNI)"))

# pd.DataFrame.from_dict(OMNI_daily_dict, orient='columns').to_csv("OMNI_27day_avg_daily.csv", index=False)
# pd.DataFrame.from_dict(OMNI_hr_dict, orient='columns').to_csv("OMNI_27day_avg_hr.csv", index=False)

# Doesn't work for OMNI daily and hr avg; will just overwrite the first OMNI thing because the identifiers are the same
for i in range(0, len(dict_list)):  # Iterating through the list of spacecraft
    # Do calculations
    # Mass flux
    # rho_v = np.array(dict_list[i].get(key1 + spacecraft_identifiers[i])) * \
    #         np.array(dict_list[i].get(key2 + spacecraft_identifiers[i])) * 10**5

    # Normalised mass flux, convert quantities to cgs first
    # form_dist = np.array(dict_list[i].get(key1 + spacecraft_identifiers[i])) * 1.496 * 10**13
    # vel = np.array(dict_list[i].get(key3 + spacecraft_identifiers[i])) * 10**5
    # norm_rho_v = np.array(dict_list[i].get(key2 + spacecraft_identifiers[i])) * vel * (form_dist**2)

    # Kinetic energy density
    # rho_v2 = np.array(dict_list[i].get(key1 + spacecraft_identifiers[i])) * \
    #          (np.array(dict_list[i].get(key2 + spacecraft_identifiers[i])))**2

    # Normalised kinetic energy density
    # form_dist = np.array(dict_list[i].get(key1 + spacecraft_identifiers[i])) * 1.496 * 10**13
    # norm_ked = form_dist**2 * np.array(dict_list[i].get(key3 + spacecraft_identifiers[i])) * \
    #            np.cos(np.array(dict_list[i].get(key2 + spacecraft_identifiers[i])))

    # Solar torque, i.e. a different calculation specific to the Sun
    sol_tor = stut.calc_solar_torque(dict_list[i].get(key2 + spacecraft_identifiers[i]),
                                     dict_list[i].get(key1 + spacecraft_identifiers[i]))

    # Add columns
    # dict_list[i]["Mass Flux [g cm^-2 s^-1]" + spacecraft_identifiers[i]] = rho_v
    # dict_list[i]["Kinetic Energy Density" + spacecraft_identifiers[i]] = rho_v2
    # dict_list[i]["Normalised Mass Flux [g s^-1]" + spacecraft_identifiers[i]] = norm_rho_v
    # dict_list[i]["Heliospheric Latitude" + spacecraft_identifiers[i]] = 0.
    # dict_list[i]["Lat Normalised Kinetic Energy Density" + spacecraft_identifiers[i]] = norm_ked
    dict_list[i]["Solar Torque [erg]" + spacecraft_identifiers[i]] = sol_tor

    # Remove columns
    # dict_list[i].pop("Normalised Mass Flux" + spacecraft_identifiers[i])

    # Write changes to file
    pd.DataFrame.from_dict(dict_list[i], orient='columns').to_csv(
        dummypath + util.format_filename(util.strip_identifier(spacecraft_identifiers[i], False)) + '_27day_avg.csv', index=False)

