import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import csv
from tools import utilities as util, stellarUtilities as stut
import pandas as pd

np.set_printoptions(threshold=np.nan)

# Declare constants in cgs
m_p = 1.67 * 10**(-24)  # mass of proton in grams
m_alpha = 6.64 * 10**(-24)  # mass of alpha particle in grams

# Import Ulysses data
SWOOPS_file = './data/ulysses/swoops_hourav.dat'
ULY_MAG_file = './data/ulysses/uly_fitted_mag.csv'

SWOOPS_yr, SWOOPS_doy, SWOOPS_hr, SWOOPS_min, SWOOPS_sec, ULY_dist, ULY_hlat, ULY_hlong, ULY_prot_dens, \
    ULY_alpha_dens, ULY_temp_l, ULY_temp_s, ULY_vr, ULY_vt, ULY_vn = np.loadtxt(SWOOPS_file, unpack=True)

print("Done loading Ulysses SWOOPS data")

ULY_br = []
ULY_bt = []
ULY_bn = []
ULY_bmag = []

with open(ULY_MAG_file, 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in reader:
        ULY_br.append(abs(float(row[0])))
        ULY_bt.append(float(row[1]))
        ULY_bn.append(float(row[2]))
        ULY_bmag.append(float(row[3]))

print("Done loading Ulysses MAG data")


# Import ACE data
SWEPAM_file = './data/ace/ace_swepam_data.txt'
ACE_MAG_file = './data/ace/ace_mag_data.txt'

ACE_fltyr, ACE_prot_dens, ACE_prot_temp, ACE_alpha_prot_ratio, ACE_proton_speed, ACE_vr, ACE_vt, ACE_vn, ACE_dist = \
    np.loadtxt(SWEPAM_file, unpack=True)
print("Done loading ACE SWEPAM data")

ACE_br = np.loadtxt(ACE_MAG_file, unpack=True)
print("Done loading ACE MAG data")

dummy_ACE_data = [ACE_prot_dens, ACE_prot_temp, ACE_alpha_prot_ratio, ACE_proton_speed, ACE_vr, ACE_vt,
                  ACE_vn, ACE_dist, ACE_br]
# Check for invalid values and change them to '1'
for i in range(0, len(dummy_ACE_data)):
    for j in range(0, len(dummy_ACE_data[i])):
        if '999' in str(dummy_ACE_data[i][j])[0:5]:
            dummy_ACE_data[i][j] = 1


# Import DSCOVR data
DSCOVR_file = './data/dscovr/dscovr_hravg_fix.csv'

DSCOVR_fltyr, DSCOVR_bmag, DSCOVR_br, DSCOVR_prot_speed, DSCOVR_vx, DSCOVR_vy, DSCOVR_vz, DSCOVR_prot_dens, \
    DSCOVR_prot_temp, DSCOVR_dist = np.loadtxt(DSCOVR_file, delimiter=',', unpack=True)
print("Done loading DSCOVR data")

dummy_DSCOVR_data = [DSCOVR_bmag, DSCOVR_br, DSCOVR_prot_speed, DSCOVR_vx, DSCOVR_vy, DSCOVR_vz, DSCOVR_prot_dens,
                     DSCOVR_prot_temp, DSCOVR_dist]

# Check for invalid values and change them to '1'
for i in range(0, len(dummy_DSCOVR_data)):
    dummy_DSCOVR_data[i] = util.change_invalid_values(dummy_DSCOVR_data[i], [999.999, 9999999, 99999.9, 9999.99], 1)


# Import Helios 1 data
HELIOS1_MAG_file = './data/helios1/helios1_mag_fix.csv'
HELIOS1_PL_file = './data/helios1/helios1_pl_interp.csv'
HELIOS1_dist_file = './data/helios1/helios1_dist_interp.csv'

HELIOS1_fltyr, HELIOS1_br, HELIOS1_bt, HELIOS1_bn, HELIOS1_bmag = np.loadtxt(HELIOS1_MAG_file, delimiter=',',
                                                                             unpack=True, usecols=range(0, 5))
print("Done loading Helios 1 MAG data")

HELIOS1_bulkspd, HELIOS1_prot_temp, HELIOS1_prot_dens = np.loadtxt(HELIOS1_PL_file, delimiter=',', unpack=True)
print("Done loading Helios 1 plasma data")

HELIOS1_rx, HELIOS1_ry, HELIOS1_rz = np.loadtxt(HELIOS1_dist_file, delimiter=',', unpack=True)
print("Done loading Helios 1 distances")


# Import Helios 2 data
HELIOS2_MAG_file = './data/helios2/helios2_mag_fix.csv'
HELIOS2_PL_file = './data/helios2/helios2_pl_interp.csv'
HELIOS2_dist_file = './data/helios2/helios2_dist_interp.csv'

HELIOS2_fltyr, HELIOS2_br, HELIOS2_bt, HELIOS2_bn, HELIOS2_bmag = np.loadtxt(HELIOS2_MAG_file, delimiter=',',
                                                                             unpack=True, usecols=range(0, 5))
print("Done loading Helios 2 MAG data")

HELIOS2_bulkspd, HELIOS2_prot_temp, HELIOS2_prot_dens = np.loadtxt(HELIOS2_PL_file, delimiter=',', unpack=True)
print("Done loading Helios 2 plasma data")

HELIOS2_rx, HELIOS2_ry, HELIOS2_rz = np.loadtxt(HELIOS2_dist_file, delimiter=',', unpack=True)
print("Done loading Helios 2 distances")


# Import STEREO A data
STEREO_A_MAG_file = './data/stereo_a/stereo_a_mag_fix.csv'
STEREO_A_PL_file = './data/stereo_a/stereo_a_prot_interp.csv'

STEREO_A_fltyr, STEREO_A_br, STEREO_A_bt, STEREO_A_bn, STEREO_A_bmag, STEREO_A_dist = \
    np.loadtxt(STEREO_A_MAG_file, delimiter=',', unpack=True, usecols=range(0, 6))
print("Done loading STEREO A MAG data")

STEREO_A_prot_dens, STEREO_A_prot_speed, STEREO_A_prot_temp = np.loadtxt(STEREO_A_PL_file, delimiter=',', unpack=True)
print("Done loading STEREO A plasma data")

dummy_STEREO_A_data = [STEREO_A_br, STEREO_A_bt, STEREO_A_bn, STEREO_A_bmag, STEREO_A_dist, STEREO_A_prot_dens,
                       STEREO_A_prot_speed, STEREO_A_prot_temp]

for i in range(0, len(dummy_STEREO_A_data)):
    dummy_STEREO_A_data[i] = util.change_invalid_values(dummy_STEREO_A_data[i], [1e34], 1)


# Import STEREO B data
STEREO_B_MAG_file = './data/stereo_b/stereo_b_mag_fix.csv'
STEREO_B_PL_file = './data/stereo_b/stereo_b_prot_interp.csv'

STEREO_B_fltyr, STEREO_B_br, STEREO_B_bt, STEREO_B_bn, STEREO_B_bmag, STEREO_B_dist = \
    np.loadtxt(STEREO_B_MAG_file, delimiter=',', unpack=True, usecols=range(0, 6))
print("Done loading STEREO B MAG data")

STEREO_B_prot_dens, STEREO_B_prot_speed, STEREO_B_prot_temp = np.loadtxt(STEREO_B_PL_file, delimiter=',', unpack=True)
print("Done loading STEREO B plasma data")

dummy_STEREO_B_data = [STEREO_B_br, STEREO_B_bt, STEREO_B_bn, STEREO_B_bmag, STEREO_B_dist, STEREO_B_prot_dens,
                       STEREO_B_prot_speed, STEREO_B_prot_temp]

for i in range(0, len(dummy_STEREO_B_data)):
    dummy_STEREO_B_data[i] = util.change_invalid_values(dummy_STEREO_B_data[i], [1e34], 1)


# Import WIND data
WIND_MAG_file = './data/wind/wind_mfi_fix.csv'
WIND_PL_file = './data/wind/wind_swe_interp.csv'

WIND_fltyr, WIND_bx, WIND_by, WIND_bz, WIND_bmag, WIND_rx, WIND_ry, WIND_rz = np.loadtxt(WIND_MAG_file, delimiter=',',
                                                                                         unpack=True)
print("Done loading WIND MAG data")

WIND_vr, WIND_vt, WIND_vn, WIND_prot_dens = np.loadtxt(WIND_PL_file, delimiter=',', unpack=True)
print("Done loading WIND plasma data")

dummy_WIND_data = [WIND_bx, WIND_by, WIND_bz, WIND_bmag, WIND_rx, WIND_ry, WIND_rz, WIND_vr, WIND_vt, WIND_vn,
                   WIND_prot_dens]

for i in range(0, len(dummy_WIND_data)):
    dummy_WIND_data[i] = util.change_invalid_values(dummy_WIND_data[i], [-1e+31], 1)


# Import Sunspot data
dly_sunspot_file = './data/sunspot/daily_sunspots_fix.csv'

dly_sunspot_fltyr, dly_sunspot_num = np.loadtxt(dly_sunspot_file, delimiter=',', unpack=True)
print("Done loading daily sunspot data")


# Calculate Ulysses basic quantities
SWOOPS_decday = SWOOPS_doy + (SWOOPS_hr / 24) + (SWOOPS_min / 1440) + (SWOOPS_sec / 86400)
ULY_fltyr = util.convert_to_fltyr(SWOOPS_yr, SWOOPS_decday)
ULY_mass_dens = (m_p * ULY_prot_dens) + (m_alpha * ULY_alpha_dens)
ULY_alpha_prot_ratio = ULY_alpha_dens / ULY_prot_dens
print("Done calculating Ulysses basic quantities")


# Calculate ACE basic quantities
ACE_alpha_dens = ACE_alpha_prot_ratio * ACE_prot_dens
ACE_mass_dens = (m_p * ACE_prot_dens) + (m_alpha * ACE_alpha_dens)
ACE_dist = 1 - (ACE_dist / (1.5 * 10**8))  # Convert into distance from Sun in AU
ACE_br = abs(ACE_br)
print("Done calculating ACE basic quantities")


# Calculate DSCOVR basic quantities
DSCOVR_mass_dens = m_p * DSCOVR_prot_dens
DSCOVR_dist = 1 - (abs(DSCOVR_dist) * 6371 * 6.68459e-9)  # Convert distance from Earth radii to AU
DSCOVR_br = abs(DSCOVR_br)
DSCOVR_vr = abs(DSCOVR_vx)
print("Done calculating DSCOVR basic quantities")


# Calculate Helios 1 basic quantities
HELIOS1_mass_dens = m_p * HELIOS1_prot_dens
HELIOS1_dist = abs((HELIOS1_rx**2 + HELIOS1_ry**2 + HELIOS1_rz**2)**0.5) * 6.68459e-9  # Convert from km to AU
HELIOS1_br = abs(HELIOS1_br)
print("Done calculating Helios 1 basic quantities")


# Calculate Helios 2 basic quantities
HELIOS2_mass_dens = m_p * HELIOS2_prot_dens
HELIOS2_dist = abs((HELIOS2_rx**2 + HELIOS2_ry**2 + HELIOS2_rz**2)**0.5) * 6.68459e-9  # Convert from km to AU
HELIOS2_br = abs(HELIOS2_br)
print("Done calculating Helios 2 basic quantities")


# Calculate STEREO A basic quantities
STEREO_A_mass_dens = m_p * STEREO_A_prot_dens
STEREO_A_dist = abs(STEREO_A_dist)
STEREO_A_br = abs(STEREO_A_br)
print("Done calculating STEREO A basic quantities")


# Calculate STEREO B basic quantities
STEREO_B_mass_dens = m_p * STEREO_B_prot_dens
STEREO_B_dist = abs(STEREO_B_dist)
STEREO_B_br = abs(STEREO_B_br)
print("Done calculating STEREO B basic quantities")


# Calculate WIND basic quantities
WIND_mass_dens = m_p * WIND_prot_dens
WIND_dist = 1 - (abs(WIND_rx) * 6371 * 6.68459e-9)  # Convert distance from Earth radii to AU
WIND_br = abs(WIND_bx)
print("Done calculating WIND basic quantities")


# Calculate Ulysses derived quantities
ULY_open_flux = stut.calc_open_flux(ULY_dist, ULY_br)
ULY_mdot = stut.calc_mass_loss_rate(ULY_dist, ULY_vr, ULY_mass_dens)
ULY_wind_magn = stut.calc_wind_magnetisation(ULY_open_flux, ULY_mdot)
ULY_rel_alfven_rad = stut.calc_rel_alfven_rad(ULY_wind_magn)
ULY_torque = stut.calc_torque(ULY_mdot, ULY_rel_alfven_rad)

print("Done calculating Ulysses derived quantities")

ULY_data = [ULY_fltyr, ULY_dist, ULY_hlat, ULY_hlong, ULY_mass_dens, ULY_vr, ULY_br, ULY_open_flux, ULY_mdot, ULY_wind_magn,
            ULY_rel_alfven_rad, ULY_torque]

ULY_keys = ["Year", "Distance [AU]", "Heliospheric Latitude", "Heliospheric Longitude", "Mass Density [g cm^-3]",
            "Radial Wind Velocity [km s^-1]", "Radial Magnetic Field Magnitude [nT]", "Open Flux [Mx]",
            "Mass Loss Rate [g s^-1]", "Wind Magnetisation", "Relative Alfvén Radius [Solar Units]", "Torque [erg]"]

ULY_keys = util.append_str(ULY_keys, " (Ulysses)")
ULY_dict = {key: value for key, value in zip(ULY_keys, ULY_data)}
print("Done creating Ulysses data dictionary")


# Calculate ACE derived quantities
ACE_open_flux = stut.calc_open_flux(ACE_dist, ACE_br)
ACE_mdot = stut.calc_mass_loss_rate(ACE_dist, ACE_vr, ACE_mass_dens)
ACE_wind_magn = stut.calc_wind_magnetisation(ACE_open_flux, ACE_mdot)
ACE_rel_alfven_rad = stut.calc_rel_alfven_rad(ACE_wind_magn)
ACE_torque = stut.calc_torque(ACE_mdot, ACE_rel_alfven_rad)

ACE_data = [ACE_fltyr, ACE_dist, ACE_mass_dens, ACE_vr, ACE_br, ACE_open_flux, ACE_mdot, ACE_wind_magn,
            ACE_rel_alfven_rad, ACE_torque]

ACE_keys = ["Year", "Distance [AU]", "Mass Density [g cm^-3]", "Radial Wind Velocity [km s^-1]",
            "Radial Magnetic Field Magnitude [nT]", "Open Flux [Mx]", "Mass Loss Rate [g s^-1]", "Wind Magnetisation",
            "Relative Alfvén Radius [Solar Units]", "Torque [erg]"]

ACE_keys = util.append_str(ACE_keys, " (ACE)")
ACE_dict = {key: value for key, value in zip(ACE_keys, ACE_data)}
print("Done creating ACE data dictionary")


# Calculate DSCOVR derived quantities
DSCOVR_open_flux = stut.calc_open_flux(DSCOVR_dist, DSCOVR_br)
DSCOVR_mdot = stut.calc_mass_loss_rate(DSCOVR_dist, DSCOVR_vr, DSCOVR_mass_dens)
DSCOVR_wind_magn = stut.calc_wind_magnetisation(DSCOVR_open_flux, DSCOVR_mdot)
DSCOVR_rel_alfven_rad = stut.calc_rel_alfven_rad(DSCOVR_wind_magn)
DSCOVR_torque = stut.calc_torque(DSCOVR_mdot, DSCOVR_rel_alfven_rad)

DSCOVR_data = [DSCOVR_fltyr, DSCOVR_dist, DSCOVR_mass_dens, DSCOVR_vr, DSCOVR_br, DSCOVR_open_flux, DSCOVR_mdot,
               DSCOVR_wind_magn, DSCOVR_rel_alfven_rad, DSCOVR_torque]

DSCOVR_keys = ["Year", "Distance [AU]", "Mass Density [g cm^-3]", "Radial Wind Velocity [km s^-1]",
               "Radial Magnetic Field Magnitude [nT]", "Open Flux [Mx]", "Mass Loss Rate [g s^-1]",
               "Wind Magnetisation", "Relative Alfvén Radius [Solar Units]", "Torque [erg]"]

DSCOVR_keys = util.append_str(DSCOVR_keys, " (DSCOVR)")
DSCOVR_dict = {key: value for key, value in zip(DSCOVR_keys, DSCOVR_data)}
print("Done creating DSCOVR data dictionary")


# Calculate Helios 1 derived quantities
HELIOS1_open_flux = stut.calc_open_flux(HELIOS1_dist, HELIOS1_br)
HELIOS1_mdot = stut.calc_mass_loss_rate(HELIOS1_dist, HELIOS1_bulkspd, HELIOS1_mass_dens)
HELIOS1_wind_magn = stut.calc_wind_magnetisation(HELIOS1_open_flux, HELIOS1_mdot)
HELIOS1_rel_alfven_rad = stut.calc_rel_alfven_rad(HELIOS1_wind_magn)
HELIOS1_torque = stut.calc_torque(HELIOS1_mdot, HELIOS1_rel_alfven_rad)

HELIOS1_data = [HELIOS1_fltyr, HELIOS1_dist, HELIOS1_mass_dens, HELIOS1_bulkspd, HELIOS1_br, HELIOS1_open_flux,
                HELIOS1_mdot, HELIOS1_wind_magn, HELIOS1_rel_alfven_rad, HELIOS1_torque]

HELIOS1_keys = ["Year", "Distance [AU]", "Mass Density [g cm^-3]", "Radial Wind Velocity [km s^-1]",
                "Radial Magnetic Field Magnitude [nT]", "Open Flux [Mx]", "Mass Loss Rate [g s^-1]",
                "Wind Magnetisation", "Relative Alfvén Radius [Solar Units]", "Torque [erg]"]

HELIOS1_keys = util.append_str(HELIOS1_keys, " (Helios 1)")
HELIOS1_dict = {key: value for key, value in zip(HELIOS1_keys, HELIOS1_data)}
print("Done creating Helios 1 data dictionary")


# Calculate Helios 2 derived quantities
HELIOS2_open_flux = stut.calc_open_flux(HELIOS2_dist, HELIOS2_br)
HELIOS2_mdot = stut.calc_mass_loss_rate(HELIOS2_dist, HELIOS2_bulkspd, HELIOS2_mass_dens)
HELIOS2_wind_magn = stut.calc_wind_magnetisation(HELIOS2_open_flux, HELIOS2_mdot)
HELIOS2_rel_alfven_rad = stut.calc_rel_alfven_rad(HELIOS2_wind_magn)
HELIOS2_torque = stut.calc_torque(HELIOS2_mdot, HELIOS2_rel_alfven_rad)

HELIOS2_data = [HELIOS2_fltyr, HELIOS2_dist, HELIOS2_mass_dens, HELIOS2_bulkspd, HELIOS2_br,
                HELIOS2_open_flux, HELIOS2_mdot, HELIOS2_wind_magn, HELIOS2_rel_alfven_rad, HELIOS2_torque]

HELIOS2_keys = ["Year", "Distance [AU]", "Mass Density [g cm^-3]", "Radial Wind Velocity [km s^-1]",
                "Radial Magnetic Field Magnitude [nT]", "Open Flux [Mx]", "Mass Loss Rate [g s^-1]",
                "Wind Magnetisation", "Relative Alfvén Radius [Solar Units]", "Torque [erg]"]

HELIOS2_keys = util.append_str(HELIOS2_keys, " (Helios 2)")
HELIOS2_dict = {key: value for key, value in zip(HELIOS2_keys, HELIOS2_data)}
print("Done creating Helios 2 data dictionary")


# Calculate STEREO A derived quantities
STEREO_A_open_flux = stut.calc_open_flux(STEREO_A_dist, STEREO_A_br)
STEREO_A_mdot = stut.calc_mass_loss_rate(STEREO_A_dist, STEREO_A_prot_speed, STEREO_A_mass_dens)
STEREO_A_wind_magn = stut.calc_wind_magnetisation(STEREO_A_open_flux, STEREO_A_mdot)
STEREO_A_rel_alfven_rad = stut.calc_rel_alfven_rad(STEREO_A_wind_magn)
STEREO_A_torque = stut.calc_torque(STEREO_A_mdot, STEREO_A_rel_alfven_rad)

STEREO_A_data = [STEREO_A_fltyr, STEREO_A_dist, STEREO_A_mass_dens, STEREO_A_prot_speed, STEREO_A_br,
                 STEREO_A_open_flux, STEREO_A_mdot, STEREO_A_wind_magn, STEREO_A_rel_alfven_rad, STEREO_A_torque]

STEREO_A_keys = ["Year", "Distance [AU]", "Mass Density [g cm^-3]", "Radial Wind Velocity [km s^-1]",
                 "Radial Magnetic Field Magnitude [nT]",  "Open Flux [Mx]", "Mass Loss Rate [g s^-1]",
                 "Wind Magnetisation", "Relative Alfvén Radius [Solar Units]", "Torque [erg]"]

STEREO_A_keys = util.append_str(STEREO_A_keys, " (STEREO A)")
STEREO_A_dict = {key: value for key, value in zip(STEREO_A_keys, STEREO_A_data)}
print("Done creating STEREO A data dictionary")


# Calculate STEREO B derived quantities
STEREO_B_open_flux = stut.calc_open_flux(STEREO_B_dist, STEREO_B_br)
STEREO_B_mdot = stut.calc_mass_loss_rate(STEREO_B_dist, STEREO_B_prot_speed, STEREO_B_mass_dens)
STEREO_B_wind_magn = stut.calc_wind_magnetisation(STEREO_B_open_flux, STEREO_B_mdot)
STEREO_B_rel_alfven_rad = stut.calc_rel_alfven_rad(STEREO_B_wind_magn)
STEREO_B_torque = stut.calc_torque(STEREO_B_mdot, STEREO_B_rel_alfven_rad)

STEREO_B_data = [STEREO_B_fltyr, STEREO_B_dist, STEREO_B_mass_dens, STEREO_B_prot_speed, STEREO_B_br,
                 STEREO_B_open_flux, STEREO_B_mdot, STEREO_B_wind_magn, STEREO_B_rel_alfven_rad, STEREO_B_torque]

STEREO_B_keys = ["Year", "Distance [AU]", "Mass Density [g cm^-3]", "Radial Wind Velocity [km s^-1]",
                 "Radial Magnetic Field Magnitude [nT]", "Open Flux [Mx]", "Mass Loss Rate [g s^-1]",
                 "Wind Magnetisation", "Relative Alfvén Radius [Solar Units]", "Torque [erg]"]

STEREO_B_keys = util.append_str(STEREO_B_keys, " (STEREO B)")
STEREO_B_dict = {key: value for key, value in zip(STEREO_B_keys, STEREO_B_data)}
print("Done creating STEREO B data dictionary")


# Calculate WIND derived quantities
WIND_open_flux = stut.calc_open_flux(WIND_dist, WIND_br)
WIND_mdot = stut.calc_mass_loss_rate(WIND_dist, WIND_vr, WIND_mass_dens)
WIND_wind_magn = stut.calc_wind_magnetisation(WIND_open_flux, WIND_mdot)
WIND_rel_alfven_rad = stut.calc_rel_alfven_rad(WIND_wind_magn)
WIND_torque = stut.calc_torque(WIND_mdot, WIND_rel_alfven_rad)

WIND_data = [WIND_fltyr, WIND_dist, WIND_mass_dens, WIND_vr, WIND_br, WIND_open_flux, WIND_mdot, WIND_wind_magn,
             WIND_rel_alfven_rad, WIND_torque]
WIND_keys = ["Year", "Distance [AU]", "Mass Density [g cm^-3]", "Radial Wind Velocity [km s^-1]",
             "Radial Magnetic Field Magnitude [nT]", "Open Flux [Mx]", "Mass Loss Rate [g s^-1]", "Wind Magnetisation",
             "Relative Alfvén Radius [Solar Units", "Torque [erg]"]

WIND_keys = util.append_str(WIND_keys, " (WIND)")
WIND_dict = {key: value for key, value in zip(WIND_keys, WIND_data)}
print("Done creating WIND data dictionary")


# Create sunspot data dictionary
dly_sunspot_data = [dly_sunspot_fltyr, dly_sunspot_num]
dly_sunspot_keys = ["Year", "Sunspot Number"]

dly_sunspot_keys = util.append_str(dly_sunspot_keys, " (SILS)")
dly_sunspot_dict = {key: value for key, value in zip(dly_sunspot_keys, dly_sunspot_data)}


# Create averaged data lists
avgd_ULY_data = [[] for _ in range(len(ULY_data))]
avgd_ACE_data = [[] for _ in range(len(ACE_data))]
avgd_DSCOVR_data = [[] for _ in range(len(DSCOVR_data))]
avgd_HELIOS1_data = [[] for _ in range(len(HELIOS1_data))]
avgd_HELIOS2_data = [[] for _ in range(len(HELIOS2_data))]
avgd_STEREO_A_data = [[] for _ in range(len(STEREO_A_data))]
avgd_STEREO_B_data = [[] for _ in range(len(STEREO_B_data))]
avgd_WIND_data = [[] for _ in range(len(WIND_data))]
avgd_dly_sunspot_data = [[] for _ in range(len(dly_sunspot_data))]


def bin_data(bin_size):
    bin_size *= 24

    # Bin Ulysses data
    for index, element in enumerate(ULY_data):
        if index is not 0:
            avgd_ULY_data[0], avgd_ULY_data[index] = util.binned_average(ULY_data[0], ULY_data[index], bin_size)

    # # Remove data points for which the distance is greater than 2 AU
    removed_indices = []

    for index, value in enumerate(avgd_ULY_data[1]):  # Going over the list of distances
        if value > 2:  # If distance > 2 AU, append the respective index to this list
            removed_indices.append(index)

    for i, element in enumerate(avgd_ULY_data):  # Go through all the averaged data and remove the respective indices
        element = np.delete(element, removed_indices)
        avgd_ULY_data[i] = element

    # Bin ACE data
    for index, element in enumerate(ACE_data):
        if index is not 0:
            avgd_ACE_data[0], avgd_ACE_data[index] = util.binned_average(ACE_data[0], ACE_data[index], bin_size)

    # Bin DSCOVR data
    for index, element in enumerate(DSCOVR_data):
        if index is not 0:
            avgd_DSCOVR_data[0], avgd_DSCOVR_data[index] = util.binned_average(DSCOVR_data[0], DSCOVR_data[index], bin_size)

    # Bin Helios 1 data
    for index, element in enumerate(HELIOS1_data):
        if index is not 0:
            avgd_HELIOS1_data[0], avgd_HELIOS1_data[index] = util.binned_average(HELIOS1_data[0], HELIOS1_data[index], bin_size)

    # Bin Helios 2 data
    for index, element in enumerate(HELIOS2_data):
        if index is not 0:
            avgd_HELIOS2_data[0], avgd_HELIOS2_data[index] = util.binned_average(HELIOS2_data[0], HELIOS2_data[index], bin_size)

    # Bin STEREO A data
    for index, element in enumerate(STEREO_A_data):
        if index is not 0:
            avgd_STEREO_A_data[0], avgd_STEREO_A_data[index] = util.binned_average(STEREO_A_data[0], STEREO_A_data[index], bin_size)

    # Bin STEREO B data
    for index, element in enumerate(STEREO_B_data):
        if index is not 0:
            avgd_STEREO_B_data[0], avgd_STEREO_B_data[index] = util.binned_average(STEREO_B_data[0], STEREO_B_data[index], bin_size)

    # Bin WIND data
    for index, element in enumerate(WIND_data):
        avgd_WIND_data[0], avgd_WIND_data[index] = util.binned_average(WIND_data[0], WIND_data[index], bin_size)

    # Bin daily sunspot data
    for index, element in enumerate(dly_sunspot_data):
        avgd_dly_sunspot_data[0], avgd_dly_sunspot_data[index] = util.binned_average(dly_sunspot_data[0],
                                                                                     dly_sunspot_data[index],
                                                                                     int(bin_size / 24))

    return avgd_ULY_data, avgd_ACE_data, avgd_DSCOVR_data, avgd_HELIOS1_data, avgd_HELIOS2_data, avgd_STEREO_A_data, \
        avgd_STEREO_B_data, avgd_WIND_data, avgd_dly_sunspot_data


# Dash plotting starts here
app = dash.Dash()


def update_dict(bin_size):  # Updates the data dictionary for plotting and also has functionality for saving files
    avgd_ULY_data, avgd_ACE_data, avgd_DSCOVR_data, avgd_HELIOS1_data, avgd_HELIOS2_data, avgd_STEREO_A_data,\
        avgd_STEREO_B_data, avgd_WIND_data, avgd_dly_sunspot_data = bin_data(bin_size)

    ULY_avg_dict = {key: value for key, value in zip(ULY_keys, avgd_ULY_data)}
    # pd.DataFrame.from_dict(ULY_avg_dict, orient='columns').to_csv('ULYSSES_27day_avg.csv', index=False)

    ACE_avg_dict = {key: value for key, value in zip(ACE_keys, avgd_ACE_data)}
    # pd.DataFrame.from_dict(ACE_avg_dict, orient='columns').to_csv('ace_27day_avg.csv', index=False)

    DSCOVR_avg_dict = {key: value for key, value in zip(DSCOVR_keys, avgd_DSCOVR_data)}
    # pd.DataFrame.from_dict(DSCOVR_avg_dict, orient='columns').to_csv('dscovr_27day_avg.csv', index=False)

    HELIOS1_avg_dict = {key: value for key, value in zip(HELIOS1_keys, avgd_HELIOS1_data)}
    # pd.DataFrame.from_dict(HELIOS1_avg_dict, orient='columns').to_csv('HELIOS1_27day_avg.csv', index=False)

    HELIOS2_avg_dict = {key: value for key, value in zip(HELIOS2_keys, avgd_HELIOS2_data)}
    # pd.DataFrame.from_dict(HELIOS2_avg_dict, orient='columns').to_csv('HELIOS2_27day_avg.csv', index=False)

    STEREO_A_avg_dict = {key: value for key, value in zip(STEREO_A_keys, avgd_STEREO_A_data)}
    # pd.DataFrame.from_dict(STEREO_A_avg_dict, orient='columns').to_csv('stereo_a_27day_avg.csv', index=False)

    STEREO_B_avg_dict = {key: value for key, value in zip(STEREO_B_keys, avgd_STEREO_B_data)}
    # pd.DataFrame.from_dict(STEREO_B_avg_dict, orient='columns').to_csv('stereo_b_27day_avg.csv', index=False)

    WIND_avg_dict = {key: value for key, value in zip(WIND_keys, avgd_WIND_data)}
    # pd.DataFrame.from_dict(WIND_avg_dict, orient='columns').to_csv('wind_27day_avg.csv', index=False)

    dly_sunspot_avg_dict = {key: value for key, value in zip(dly_sunspot_keys, avgd_dly_sunspot_data)}
    # pd.DataFrame.from_dict(dly_sunspot_avg_dict, orient='columns').to_csv('dly_sunspot_27day_avg.csv', index=False)

    avg_data_dict = {**ULY_avg_dict, **ACE_avg_dict, **DSCOVR_avg_dict, **HELIOS1_avg_dict, **HELIOS2_avg_dict,
                     **STEREO_A_avg_dict, **STEREO_B_avg_dict, **WIND_avg_dict, **dly_sunspot_avg_dict}

    return avg_data_dict


avg_data_dict = update_dict(bin_size=27)  # Initialise to 27-day bins


app.layout = html.Div(children=[
    html.Div(children='''Choose parameters to plot from drop-down boxes'''),
    dcc.Dropdown(id='x-drop-state',
                 options=[{'label': s, 'value': s} for s in avg_data_dict.keys()],
                 multi=True,
                 value=['Year (Ulysses)', 'Year (ACE)', 'Year (DSCOVR)', 'Year (Helios 1)', 'Year (Helios 2)',
                        'Year (STEREO A)', 'Year (STEREO B)', 'Year (WIND)']),
    dcc.RadioItems(
        id='x-graph-scale',
        options=[
            {'label': 'Linear', 'value': 'lin'},
            {'label': 'Logarithmic', 'value': 'log'}
        ],
        value='lin'
    ),
    dcc.Dropdown(id='y-drop-state',
                 options=[{'label': s, 'value': s} for s in avg_data_dict.keys()],
                 multi=True,
                 value=['Distance [AU] (Ulysses)', 'Distance [AU] (ACE)', 'Distance [AU] (DSCOVR)', 'Distance [AU] (Helios 1)',
                        'Distance [AU] (Helios 2)', 'Distance [AU] (STEREO A)',
                        'Distance [AU] (STEREO B)', 'Distance [AU] (WIND)']),
    dcc.RadioItems(
        id='y-graph-scale',
        options=[
            {'label': 'Linear', 'value': 'lin'},
            {'label': 'Logarithmic', 'value': 'log'}
        ],
        value='lin'
    ),
    dcc.Checklist(
        id='raw-data-checkbox',
        options=[
            {'label': 'Show Raw Data (requires re-plot)', 'value': 'show-data'}
        ],
        values=[]
    ),
    dcc.RadioItems(
        id='line-style',
        options=[
            {'label': 'Lines', 'value': 'lines'},
            {'label': 'Markers', 'value': 'markers'},
            {'label': 'Lines and Markers', 'value': 'lines+markers'}
        ],
        value='markers'
    ),
    html.Div('''Enter bin size in days (will take longer to update for smaller bins!): '''),
    dcc.Input(
        id='bin-size-state', type='number', value=27, min='1', max='30'
    ),
    html.Button(id='plot-button', children='Plot'),
    html.Div(id='output-graph')
])


@app.callback(
    Output(component_id='output-graph', component_property='children'),
    [Input(component_id='plot-button', component_property='n_clicks')],
    [State(component_id='x-drop-state', component_property='value'),
     State(component_id='x-graph-scale', component_property='value'),
     State(component_id='y-drop-state', component_property='value'),
     State(component_id='y-graph-scale', component_property='value'),
     State(component_id='raw-data-checkbox', component_property='values'),
     State(component_id='line-style', component_property='value'),
     State(component_id='bin-size-state', component_property='value')]
)
def update_graph(n, x_data_names, x_graph_scale, y_data_names, y_graph_scale, show_raw_data, line_style, bin_size):
    global avg_data_dict
    avg_data_dict = update_dict(bin_size)  # This recalculates quantities within the avg data dict

    if len(y_data_names) > 1:
        avg_list = [
            np.sum(avg_data_dict.get(y)) / len(avg_data_dict.get(y)) for y in y_data_names
        ]

        data = [
            {'x': avg_data_dict.get(x), 'y': avg_data_dict.get(y), 'mode': line_style,
             'name': util.strip_identifier(y, False),
             'text': "Average is " + str(np.sum(avg_data_dict.get(y)) / len(avg_data_dict.get(y)))}
            for x in x_data_names for y in y_data_names if len(avg_data_dict.get(x)) == len(avg_data_dict.get(y))
        ]

        start_x_list = []
        end_x_list = []

        for x in x_data_names:
            start_x_list.append(avg_data_dict.get(x)[0])
            end_x_list.append(avg_data_dict.get(x)[-1])

        start_x = min(start_x_list)
        end_x = max(end_x_list)

        shapes = [{
            'type': 'line',
            'x0': start_x,
            'y0': a,
            'x1': end_x,
            'y1': a,
            'line': {
                'width': 1,
                'dash': 'dot'
            }
        } for a in avg_list]
    else:
        avg = np.sum(avg_data_dict.get(y_data_names[0])) / len(avg_data_dict.get(y_data_names[0]))
        data = [
            {'x': avg_data_dict.get(x_data_names[0]), 'y': avg_data_dict.get(y_data_names[0]), 'mode': line_style,
             'name': util.strip_identifier(y_data_names[0], False), 'text': "Average is " + str(avg)}
        ]

        start_x = avg_data_dict.get(x_data_names[0])[0]
        end_x = avg_data_dict.get(x_data_names[0])[-1]
        shapes = [{
            'type': 'line',
            'x0': start_x,
            'y0': avg,
            'x1': end_x,
            'y1': avg,
            'line': {
                'width': 1,
                'dash': 'dot'
            }
        }]

    return dcc.Graph(
        id='graph1',
        figure={
            'data': data,
            'layout': go.Layout(
                    xaxis=dict(
                        type=x_graph_scale,
                        autorange=True,
                        showgrid=False,
                        title=util.strip_identifier(x_data_names[0], True),
                    ),
                    yaxis=dict(
                        type=y_graph_scale,
                        autorange=True,
                        showgrid=False,
                        title=util.strip_identifier(y_data_names[0], True),
                        exponentformat='power',
                        showexponent='all'
                    ),
                    shapes=shapes
            )
        }
    )


if __name__ == "__main__":
    app.run_server()

