import numpy as np
import sys


sun_radius = 6.96 * 10**10  # Radius of the Sun in centimetres
sun_v_esc = 6.18 * 10**7  # Escape velocity at the stellar surface in cm/s
sun_rot_rate = 2.6 * 10**(-6)  # Rotation rate of the Sun in rad/s
k_0 = 0.33  # Dipolar coefficient
m_0 = 0.371  # Dipolar coefficient


class SolarCycle:

    def __init__(self, cycle_number, cycle_start, cycle_end, data):
        self.cycle_number = cycle_number
        self.cycle_start = cycle_start
        self.cycle_end = cycle_end
        self.data = data


def calc_open_flux_r2b(rb_list):
    open_flux_list = []

    # Defined in Maxwells
    for index, value in enumerate(rb_list):
        open_flux = 4 * np.pi * value
        open_flux_list.append(open_flux)


# Takes in distance in AU and B_r in nT to give open flux in Mx
def calc_open_flux(dist_list, br_list):
    open_flux_list = []

    # Defined in Maxwells
    for index, distance in enumerate(dist_list):
        if distance is 0 or br_list[index] is 0:
            open_flux = 0
        else:
            open_flux = 4 * np.pi * (dist_list[index] * 1.5 * 10 ** 11) ** 2 * (br_list[index] * 10 ** -9) * 10 ** 8

        open_flux_list.append(open_flux)

    return np.array(open_flux_list)


def calc_mass_loss_rate(dist_list, v_r_list, dens_list):
    m_dot_list = []

    # Need to convert distances from AU -> cm & velocities from km/s -> cm/s to get mDot in g/s
    for index, distance in enumerate(dist_list):
        if distance is 0 or v_r_list[index] is 0 or dens_list[index] is 1:
            m_dot = 0
        else:
            m_dot = 4 * np.pi * (distance * 1.5 * 10**13)**2 * v_r_list[index] * 10**5 * dens_list[index]

        m_dot_list.append(m_dot)

    return np.array(m_dot_list)


def calc_wind_magnetisation(open_flux_list, m_dot_list):
    wind_mag_list = []

    for index, element in enumerate(open_flux_list):
        if element is 0 or m_dot_list[index] is 0:
            wind_mag = 0
        else:
            wind_mag = (element / sun_radius)**2 / (sun_v_esc * m_dot_list[index])

        wind_mag_list.append(wind_mag)

    return np.array(wind_mag_list)


def calc_rel_alfven_rad(wind_mag_list):
    rel_alfven_rad_list = []

    for index, element in enumerate(wind_mag_list):
        if element is 0:
            rel_alfven_rad = 0
        else:
            rel_alfven_rad = k_0 * element**m_0

        rel_alfven_rad_list.append(rel_alfven_rad)

    return np.array(rel_alfven_rad_list)


def calc_torque(m_dot_list, rel_alf_rad_list):
    torque_list = []

    for index, element in enumerate(m_dot_list):
        if element is 0 or rel_alf_rad_list[index] is 0:
            torque = 0
        else:
            torque = element * sun_rot_rate * sun_radius**2 * rel_alf_rad_list[index]**2

        torque_list.append(torque)

    return np.array(torque_list)


def calc_solar_torque(m_dot_list, open_flux_list):
    torque_list = []

    for index, element in enumerate(m_dot_list):
        if element is 0 or open_flux_list[index] is 0:
            torque = 0
        else:
            torque = 1.84 * 10**(-7) * element**0.258 * open_flux_list[index]**1.484

        torque_list.append(torque)

    return torque_list


# Takes in a list of solarCycles
