import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# Script purely for testing out curve fits and plotting them; plotting scripts should be clean

# Read data into dictionaries
owens_dict = pd.read_csv('data/models/owens_equatorial.csv').to_dict('list')
omni_dict = pd.read_csv("data/27_day_avg/OMNI_27day_avg_hr.csv").to_dict("list")

# Restrict Owens model data to past 1960 (index 345)
owens_fltyr = np.array([yr for i, yr in enumerate(owens_dict.get("Year (Owens)"))
                             if i >= 345])
owens_phi = np.array([phi for i, phi in enumerate(owens_dict.get("Open Flux [Mx] (Owens)"))
                           if i >= 345]) / 2
owens_vr = np.array([vr * 10**5 for i, vr in enumerate(owens_dict.get("Radial Wind Velocity [km s^-1] (Owens)"))
                          if i >= 345])

# Filter out the bad data for OMNI
# time_omni_daily = np.array([yr for i, yr in enumerate(omni_dict.get("Year (OMNI Daily)"))
#                             if omni_dict.get("Flags (OMNI Daily)")[i] != 1])
# mdot_omni_daily = np.array([mdot for i, mdot in enumerate(omni_dict.get("Mass Loss Rate [g s^-1] (OMNI Daily)"))
#                             if omni_dict.get("Flags (OMNI Daily)")[i] != 1])
# phi_omni_daily = np.array([phi for i, phi in enumerate(omni_dict.get("Open Flux [Mx] (OMNI Daily)"))
#                             if omni_dict.get("Flags (OMNI Daily)")[i] != 1])
# vr_omni_daily = np.array([vr for i, vr in enumerate(omni_dict.get("Radial Wind Velocity [km s^-1] (OMNI Daily)"))
#                             if omni_dict.get("Flags (OMNI Daily)")[i] != 1]) * 10**5
# torque_omni_daily = np.array([torque for i, torque in enumerate(omni_dict.get("Torque [erg] (OMNI Daily)"))
#                             if omni_dict.get("Flags (OMNI Daily)")[i] != 1])


time_omni_hr = np.asarray([yr for i, yr in enumerate(omni_dict.get("Year (OMNI Hr)"))
                            if omni_dict.get("Flags (OMNI Hr)")[i] != 1])
br_omni_hr = np.asarray([br for i, br in enumerate(omni_dict.get("Radial Magnetic Field Magnitude [nT] (OMNI Hr)"))
                         if omni_dict.get("Flags (OMNI Hr)")[i] != 1])
mdot_omni_hr = np.asarray([mdot for i, mdot in enumerate(omni_dict.get("Mass Loss Rate [g s^-1] (OMNI Hr)"))
                            if omni_dict.get("Flags (OMNI Hr)")[i] != 1])
phi_omni_hr = np.asarray([phi for i, phi in enumerate(omni_dict.get("Open Flux [Mx] (OMNI Hr)"))
                            if omni_dict.get("Flags (OMNI Hr)")[i] != 1])
vr_omni_hr = np.asarray([vr * 1e5 for i, vr in enumerate(omni_dict.get("Radial Wind Velocity [km s^-1] (OMNI Hr)"))
                            if omni_dict.get("Flags (OMNI Hr)")[i] != 1])
torque_omni_hr = np.asarray([torque for i, torque in enumerate(omni_dict.get("Torque [erg] (OMNI Hr)"))
                            if omni_dict.get("Flags (OMNI Hr)")[i] != 1])

# Define lambdas for all the fits
# Single variable
fit1v = lambda x, a1, a2: a1 * x**a2

# 2 variable, same as normalised, just pass in the normalised quantity
fit2v = lambda x_inputs, a1, a2, a3, a4: a1 * x_inputs[0]**a2 + a3 * x_inputs[1]**a4

# Cross term
fitcross = lambda x_inputs, a1, a2: a1 * (x_inputs[0] * x_inputs[1])**a2

# 2 variable + crossterm
fitcomplex = lambda x_inputs, a1, a2, a3, a4, a5, a6: a1 * x_inputs[0]**a2 + a3 * x_inputs[1]*a4 + \
                                                      a5 * (x_inputs[0] * x_inputs[1])**a6

# Define input arrays for the 2 variable lambdas
omni_input = np.asarray((phi_omni_hr, vr_omni_hr))
omni_norm_input = np.asarray((phi_omni_hr / 8e22, vr_omni_hr / 500e5))
owens_input = np.asarray((owens_phi, owens_vr))
owens_norm_input = np.asarray((owens_phi / 8e22, owens_vr / 500e5))

# Generate the individual fits
phi_fit1_params, _ = opt.curve_fit(fit1v, phi_omni_hr, mdot_omni_hr, maxfev=10000)
vr_fit1_params, _ = opt.curve_fit(fit1v, vr_omni_hr, mdot_omni_hr, maxfev=10000)

# Generate the 2 parameter fits (regular and normalised)
fit2_reg_params, _ = opt.curve_fit(fit2v, omni_input, mdot_omni_hr, maxfev=10000)
fit2_norm_params, _ = opt.curve_fit(fit2v, omni_norm_input, mdot_omni_hr / 1e12, maxfev=10000)

# Generate the crossterm fit
crossterm_fit_reg_params, _ = opt.curve_fit(fitcross, omni_input, mdot_omni_hr, maxfev=10000)
crossterm_fit_norm_params, _ = opt.curve_fit(fitcross, omni_norm_input, mdot_omni_hr / 1e12, maxfev=10000)


# Generate the complex fit (normalised)
complex_fit_params, _ = opt.curve_fit(fitcomplex, omni_norm_input, mdot_omni_hr / 1e12, maxfev=10000)

# Print everything.
print("-=-=-=-Single variable-=-=-=-\n"
      "Variable: [a1, a2]")
print("Open Flux: " + str(phi_fit1_params))
print("v_r: " + str(vr_fit1_params))

print("-=-=-=-Two variable-=-=-=-\n"
      "Type: [a1, a2, a3, a4]")
print("Regular: " + str(fit2_reg_params))
print("Normalised: " + str(fit2_norm_params))

print("-=-=-=-Crossterm fits-=-=-=-\n"
      "Type: [a1, a2]")
print("Regular: " + str(crossterm_fit_reg_params))
print("Normalised: " + str(crossterm_fit_norm_params))
print("-=-=-=-Complex fit (normalised)-=-=-=-\n"
      "[a1, a2, a3, a4, a5, a6]")
print(str(complex_fit_params))

# Plot everything.
fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
axarr = [ax1, ax2, ax3]
# Plot OMNI data on all axes
for i in range(0, len(axarr)):
    axarr[i].plot(time_omni_hr, mdot_omni_hr, '.', markersize=1)

# Axis 1: Individual fits
ax1.plot(owens_fltyr, fit1v(owens_phi, phi_fit1_params[0], phi_fit1_params[1]),
         color="#0033cc", ls='-', label=r"$\Phi$$_{open}$")
ax1.plot(owens_fltyr, fit1v(owens_vr, vr_fit1_params[0], vr_fit1_params[1]),
         color="#cc0000", ls='-', label=r"v_r")

ax1.set_xlabel("Year")
ax1.set_ylabel("Mass Loss Rate")
ax1.legend()
ax1.set_title("Single variable fits")
# Axis 2: 2 parameter fits
ax2.plot(owens_fltyr, fit2v(owens_input, fit2_reg_params[0], fit2_reg_params[1], fit2_reg_params[2], fit2_reg_params[3]),
         color="#b38600", ls='-', label="Regular")
ax2.plot(owens_fltyr, fit2v(owens_norm_input, fit2_norm_params[0], fit2_norm_params[1], fit2_norm_params[2], fit2_norm_params[3]) * 1e12,
         color="#408000", ls='-', label="Normalised")

ax2.set_xlabel("Year")
ax2.legend()
ax2.set_title("2 variable fits")
# Axis 3: crossterm fits
# ax3.plot(owens_fltyr, fitcross(owens_input, crossterm_fit_reg_params[0], crossterm_fit_reg_params[1]),
#          color="#b38600", ls='-', label="Regular")
ax3.plot(owens_fltyr, fit2v(owens_norm_input, fit2_norm_params[0], fit2_norm_params[1], fit2_norm_params[2], fit2_norm_params[3]) * 1e12,
         color="#cc9900", ls='-', label="2 variable")
ax3.plot(owens_fltyr, fitcross(owens_norm_input, crossterm_fit_norm_params[0], crossterm_fit_norm_params[1]) * 1e12,
         color="#408000", ls='-', label="Crossterm")
ax3.plot(owens_fltyr, fitcomplex(owens_norm_input, complex_fit_params[0], complex_fit_params[1], complex_fit_params[2],
                                 complex_fit_params[3], complex_fit_params[4], complex_fit_params[5]) * 1e12,
         color="#cc00cc", ls='-', label="Complex")
# Plot difference
plt.plot(owens_fltyr, abs(fit2v(owens_norm_input, fit2_norm_params[0], fit2_norm_params[1], fit2_norm_params[2], fit2_norm_params[3]) -
         fitcomplex(owens_norm_input, complex_fit_params[0], complex_fit_params[1], complex_fit_params[2],
                    complex_fit_params[3], complex_fit_params[4], complex_fit_params[5])) * 1e12,
         color="#e60000", ls="--")

ax3.set_xlabel("Year")
ax3.legend()
ax3.set_title("Normalised mixed fits")

# Test fits directly
fig2 = plt.figure()

test_phi = np.linspace(0.4, 2., 20) * 1e22 / 8e22
test_vr = np.linspace(300, 900, 20) * 1e5 / 500e5
test_fit = fitcomplex([test_phi, test_vr], complex_fit_params[0], complex_fit_params[1], complex_fit_params[2],
                       complex_fit_params[3], complex_fit_params[4], complex_fit_params[5]) * 1e12

# test_phi, test_vr = np.meshgrid(test_phi, test_vr)
# test_surf = fitcomplex([test_phi, test_vr], complex_fit_params[0], complex_fit_params[1], complex_fit_params[2],
#                        complex_fit_params[3], complex_fit_params[4], complex_fit_params[5]) * 1e12

# 2d stuff
axtest2d_phi = fig2.add_subplot(121)
axtest2d_phi.plot(phi_omni_hr, mdot_omni_hr, '.', markersize=1)
axtest2d_phi.plot(phi_omni_hr, fit1v(phi_omni_hr, phi_fit1_params[0], phi_fit1_params[1]))

axtest2d_vr = fig2.add_subplot(122)
axtest2d_vr.plot(vr_omni_hr, mdot_omni_hr, '.', markersize=1)
axtest2d_vr.plot(vr_omni_hr, fit1v(vr_omni_hr, vr_fit1_params[0], vr_fit1_params[1]))

# 3d stuff
# ax = fig2.add_subplot(122, projection="3d")
#
# ax.plot(phi_omni_hr, vr_omni_hr, mdot_omni_hr, '.', markersize=1)
# # ax.plot(owens_phi, owens_vr, fitcomplex(owens_norm_input, complex_fit_params[0], complex_fit_params[1], complex_fit_params[2],
# #          complex_fit_params[3], complex_fit_params[4], complex_fit_params[5]) * 1e12)
#
#
# # ax.plot_surface(test_phi, test_vr, test_surf)
#
# ax.set_xlabel("Open Flux [Mx]")
# ax.set_ylabel("Radial Wind Velocity [km/s]")
plt.figure()
plt.plot(time_omni_hr, br_omni_hr, '.', markersize=2)

plt.xlabel("Year")
plt.ylabel(r"B$_r$ [nT]")

plt.tight_layout()
plt.show()
