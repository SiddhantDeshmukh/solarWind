import os
import scipy.interpolate as interpolate
import numpy as np
import pandas as pd
from collections import defaultdict
import datetime as dt
from scipy import optimize, stats
import matplotlib.pyplot as plt
import operator


powerlaw = lambda x, amp, index: amp * (x**index)
fitfunc = lambda p, x: p[0] + p[1] * x
errfunc = lambda p, x, y: y - fitfunc(p, x)


def chunk_list(list, n):
    chunks = []
    step = len(list) // n - 1

    for i in range(0, len(list), step):
        chunks.append(list[i: i + step])

    return chunks


def concat_files(folder_name, concat_file_name):  # Concatenate an array of files
    folder = folder_name
    fname_array = os.listdir(folder)
    fname_array = sorted(fname_array)

    for index in range(0, len(fname_array)):
        fname_array[index] = folder + '/' + fname_array[index]

    with open(concat_file_name, 'w') as outfile:
        for fname in fname_array:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)


def convert_to_fltyr(yr, decday):
    fltyr_list = []

    for index, element in enumerate(yr):
        if str(yr[index]).startswith('9'):
            if len(str(int(yr[index]))) > 1:
                yr[index] = float('19' + str(yr[index]))
            else:
                yr[index] = 2009
        else:
            yr[index] = float('200' + str(yr[index]))

        fltyr = yr[index] + decday[index] / 365.25
        fltyr_list.append(fltyr)

    return fltyr_list


def calc_time_since_jan1_1990(yr, decday):
    yr = str(int(yr))  # Cast to int to remove dec places THEN cast to string for if statement; could do this in call

    if yr.startswith('9'):
        if len(yr) > 1:
            yr = '19' + yr
        else:
            yr = '2009'

    else:
        yr = '200' + yr

    yr_diff = int(yr) - 1990
    leap_days = 0

    if yr_diff > 18:
        leap_days = 5
    elif yr_diff > 14:
        leap_days = 4
    elif yr_diff > 10:
        leap_days = 3
    elif yr_diff > 6:
        leap_days = 2
    elif yr_diff > 2:
        leap_days = 1

    abs_decday = yr_diff * 365 + leap_days + decday

    return abs_decday


# Takes in a list of float years and converts them into Carrington rotations
def calc_carrington_rot(fltyr_list):
    start_date = 1853.78247  # Carrington rotation 0 was on 13/10/1853
    car_rots = []

    for index, element in enumerate(fltyr_list):
        car_rot = (element - start_date) * (365.25 / 27.2753)
        car_rots.append(car_rot)

    return car_rots


# Takes in a single x-variable list and a list of y-variable lists to generate fits for each
def determine_fits(x_list, data_list):
    fits = []
    # x_list = sorted(x_list)

    for data in data_list:
        fit = interpolate.splrep(x_list, data)
        fits.append(fit)

    return fits


# Takes in a list of x-value lists and an associated fits list for interpolation
def evaluate_fits(x_list, fits_list):
    y_data_list = [[] for _ in range(len(fits_list))]

    for x in x_list:
        for index, fit in enumerate(fits_list):
            y_data_list[index].append(float(interpolate.splev(x, fit)))

    return y_data_list


def binned_average(x_list, y_list, bin_size):
    x_means = np.array([])
    y_means = np.array([])
    num_bins = int(len(x_list) / bin_size)

    done = False
    current_bin = 1

    while not done:
        x_values = x_list[bin_size * (current_bin - 1):bin_size * current_bin]
        y_values = y_list[bin_size * (current_bin - 1):bin_size * current_bin]

        x_mean = np.nansum(x_values) / bin_size
        y_mean = np.nansum(y_values) / bin_size

        x_means = np.append(x_means, x_mean)
        y_means = np.append(y_means, y_mean)

        current_bin += 1

        if current_bin >= num_bins:
            done = True

    return x_means, y_means


def binned_average_flag(x_list, y_list, bin_size, flags):
    x_means = np.array([])
    y_means = np.array([])
    flag_means = np.array([])

    num_bins = len(x_list) // bin_size  # Floor division

    current_bin = 1
    done = False

    # print(x_list[0], y_list[0], flags[0])

    while not done:
        x_values = x_list[bin_size * (current_bin - 1):bin_size * current_bin]
        y_values = y_list[bin_size * (current_bin - 1):bin_size * current_bin]
        flag_values = flags[bin_size * (current_bin - 1):bin_size * current_bin]

        print(flag_values)

        # Calculate mean index-wise, not adding the index if the data is flagged as bad
        bad_points = 0
        x_sum = 0
        y_sum = 0
        flag_mean = 0

        for i in range(0, len(x_values)):
            if int(flag_values[i]) == 1:
                bad_points += 1
            else:
                # print(x_values[i])
                x_sum += x_values[i]
                y_sum += y_values[i]

        if bad_points == bin_size:  # All bad points
            x_mean = np.mean(x_values)  # just the time, this is never a bad value
            y_mean = 0
            flag_mean = 1

        elif bad_points > 0 and bad_points < bin_size:  # Some bad points
            flag_mean = 2

        if bad_points != bin_size:
            x_mean = x_sum / (bin_size - bad_points)
            y_mean = y_sum / (bin_size - bad_points)

        x_means = np.append(x_means, x_mean)
        y_means = np.append(y_means, y_mean)
        flag_means = np.append(flag_means, flag_mean)

        current_bin += 1

        if current_bin >= num_bins:
            done = True

    # print(flag_means)
    return x_means, y_means, flag_means


def setup_dict(key_tuple, value_tuple):
    final_dict = {}

    for index, key_list in key_tuple:
        dummy_dict = {key_list: value_tuple[index]}
        final_dict.update(dummy_dict)

    return final_dict


def append_str(list, string):
    new_list = []

    for element in list:
        new_element = str(element) + string
        new_list.append(new_element)

    return new_list


def write_to_csv(filepath, data):
    df = pd.DataFrame(data)
    df.to_csv(filepath)


# Takes in a "date" formatted as yyyymmddhhmmss and returns a float year
def fltyr_from_string(string_date):
    fltyr_list = list(string_date)
    yr = ''.join(map(str, fltyr_list[0:4]))
    day = ''.join(map(str, fltyr_list[4:7]))
    hr = ''.join(map(str, fltyr_list[7:9]))
    # month = ''.join(map(str, fltyr_list[4:6]))
    # day = ''.join(map(str, fltyr_list[6:8]))
    # date = ''.join(map(str, fltyr_list[0:8]))
    # day = date_to_nth_day(date)
    # hr = ''.join(map(str, fltyr_list[8:10]))
    # mins = ''.join(map(str, fltyr_list[10:12]))
    # sec = ''.join(map(str, fltyr_list[12:14]))

    if int(day) > 365:
        flt_day = float(day) / 367
    else:
        flt_day = float(day) / 366

    flt_hr = float(hr) / (365.25 * 24)

    if flt_hr > 1:
        print(flt_hr)
    # flt_min = float(mins) / (365.25 * 24 * 60)
    # flt_sec = float(sec) / (365.25 * 24 * 3600)

    flt_yr = float(yr) + flt_day + flt_hr

    return flt_yr


def date_to_nth_day(date, format='%Y%m%d'):
    date = dt.datetime.strptime(date, format)
    new_year_day = dt.datetime(year=date.year, month=1, day=1)
    return (date - new_year_day).days + 1


# Turns a datetime of format year/doy/hr into a float year for hourly averaged data
def datetime_to_fltyr(datetime_string):
    fltyr_list = list(datetime_string)
    yr = ''.join(map(str, fltyr_list[0:4]))
    doy = ''.join(map(str, fltyr_list[4:7]))
    hr = ''.join(map(str, fltyr_list[7:9]))

    if float(doy) > 365:
        flt_doy = float(doy) / 366
    else:
        flt_doy = float(doy) / 365

    flt_hr = float(hr) / (24 * 365.25)
    flt_yr = float(yr) + flt_doy + flt_hr

    return flt_yr


def get_date_from_path(filepath):
    filename = os.path.basename(filepath)
    filename_array = filename.split('_')
    flt_yr = fltyr_from_string(filename_array[0])

    return flt_yr


# Takes in a string and returns the string within the '()' characters or the altered string
def strip_identifier(string, ret_mod_str):
    start = string.index('(') + 1
    end = string.index(')')
    identifier = string[start:end]
    mod_string = string[0:start - 1] + string[end:-1]

    if ret_mod_str:
        return mod_string
    else:
        return identifier


def find_duplicates(x_list):
    tally = defaultdict(list)
    for i, item in enumerate(x_list):
        tally[item].append(i)

    return ((key, locs) for key, locs in tally.items() if len(locs) > 1)


def change_invalid_values(data_array, inv_val, replacement):
    for i in range(0, len(inv_val)):
        data_array[data_array == inv_val[i]] = replacement
    return data_array


# Computes residuals of least squares regression
def ls_res(x, t, y):
    return x[0] * np.exp(-x[1] * t) * np.sin(x[2] * t) - y


def generate_trend(x_data, y_data, full_output):
    logx = np.log10(x_data)
    logy = np.log10(y_data)

    pinit = np.array([x_data[0], y_data[0]])  # First value of fit
    # print(logx, logy)
    out = optimize.leastsq(errfunc, pinit, args=(logx, logy), full_output=True)

    pfinal = out[0]
    covar = out[1]  # Only used for errors so not used here as data taken to be absolute

    # print(pfinal)

    # Power law parameters
    amp = 10 ** pfinal[0]
    index = pfinal[1]

    # Distribution parameters for KS test
    distribution = 'powerlaw'
    distr = getattr(stats, distribution)
    params = distr.fit(y_data)

    # Find the associated KS values
    f_obs = y_data
    f_exp = powerlaw(x_data, amp, index)
    ks = stats.kstest(f_exp, distribution, args=params, alternative='two-sided', mode='approx')

    if full_output:
        return [f_exp, ks, amp, index]
    else:
        return f_exp


def generate_piece_fits(x_list, y_list, n_sect):
    n = int(len(x_list) / n_sect)
    # print(n)
    sort_list = np.array(sorted(np.column_stack((x_list, y_list)), key=lambda x: x[0]))
    x_chunk_list = chunk_list(sort_list[:, 0], n)
    y_chunk_list = chunk_list(sort_list[:, 1], n)
    # for x in x_chunk_list:
    #     print(len(x))
    # for y in y_chunk_list:
    #     print(len(y))
    f_exp_list = [generate_trend(x, y, full_output=False) for x in x_chunk_list for y in y_chunk_list
                  if len(x) == len(y)]

    tot_piece_fits = np.concatenate(f_exp_list)
    # for i, x in enumerate(tot_piece_fits):
    #     print(x_list[i], tot_piece_fits[i])
    #     print("Actual: " + str(x_list[i]), str(y_list[i]))

    return tot_piece_fits


def get_minmax(data_list):
    min_v = min(np.min(data_list))
    max_v = max(np.max(data_list))

    return min_v, max_v


# Works for 1D fits but not 2D fits since the dispersion would depend on both parameters
def get_dispersion(x_data, y_data, fit, percentile):
    # Get the dispersions of the points above and below the fit line
    difference = y_data - fit
    # The two lists are ordered as (index, value) where the index is for the original dispersion list
    upper_diff = abs(np.array([(i, d) for i, d in enumerate(difference) if d > 0]))
    x_upper = np.array([x_data[int(upper_diff[i][0])] for i in range(0, len(upper_diff))])
    y_upper = np.array([upper_diff[i][1] for i in range(0, len(upper_diff))])

    lower_diff = abs(np.array(sorted([(i, d) for i, d in enumerate(difference) if d < 0], key=lambda x: x[0])))
    x_lower = np.array([x_data[int(lower_diff[i][0])] for i in range(0, len(lower_diff))])
    y_lower = np.array([lower_diff[i][1] for i in range(0, len(lower_diff))])

    print(lower_diff)
    # Get nth percentile and index & first dispersion value for upper and lower lines
    up_perc = abs(np.percentile(y_upper, percentile))
    up_idx = int((percentile / 100) * len(y_upper))
    up_first = y_upper[0]

    low_perc = abs(np.percentile(y_lower, percentile))
    low_idx = int((percentile / 100) * len(y_lower))
    low_first = y_lower[0]

    # Find slopes, y-intercepts and hence lines for upper and lower dispersions
    m_up = (up_perc - up_first) / (x_upper[up_idx] - x_upper[0])
    c_up = abs(-m_up * x_upper[0] - up_first)

    print("(" + str(x_upper[up_idx]) + ", " + str(up_perc) + "), " + "(" + str(x_upper[0]) + ", " + str(up_first) + ")")
    print(m_up, c_up)

    m_low = (low_perc - low_first) / (x_lower[low_idx] - x_lower[0])
    c_low = abs(-m_low * x_lower[0] - low_first)

    print("(" + str(x_lower[low_idx]) + ", " + str(low_perc) + "), " + "(" + str(x_lower[0]) + ", " + str(low_first) + ")")
    print(m_low, c_low)

    disp_upper = m_up * x_upper + c_up
    disp_lower = m_low * x_lower + c_low

    fit_up = np.array([fit[int(upper_diff[i][0])] for i in range(0, len(upper_diff))])
    fit_low = np.array([fit[int(lower_diff[i][0])] for i in range(0, len(lower_diff))])

    f_upper = fit_up + disp_upper
    f_lower = fit_low + disp_lower

    plt.figure()
    plt.plot(x_upper, y_upper, 'k.')
    plt.plot(x_upper, disp_upper, 'r-')
    plt.plot(x_upper[0], y_upper[0], 'g*')
    plt.plot(x_upper[up_idx], up_perc, 'b*')

    plt.xlabel("Open Flux [Mx]")
    plt.ylabel("Dispersion")
    plt.title("Upper Dispersion")

    plt.figure()
    plt.plot(x_lower, y_lower, 'k.')
    plt.plot(x_lower, disp_lower, 'r-')
    plt.plot(x_lower[0], y_lower[0], 'g*')
    plt.plot(x_lower[low_idx], low_perc, 'b*')

    plt.xlabel("Open Flux [Mx]")
    plt.ylabel("Dispersion")
    plt.title("Lower Dispersion")

    # # Get the nth percentile and the index as well as the first dispersion value
    # perc = abs(np.percentile(dispersion, percentile))
    # idx = int((percentile / 100) * len(dispersion))
    # first_val = abs(dispersion[0])
    #
    # # Define the gradient of the line and y-intercept, then draw line from first value through nth percentile
    # m = (perc - first_val) / (x_data[idx] - x_data[0])
    # c = -m * x_data[0] - first_val
    # disp_line = abs(m * x_data + c)
    #
    # # Define upper and lower dispersion around the fit
    # f_upper = fit + disp_line
    # f_lower = fit - disp_line

    # print(m, c)

    return f_upper, f_lower


def get_2d_dispersion(x1, x2, y, fit, invpercentile=5):
    dispersion = y - fit

    # Get nth percentile and index for both x values; actually it's the inverse percentile (100 - percentile)
    # For some reason I had to invert it to get the line to draw properly
    perc = abs(np.percentile(dispersion, invpercentile))
    idx = int((invpercentile / 100) * len(dispersion))
    first_val = abs(dispersion[0])

    print(perc, first_val, idx)

    # Define gradients and y-intercepts of the 2 dispersion lines then draw the lines themselves (lines on axes)
    m1 = (perc - first_val) / (x1[idx] - x1[0])
    m2 = (perc - first_val) / (x2[idx] - x2[0])

    print(x1[idx], x1[0])
    print(x2[idx], x2[0])

    c1 = -m1 * x1[0] - first_val
    c2 = -m2 * x2[0] - first_val

    disp_x1 = abs(m1 * x1 + c1)
    disp_x2 = abs(m2 * x2 + c2)

    # Now "average" the 2 lines to get the central (overall) dispersion
    disp_overall = np.average(np.array([disp_x1, disp_x2]), axis=0)

    print(m1, c1)
    print(m2, c2)

    plt.figure()
    plt.plot(x1, y, 'k.')
    plt.plot(x1, disp_x1, 'r-')
    plt.plot(x1[idx], perc, 'bx')
    plt.xlabel("Open Flux [Mx]")
    plt.ylabel("Dispersion")

    plt.figure()
    plt.plot(x2, y, 'k.')
    plt.plot(x2, disp_x2, 'r')
    plt.plot(x2[idx], perc, 'bx')
    plt.xlabel("Radial Wind Velocity [km s^-1]")
    plt.ylabel("Dispersion")


def get_chunk_dispersion(x_data, y_data, fit, percentile, n_chunks=4):
    dispersion = y_data - fit

    # Chunk lists
    x_chunks = chunk_list(x_data, n_chunks)
    disp_chunks = chunk_list(dispersion, n_chunks)

    # Get nth percentiles of chunk lists and the respective indices as well as first and final values for line drawing
    percs = [np.percentile(d, percentile) for d in disp_chunks]
    idxs = [int((percentile / 100) * len(x)) for x in x_chunks]  # Index WITHIN a chunk, not in the entire list

    first_val = disp_chunks[0][0]
    final_val = disp_chunks[-1][-1]

    first_idx = x_chunks[0][0]
    final_idx = x_chunks[-1][-1]

    # Draw lines
    disp_lines = []

    for i in range(0, len(x_chunks)):
        if i == 0:  # First chunk, draw line between very first point and 95th percentile of this chunk
            m = (percs[i] - first_val) / (idxs[i] - first_idx)

        elif i == len(x_chunks) - 1:  # Last chunk, draw line between 95th percentile of this chunk and very last point
            m = (final_val - percs[i]) / (final_idx - idxs[i])

        else:  # Middle chunks, draw line between the previous chunk and this one
            curr_idx = i * len(x_chunks) + idxs[i]
            prev_idx = (i - 1) * len(x_chunks) + idxs[i - 1]
            m = (percs[i] - percs[i - 1]) / (curr_idx - prev_idx)

        disp = [abs(m * x) / 10**22 for x in x_chunks[i]]  # Divide by 10^22 to convert from mdot vs phi to mdot vs time
        disp_lines.append(disp)

    disp_line = [item for sublist in disp_lines for item in sublist]

    f_upper = fit + disp_line
    f_lower = fit - disp_line

    return f_upper, f_lower


def format_filename(name):
    name = name.replace(" ", "_").upper()

    return name


# Tailored for the function y = a*x1^b +c*x1^d + e*x3^f
def powerlaw_func(logxlist, loga, b, d):
    logy = loga + b*logxlist[0] + d*logxlist[1]
    return logy


# p0 is a tuple of initial guesses for the parameters a-f
def powerlaw_fit(xdata_list, ydata, p0):
    # Take logs of data for power law fitting
    logxData_list = np.log10(xdata_list)
    logyData = np.log10(ydata)
    print(p0)

    # Do the curve fit
    coeffs, covars = optimize.curve_fit(powerlaw_func, logxData_list, logyData, p0=p0)

    return coeffs, covars


def get_minmax_val_idx(array, is_max):
    if is_max:  # Get max value and the index
        idx, val = max(enumerate(array), key=operator.itemgetter(1))
    else:  # Get min value and the index
        idx, val = min(enumerate(array), key=operator.itemgetter(1))

    return int(idx), val


# Find the median of an array or a truncated part of it by passing in indices to truncate
def get_median(full_array, trunc_start=0, trunc_end=-1):

    if trunc_start != 0 or trunc_end != -1:  # if array is meant to be truncated
        trunc_array = full_array[trunc_start: trunc_end]
        median = np.median(trunc_array)
        idx = np.argsort(trunc_array)[len(trunc_array)//2] + trunc_start

    else:
        median = np.median(full_array)
        idx = np.argsort(full_array)[len(full_array) // 2]

    return idx, median