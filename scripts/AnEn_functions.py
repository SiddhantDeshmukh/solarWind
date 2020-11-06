import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt


def preliminary_input_checks(t0, training_window, forecast_length, weighting_to_reacent):
    '''
    Checks the input parameters are of the correct form
    :param t0:
    :param training_window:
    :param forecast_length:
    :param weighting_to_reacent:
    :return:
    '''
    assert isinstance(t0, (datetime.datetime, pd._libs.tslibs.timestamps.Timestamp)), "t0 must be datetime object or pandas Timestamp"
    assert pd.Timestamp(1868,1,1,1,30) <= t0 <= pd.Timestamp(2017,12,31,22,30) , "t0 is not inside data period."

    assert isinstance(training_window,(int, float, np.float64, np.float32)), "training window must be a number"
    assert training_window > 0, "Positive value of training window required."

    assert isinstance(forecast_length, (int, float, np.float64, np.float32)), "forecast length must be a number"
    assert forecast_length > 0, "Positive value of forecast length required."

    assert isinstance(weighting_to_reacent, (str, list, np.ndarray)),  "weighting_to_reacent must be a string, list or array"
    if isinstance(weighting_to_reacent, str):
        assert weighting_to_reacent == 'linear' or 'quadratic' or 'cubic', "Built in weighting options include: 'linear', 'quadratic' and 'cubic"
    elif isinstance(weighting_to_reacent, (list, np.ndarray)):
        assert len(weighting_to_reacent) == int(round_down(training_window, 3)/3 +1), "weighting_to_reacent must be of length round_down(training_window, 3) +1"


def read_data(file_path):
    '''
    reads in aaH data
    :param file_path: str
    :return: dataframe of dates and data. Dates are in UNIX time
    '''
    try:
        my_data = pd.read_table(file_path, header=None, delim_whitespace=True)  # import data as pandas dataframe, separate columns at whitespace
    except FileNotFoundError:
        print("Your input file does not exist.")
        raise FileNotFoundError('Your input file does not exist.')

    my_data.columns = ['year', 'month', 'day', 'hour', '5', 'data', '7', '8', '9', '10', '11']  # name columns
    my_data['datetime'] = pd.to_datetime(my_data[['year', 'month', 'day', 'hour']])   # make colunm with timestamp
    my_data = my_data.drop(['year', 'month', 'day', 'hour', '5', '7', '8', '9', '10', '11'], axis=1) # keep only relevent colunms
    my_data['datetime'] = my_data['datetime'].astype(np.int64) // 10 ** 9 # Convert datetime into Unix time
    return my_data

def t0_interval(file_path, t0_start, t0_end):
    '''
    Usefull if running forecast for numerous ferecast times.
    creates an array of timestamps from the observation data between t0_start and t0_end
    :param file_path: str: relative path of observation data
    :param t0_start: inclusive start point of required output in UNIX time
    :param t0_end: inclusive end point of required output in Unix time
    :return: array of possible times to begin the forecast within given interval
    '''
    df_all = read_data(file_path)
    t0_array = df_all[(df_all['datetime'] >= t0_start) & (df_all['datetime'] <= t0_end)]['datetime'].values
    return t0_array


def block_out_period(df, t0, period_to_block):
    '''
    Block out a period of data around the time when the forecast is used for to avoid bias
    :param df: pandas dataframe of aaH data
    :param t0: time of forecast
    :param period_to_block: length of time to block out either side of forecast (hours)
    :return: pandas dataframes for dates and data before and after the block out period
    '''
    assert isinstance(period_to_block, (int, float, np.float64, np.float32)), "block_out length must be a number"
    assert period_to_block > 0, "Positive value of block_out required."
    start_time = t0 - period_to_block *60*60 # convert hours to seconds and add to t0
    end_time =t0 + period_to_block *60*60 # convert hours to seconds and add to t0

    df_before = df[(df['datetime'] < start_time)]
    df_after = df[(df['datetime'] > end_time)]
    return df_before, df_after


def get_observed_data(df, t0, training_window):
    '''
    Get an array of the recent time history data before forecast time t0 as the training data
    :param df: pandas dataframe of aaH data
    :param t0: Start time of forecast
    :param training_window: length of time before t0 (hours)
    :return: array of data including data at t0
    '''
    start_window = t0 - training_window*60*60 # find time that the training window starts
    observed = df[(df['datetime'] > start_window) & (df['datetime'] <= t0)]
    return observed


def square_error_matrix(data_array, observed_data):
    '''
    make a 2d array of the squared error such that it contains the square error between observations and all possible
    analogues
    :param data_array: array containing the values of aaH outside the blocked out period
    :param observed_data: array of observations during the training window
    :return: large 2D matrix of squared error
    '''
    matrix = np.full((len(data_array),len(observed_data)), np.NaN)
    for i in range(len(observed_data)): # for length of datapoints in the observed training window
        matrix[:,i] = np.roll(data_array, -i) #iteratively roll the data in adjacent cilumns such that each row represents a possible analogue
    matrix = matrix[:-(len(observed_data)-1),:] # cut the last few rows that do not contain a true timeseries due to the np.roll() moving i.e. the first observation to tthe last row with roll=-1
    matrix = matrix - observed_data #compute error between analogues and observations
    matrix = np.square(matrix) # square each error in matrix
    return matrix


def weighted_mean(squared_error_matrix, weighting_array):
    '''
    Return the weighted mean error for each potential analogue.
    :param squared_error_matrix: 2D array. Output from square_error_matrix()
    :param weighting_array: str or array: rule by which to weight the mean
    :return: 1D array of weighted mean for each potential analogue
    '''
    wmse = None
    if weighting_array:
        if weighting_array == 'linear':
            weight = np.linspace(0, 1, squared_error_matrix.shape[1]+1)[1:]  #  define array with values for weighting
            weighted_matrix = squared_error_matrix * np.array(weight)  #  multiple the matrix value by the appropraite weight
            wmse = np.mean(weighted_matrix, axis=1) # compute the mean of error of each analogue
        elif weighting_array == 'quadratic':
            weight = np.linspace(0, 1, squared_error_matrix.shape[1] + 1)[1:]**2 #define array with values for weighting
            weighted_matrix = squared_error_matrix * np.array(weight) # multiple the matrix value by the appropraite weight
            wmse = np.mean(weighted_matrix, axis=1) # compute the mean of error of each analogue
        elif weighting_array == 'cubic':
            weight = np.linspace(0, 1, squared_error_matrix.shape[1] + 1)[1:] ** 3 #define array with values for weighting
            weighted_matrix = squared_error_matrix * np.array(weight) # multiple the matrix value by the appropraite weight
            wmse = np.mean(weighted_matrix, axis=1) # compute the mean of error of each analogue
        elif isinstance(weighting_array, (list, np.ndarray)): # check if given weighting array is a list or np.ndarray
            weight = weighting_array #define array with values for weighting as the array inputed
            weighted_matrix = squared_error_matrix * np.array(weight) # multiple the matrix value by the appropraite weight
            wmse = np.mean(weighted_matrix, axis=1) # compute the mean of error of each analogue
    else:
        wmse = np.mean(squared_error_matrix, axis=1) # if no valid weighting rule is specified then compute the unweighted mean error for each analogue
    return wmse


def wmse_n(df, n):
    '''
    returns a dataframe of n analogues with the smallest values of WSME i.e. the best analogues
    to be used by wmse_n_non_overlapping()
    :param df: dataframe containing column with wmse
    :param n: number of best analogues to return
    :return: dataframe of dates and WMSE
    '''
    assert isinstance(n, (int, float, np.float32, np.float64)), "number_of_analogues must be a number."
    assert n > 0, "number_of_analogues must be greater than zero."
    #use percentile approach and call wmse_threshold() to reduce time spent in Order
    analogue_df = df.nsmallest(n, 'wmse') #cut dataframe back to n smallest wmse
    analogue_df = analogue_df.sort_values(['datetime']) # Sort dataframe into date order
    analogue_df.index = range(len(analogue_df)) #reset index as consecutive integers
    return analogue_df


def remove_overlapping_analogues(analogue_df, training_window):
    '''
    finds analogues that overlap and removes the one with highest WMSE
    to be used by wmse_n_non_overlapping()
    :param analogue_df: dataframe containing best analogues
    :param training_window:
    :return: dataframe of analogues which do not overlap in training window
    '''
    my_df = analogue_df.copy(deep=True) #take a copy of dataframe
    my_df.index = range(len(my_df)) #reset index as consecutive integers
    i=0
    while i < len(my_df)-2: #This routine sorts through the dataframe to ensure none of the analogues overlap in time
        # and while they do overlap it deletes the analogue with largest wmse until no analogues are overlapping.
        # Uses premise that if the first analogue does not overlap with the second then the first analogue does not
        # overlap with any other analogues

        while (i < len(analogue_df)-2) & (abs( my_df['datetime'].iloc[i+1] - my_df['datetime'].iloc[i] ) < 3600 * training_window): #compare two analogues that are next to eachother in time to see if they are further apart than training window
            my_df.index = range(len(my_df))#reset index as consecutive integers
            if (my_df['wmse'].iloc[i+1] < my_df['wmse'].iloc[i]) & (i < len(analogue_df)-2):  #find which analogue is best
                my_df = my_df.drop(i) #delete worste analogue
                my_df.index = range(len(my_df)) #reset index as consecutive integers
            elif (my_df['wmse'].iloc[i+1] >= my_df['wmse'].iloc[i]) & (i < len(analogue_df)-2): #find which analogue is best
                my_df = my_df.drop(i + 1) #delete worste analogue
                my_df.index = range(len(my_df))#reset index as consecutive integers
            my_df.index = range(len(my_df)) #reset index as consecutive integers
        
        i = i+1 #change which analogue is compared to the others
        if (i < len(analogue_df)-2):
            break
    return my_df

def wmse_n_non_overlapping(number_of_analogues, wmse, df_blocked_out, training_window):
    '''
    Uses wmse_n() and remove_overlapping_analogues() to decide which are the best n analogues that do not overlap
    :param number_of_analogues:
    :param wmse: array of weighted mean wmse for each analogue
    :param df_blocked_out: dataframe containing all aaH data exept that in the blocked out period
    :param training_window:
    :return: dataframe of n analogues which do not overlap
    '''
    df_blocked_out['wmse'] = wmse # Create a pandas dataframe with wmse as a column
    analogue_df = wmse_n(df_blocked_out, number_of_analogues*1) # find best n analogues
    analogue_df = remove_overlapping_analogues(analogue_df, training_window) #remove analogues that overlap in time

    if len(analogue_df) == number_of_analogues: # check analogue_df still contains n analogues
        pass
    elif len(analogue_df) > number_of_analogues: # if there are too many analogues then choose n best
        analogue_df = wmse_n(analogue_df, number_of_analogues)
    else: #else there are too few analogues
        mult = 2
        while (len(analogue_df) < number_of_analogues) & (mult < len(df_blocked_out)/training_window): #iterate until there are n or more analogues that do not overlap
            #print('Less')
            #print('mult = ' + str(mult))
            analogue_df = wmse_n(df_blocked_out, number_of_analogues * mult) # find best n*mult analogues
            analogue_df = remove_overlapping_analogues(analogue_df, training_window) #remove analogues overlapping in time
            mult = mult * 2 #increase value of mult
        analogue_df = wmse_n(analogue_df, number_of_analogues) # cut down to n best analogues
    return analogue_df


def get_analogues(analogue_df, all_data_df, training_window, lead_time, temporal_resolution):
    '''
    creates a 2d array of analogues for t0<t<= t0+lead_time with their data values
    :param analogue_df: dataframe containing analogue information (datetime of analogue is used here)
    :param all_data_df: dataframe containing the whole timeseries of data
    :param training_window: length of time the model is trained on (hours)
    :param lead_time: time the model forecasts for (hours)
    :return: matrix of analogue datapoints
    '''
    start_time = analogue_df['datetime'] - training_window *3600 #array of times when the analogue training period begins
    end_time = analogue_df['datetime'] + lead_time * 3600 # array of times when lead time window ends

    num_points = int(round_down(training_window, temporal_resolution) / temporal_resolution  + round_down(lead_time, temporal_resolution) / temporal_resolution) #number of 3-hourly datapoints each analogue period contains
    analogue_matrix = np.full((len(analogue_df), num_points), np.NaN)

    # print(analogue_matrix.shape)
    
    # print(num_points)
    for i in range(len(analogue_df)):
        # print(-(start_time[i] - end_time[i]) / 3600)
        analogue = all_data_df[(all_data_df['datetime'] > start_time.iloc[i]) & (all_data_df['datetime'] <= end_time.iloc[i])] #get data for each point in analogue period
        # try:
        analogue_matrix[i] = analogue['data'].values #save into a matrix
        # except: #likely due to analogue period running off end of data or into the blocked out period
        #     nan_length = int(int(round_down(lead_time, temporal_resolution) / temporal_resolution) + int(round_down(training_window, 3) / 3) + 1 - len(analogue['data'].values)) # find how many values cannot be found
        #     analogue_matrix[i] = np.concatenate((analogue['data'].values, np.full(nan_length,np.NaN))) #saves the data values that can be found along with NaNs in the output array
    
    return analogue_matrix


def analogue_median(analogue_matrix):
    '''
    computes median of analogues which is the forecast
    :param analogue_matrix:
    :return: forecast array
    '''
    analogue_median = np.nanmedian(analogue_matrix, axis=0)
    return analogue_median


def round_down(num, divisor):
    '''
    rounds down to the nearest "divisor"
    :param num:
    :param divisor:
    :return:
    '''
    return num - (num%divisor)


def observed_after_t0(df, t0, lead_time):
    '''
    get an array of the data observed after  forecast time
    :param df: dataframe containing entire dataseries
    :param t0: time of forecast
    :param lead_time: length of time after t0 (hours)
    :return: array of data including data at t0
    '''
    end_window = t0 + lead_time * 3600
    observed_after = df[(df['datetime'] > t0) & (df['datetime'] <= end_window)]
    return observed_after


def absolute_difference(observed, forecasted):
    '''
    Computes the absolute difference between observations and forecast.
    :param observed: array of observed data
    :param forecasted: array of forecast data
    :return:
    '''
    difference = np.subtract(observed, forecasted)
    abs_diff = np.absolute(difference)
    return abs_diff




def main(file_path, t0, training_window, forecast_length, weighting_to_recent=False,
         number_of_analogues=10, block_out=False, temporal_resolution=3):
    '''
    uses above functions to operate the analogue forecast
    :param file_path: str: file containing aaH dataset
    :param t0: pandas timestamp: start point of forecast
    :param training_window: int: length of recent observation for which to find analogues for
    :param forecast_length: maximum lead time of forecast
    :param weighting_to_recent: rule to say how much more important the most reacent observations are when finding analogues
    :param number_of_analogues: number of analogues to be found
    :param block_out: specify amount of time to block out. If False will default to (training_window+forecast_length)*2
    :param temporal_resolution: int: time resolution of data in hours. Defaults to 3 hours for aaH
    :return: observations and analogues
    '''
    preliminary_input_checks(t0, training_window, forecast_length, weighting_to_recent)
    
    training_window = round_down(training_window, temporal_resolution)
    forecast_length = round_down(forecast_length, temporal_resolution)

    t0 = t0.value//10**9 # convert to unix time
    df_all = read_data(file_path)
    if block_out:
        df_before, df_after = block_out_period(df_all, t0,  block_out)
        print(block_out)
    else:
        df_before, df_after = block_out_period(df_all, t0, (training_window+forecast_length) *2)
        print((training_window+forecast_length) *2)

    observed = get_observed_data(df_all, t0, training_window)

    square_error_before = square_error_matrix(df_before['data'].values, observed['data'].values)
    square_error_after = square_error_matrix(df_after['data'].values, observed['data'].values)

    square_error = np.concatenate([square_error_before, square_error_after]) # use before and after blocked out period so that an analogue cannot cross the blocked out period

    df_blocked_out = pd.concat([df_before.iloc[len(observed)-1:], df_after.iloc[len(observed)-1:]])

    wmse = weighted_mean(square_error, weighting_to_recent)

    analogue_df = wmse_n_non_overlapping(number_of_analogues, wmse, df_blocked_out, training_window)

    analogue_matrix = get_analogues(analogue_df, df_blocked_out, training_window, forecast_length, temporal_resolution)

    analogue_weighted_median = analogue_median(analogue_matrix)

    observed_after = observed_after_t0(df_all, t0, forecast_length)

    return observed['data'].values, observed_after['data'].values, analogue_weighted_median, analogue_matrix, analogue_df


def plot_analogue_forecast(file_path, t0, training_window, forecast_length, weighting_to_recent=None,
                          number_of_analogues=10, block_out=False, temporal_resolution=3):
    '''
    Wrapper around main with code to plot the output
    :param file_path: str: file containing aaH dataset
    :param t0: pandas timestamp: start point of forecast
    :param training_window: int: length of recent observation for which to find analogues for
    :param forecast_length: maximum lead time of forecast
    :param weighting_to_recent: rule to say how much more important the most reacent observations are when finding analogues
    :param number_of_analogues: number of analogues to be found
    :param block_out: specify amount of time to block out. If False will default to (training_window+forecast_length)*2
    :param temporal_resolution: int: time resolution of data in hours. Defaults to 3 hours for aaH
    :return: plot of analogue ensemble forecast
    '''
    training_window = round_down(training_window, temporal_resolution)
    forecast_length = round_down(forecast_length, temporal_resolution)
    observed, observed_after, analogue_weighted_median, analogue_matrix, analogue_df = main(file_path, t0,
                                                                                            training_window,
                                                                                            forecast_length,
                                                                                            weighting_to_recent,
                                                                                            number_of_analogues,
                                                                                            block_out,
                                                                                            temporal_resolution)
    t0_index = int(round_down(training_window, temporal_resolution) / temporal_resolution)

    fig, ax = plt.subplots()
    num_points = len(analogue_weighted_median)

    print(t0_index, num_points)

    xgrid = temporal_resolution * np.linspace(-t0_index+1, num_points - t0_index, num_points)
    ax.axvline(0, linestyle='--', color='lightgrey')
    ax.plot(xgrid, np.transpose(analogue_matrix), color='lightgrey')
    ax.plot(xgrid, np.concatenate((observed, observed_after)), label = 'Observations', color='k')
    ax.plot(xgrid, analogue_weighted_median, label='Analogue Median', color='#A11E22')
    ax.legend()
    ax.set_xlabel(r'Time from $t_0$ (hours)')
    ax.set_ylabel(r'$aa_H (nT)$')
    plt.savefig('test.svg')


if __name__ == "__main__":
    plot_analogue_forecast(file_path='../data/aaH_data.txt',
                           t0 = pd.Timestamp(2017, 12, 5, 10, 30),
                           training_window=24,
                           forecast_length=24,
                           weighting_to_recent='quadratic',
                           number_of_analogues=50,
                           block_out=False,
                           temporal_resolution=3)
