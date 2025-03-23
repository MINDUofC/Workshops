from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy import signal
import mne

eeg_data = pd.read_excel("tempData.xlsx")


# convert from Digital Value of Analog Digital converter (ADC) ADS1299 to microvolts

"""
converts raw EEG data from a 24-bit ADC (ranging from 0 to 16777215) into microvolts by normalizing the data,
scaling it to a 4.5V reference voltage, and converting to microvolts,
then rounding to two decimal places. This transformation is necessary for accurately interpreting EEG signals, which are typically measured in microvolts
"""
#conversion to microvolts 4.5V to 4.5 millions microV
Volts_to_microVolts = 1000000*4.5
ADC_To_microVolts = eeg_data/16777215

eeg_data = round(Volts_to_microVolts*ADC_To_microVolts,2)  # 2 to the power of 24 = 16777216

def butter_highpass_filter(data, cutoff, nyq, order=5):
    """Butterworth high-pass filter.
    Args:
        data (array_like): data to be filtered.
        cutoff (float): cutoff frequency.
        order (int): order of the filter.
        nyq (int): half of the sampling frequency according to the Nyquist frequency,
        which allows for a signals full reconstruction as long as it contains half or less of the original
        sampling frequency.
    Returns:
        array: filtered data."""
    normal_cutoff = cutoff / nyq  # normalized cutoff frequency
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data


def butter_lowpass_filter(data, cutoff, nyq, order=5):
    """Butterworth low-pass filter.
    Args:
        data (array_like): data to be filtered.
        cutoff (float): cutoff frequency.
        order (int): order of the filter.
    Returns:
        array: filtered data."""
    normal_cutoff = cutoff / nyq  # normalized cutoff frequency
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = signal.lfilter(b, a, data)
    return filtered_data



#EXAMPLE OF BP FIlTERING

channel_data_bp_filtered = eeg_data.iloc[:,0]  #channel 1 data

fps = 250  # sampling frequency
cutoff_high = 1  # cutoff frequency of the high-pass filter
cutoff_low = 10  # cutoff frequency of the low-pass filter
nyq = 0.5 * fps  # Nyquist's frequency (half of the sampling frequency)

# apply the band-pass filter
# the band-pass filter is just both the butter highpass and low pass combined

channel_data_bp_filtered = butter_highpass_filter(
    data=channel_data_bp_filtered,
    cutoff=cutoff_high,
    nyq=nyq,
    order=5)
channel_data_bp_filtered = butter_lowpass_filter(
    data=channel_data_bp_filtered,
    cutoff=cutoff_low,
    nyq=nyq,
    order=4)






# AVERAGE FILTER - Smoots out the signal by replacing each data point with the average value of neighboring data (within a define window)

#filter all data via bandpass

channel = 0
def average_filter(data, average_length):
    """Average filter.
    Args:
        data (array_like): data to be filtered.
        average_length (int): length of the filter window.
    Returns:
        array: filtered data."""
    data = pd.DataFrame(data)
    averaged_data = data.rolling(window=average_length).mean()
    return averaged_data

# Note that  the main issue with average filtering is that their is a right shift in the signal, aka a little delay,
# the peaks are delayed for later times

# Note that .rolling from the pandas library allows us to perform rolling window calculations.
# Specifically it allows us to do in this context,rolling averages, using the current row combined with N previous/fwd rows
# As you may assume N is the window = in the arguments of the function.
# Example:
# Dataset: [1, 2, 3, 4, 5]
# Window size: 3
# Operation: Calculate the average.
# First window: [1, 2, 3] → Average = (1+2+3)/3 = 2
# Second window: [2, 3, 4] → Average = (2+3+4)/3 = 3
# Third window: [3, 4, 5] → Average = (3+4+5)/3 = 4
# you can control the "smoothness" of the data by changing the window size, larger window = smoother
average_length_window_1 = 40
average_length_window_2 = 70

#window of 40 rows
channel_data_avg_window_1 = average_filter(data=channel_data_bp_filtered, average_length=average_length_window_1)

#window of 70 rows
channel_data_avg_window_2 = average_filter(data=channel_data_bp_filtered, average_length=average_length_window_2)


plt.title("Average/Mean filter, Channel " +  str(channel+1))
plt.plot(channel_data_bp_filtered, label='Data before average filter')
plt.plot(channel_data_avg_window_1, label='Data after average filter (window='+str(average_length_window_1)+")")
plt.plot(channel_data_avg_window_2, label='Data after average filter (window='+str(average_length_window_2)+")")
plt.ylabel('EEG, µV')
plt.xlabel('Sample')
plt.legend(loc='upper left')
plt.xlim([0, 2000])  # zoom in the data
plt.show()
# Very clear that it gets smoother as window size increases

# GUASSIAN FILTER

from scipy.ndimage import gaussian_filter1d # this is the specific built-in function for guassian 1D filtering from scipy

# you can control the "smoothness" of the data by changing
# the kernel's standard deviation value,
sigma_value_1 = 45 # standard deviation of the Gaussian kernel
sigma_value_2 = 15 # standard deviation of the Gaussian kernel

channel_data_gauss_window_1 = gaussian_filter1d(channel_data_bp_filtered, sigma=sigma_value_1)
channel_data_gauss_window_2 = gaussian_filter1d(channel_data_bp_filtered, sigma=sigma_value_2)

# plot the data
plt.title("Gaussian filter, Channel " +  str(channel+1))
plt.xlabel("Sample")
plt.ylabel("EEG, µV")
plt.plot(channel_data_bp_filtered, label='Data before Gaussian filter')
plt.plot(channel_data_gauss_window_1, label='Data after Gaussian filter (sigma='+str(sigma_value_1)+")")
plt.plot(channel_data_gauss_window_2, label='Data after Gaussian filter (sigma='+str(sigma_value_2)+")")
plt.legend(loc='upper left')
plt.xlim([0, 2000])  # zoom in the data
plt.show()
# Very clear that higher Std Deviation of the kernel increases smoothness, and lower std dev preserves small details
# of original signal


#MEDIAN FILTER
from scipy.ndimage import median_filter

# you can control the "smoothness" of the data by changing the window size
size_window_1 = 100
size_window_2 = 50

channel_data_median_window_1 = median_filter(channel_data_bp_filtered, size=size_window_1)  # size It defines the size of the window for the median filter operation
channel_data_median_window_2 = median_filter(channel_data_bp_filtered, size=size_window_2)

# Rolling Window is also used in median filter, example below
"""The Scipy median filter also uses a rolling window calculation as well, but calculates medians rather than means
How It Works:
Dataset: [1, 2, 6, 4, 3]
Window size: 3
Operation: Calculate the median.
First window: [1, 2, 6] → Median = 2
Second window: [2, 6, 4] → Median = 4
Third window: [6, 4, 3] → Median = 4
The window slides across the dataset, and the median is computed for each position."""


# plot the data
plt.title("Median filter, Channel " +  str(channel+1))
plt.xlabel("Sample")
plt.ylabel('EEG, µV')

plt.plot(channel_data_bp_filtered, label='Data before median filter')
plt.plot(channel_data_median_window_1, label='Data after median filter (size='+str(size_window_1)+")")
plt.plot(channel_data_median_window_2, label='Data after median filter (size='+str(size_window_2)+")")
plt.legend(loc='upper left')
plt.xlim([0, 2000])  # zoom in the data, however this is optional as we can manually zoom as well once code runs
plt.show()
# Once again larger window creates a smoother signal post filtering, overall we can see that window 10 is too small
# it essentially just mimics the orginal signal, while window 100, is too much, as it diminishes and essentially
# flattens the signal
