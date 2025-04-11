from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy import signal


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
cutoff_high =  1  # cutoff frequency of the high-pass filter
cutoff_low = 10 # cutoff frequency of the low-pass filter
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


# FFT - FAST FOURIER TRANSFORM
# FFT is a mathematical algorithm that converts time-domain EEG signals into the frequency domain. It does this by
# breaking down the EEG data into its constituent sinusoidal frequency components, revealing the strength of each
# frequency’s presence in the signal. This process helps identify dominant frequencies and their distribution, offering
# insights into the underlying neural oscillations present in the EEG data.

from numpy.fft import fft #direct import for fft function from numpy

data_len = len(channel_data_bp_filtered)  # number of observations
fourier_transform = fft(channel_data_bp_filtered)  # compute FFT


"""
Why Normalize FFT Values?
When you compute an FFT, the raw output is scaled by factors related to the number of points in the input signal and the
specific implementation of the FFT algorithm. Without normalization, the amplitude values of the FFT output may not 
directly correspond to the actual signal amplitudes.

Common Normalization Methods
Divide by the Number of Points (N):
If the FFT is performed on a signal with N points, dividing the FFT output by N scales the result back to the correct 
amplitude.

FFT OUTPUT / N = NORMALIZE OUTPUT
"""

fourier_transform = fourier_transform / data_len  # normalize values
fourier_transform = fourier_transform[:int(data_len/2)]  # take half of the data

time_period = data_len / fps  # time period of the signal
values = np.arange(int(data_len/2))  # x-axis values up to Nyquist frequency

frequencies = values / time_period  # x-axis values in Hz

# plot the frequency spectrum
plt.plot(frequencies, abs(fourier_transform))
plt.title("Frequency spectrum of Channel 1")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.xlim([0, 20])
plt.show()



# HILBERT TRANSFORM

channel = 0

from scipy.signal import hilbert  #libary to get the data from

analytic_signal = hilbert(channel_data_bp_filtered)  # apply Hilbert transform
amplitude_envelope = np.abs(analytic_signal)  # compute amplitude envelope aka take the abs value and remove imaginary portions

# plot the amplitude envelope
plt.plot(amplitude_envelope)
plt.title("Hilbert Transform , Channel " + str(channel+1))
plt.ylabel('EEG, µV')
plt.xlabel('Sample')
plt.show()
