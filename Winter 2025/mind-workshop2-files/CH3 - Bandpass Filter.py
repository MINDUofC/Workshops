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


print(channel_data_bp_filtered)

plt.plot(channel_data_bp_filtered)
plt.title("Channel Data after Band-pass Filter " + str(cutoff_high) + "-" + str(cutoff_low) + "Hz")
plt.ylabel('EEG, µV')
plt.xlabel('Sample')
plt.show()

#filter all data via bandpass

eeg_data_filtered = eeg_data.copy()
for channel in range(eeg_data.shape[1]): #0 = row, 1 = columns aka channels

    eeg_data_filtered.iloc[:, channel] = butter_highpass_filter(
        data=eeg_data.iloc[:, channel],
        cutoff=cutoff_high,
        nyq=nyq,
        order=5)
    eeg_data_filtered.iloc[:, channel] = butter_lowpass_filter(
        data=eeg_data_filtered.iloc[:, channel],
        cutoff=cutoff_low,
        nyq=nyq,
        order=4)

print(eeg_data_filtered)
