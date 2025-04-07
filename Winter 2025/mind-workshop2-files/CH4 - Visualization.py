from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy import signal
import mne

#RAW DATA TO USABLE DATA

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

# PLOTTING DATA

channel = 0
channel_data = eeg_data.iloc[:,channel]
# .iloc is a pandas method used for selecting data by row and column indices.
# iloc[rows,columns] where : selects all rows and channel, specified the channel we selected (aka 0)

# now we can create a plot for the data

plt.plot(channel_data)
plt.title("Signal Data from Channel " + str(channel + 1))
plt.ylabel('EEG, µV')
plt.xlabel('Sample')
plt.show()

print(eeg_data.shape[0]) #index 0 represents the number of rows and index 1 represents the number of columns


#BP FILTERING

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

channel_data_bp_filtered = channel_data.copy()  # copy the data

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

# HEATMAP


plt.figure(figsize=(10,6))
#Creates a new figure with a specified size (10 inches wide and 6 inches tall)

heatmap = plt.imshow(eeg_data_filtered.T,cmap="viridis",aspect = "auto")
# .T: Transposes the DataFrame so that channels are plotted along the y-axis and time along the x-axis.
# cmap='viridis': Uses the "Viridis" color map, where different colors represent different EEG amplitude ranges.
# aspect='auto': Automatically adjusts the aspect ratio for a better fit.

plt.colorbar(heatmap,label="EEG Amplitude")
# Adds a color bar to the side of the heatmap.
# The color bar provides a legend for interpreting the amplitude values

plt.xlabel("Time")
plt.ylabel("Channel #")
plt.title("EEG Heatmap")
# x and y axis labelling and title as well

plt.show()


# TOPOMAP

#MNE LIBRARY APPLICATION
"""
Here we make preparation to convert our dataset to MNE library format
"""

# Convert the filtered EEG data to a NumPy array (MNE requires data in array format)
data_after_band_pass_filter = np.array(eeg_data_filtered)

# Reshape the array to ensure it matches the expected shape for MNE processing.
# (8 channels, 7120 time points per channel in this example)
data_after_band_pass_filter = data_after_band_pass_filter.reshape((8, 7120))

# Convert the reshaped array back to a DataFrame for easier manipulation or inspection if needed.
data_after_band_pass_filter = pd.DataFrame(data_after_band_pass_filter)

# Create a standard EEG electrode montage.
# A montage defines the spatial arrangement of EEG electrodes on the scalp.
# 'standard_alphabetic' is a predefined configuration provided by MNE.
standard_montage = mne.channels.make_standard_montage('standard_alphabetic')

# Define the number of EEG channels (electrodes).
# In this case, we are using 8 channels.
n_channels = 8

# Create fake metadata (info object) required for MNE to process the EEG data.
# This includes channel names, sampling frequency, and channel types.
fake_info = mne.create_info(
    ch_names=["Fp1", "Fz", "Cz", "Pz", "T3", "C3", "C4", "T4"],  # Names of the 8 channels
    sfreq=250.,  # Sampling frequency in Hz (e.g., 250 samples per second)
    ch_types='eeg',  # Specify that the data type is EEG

)

# Print the 'info' object to verify its contents.
# This displays metadata about the EEG data, such as channel names and sampling frequency.
print(fake_info)

# Create an Evoked object from the provided EEG data and metadata.
# An Evoked object in MNE represents averaged data from multiple trials or epochs.
# Here, the `data_after_band_pass_filter` is the EEG data (array), and `fake_info` contains metadata.
fake_evoked = mne.EvokedArray(data_after_band_pass_filter, fake_info)

# Attach the standard montage (electrode arrangement) to the Evoked object.
# 'on_missing="ignore"' ensures the function doesn't throw errors if some channel locations are missing in the montage.
fake_evoked.set_montage(standard_montage, on_missing='ignore')

times_to_plot = np.arange(0, 28., 4)
# plot topo maps at various times
fake_evoked.plot_topomap(times_to_plot, ch_type="eeg", ncols=len(times_to_plot), nrows="auto")