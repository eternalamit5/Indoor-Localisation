import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt


def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter( order, normal_cutoff, btype='low', analog=False )
    filtered_data = lfilter(b, a, data)
    sample_number = []
    for i in range(0, len(data)):
        sample_number.append(i)
    return filtered_data,sample_number


def read_motion_measurements(filepath=None, cols=None):
    measurements = pd.read_csv( filepath, usecols = cols )
    return measurements


def plot2d(x, y, title="", legend="", overwrite=True):
    if not overwrite:
        fig = plt.figure()
        subplot1 = fig.add_subplot( 111 )
        subplot1.title( title )
        subplot1.plot( x, y, label=legend )
    else:
        plt.title( title )
        plt.plot( x, y, label=legend )


acc_meas_xyz = read_motion_measurements(filepath="Dataset/shan_test_01_walk.csv", cols=['accx', 'accy', 'accz'])
filtered_data, sample_count = butter_lowpass_filter(data = acc_meas_xyz['accy'], cutoff=10, fs=80, order=5)
plot2d(x=sample_count, y=acc_meas_xyz['accx'], title="Filter measurement", legend="measurement")
plot2d(x=sample_count, y=filtered_data, title="Filter measurement", legend="Filtered values")
plt.legend()
plt.show()


