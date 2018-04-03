# collects data that we group with group_data.py
# plots average power spectrum to confirm whether it's a 
# valid or invalid fm signal

import numpy as np
import matplotlib.pyplot as plt 
from numpy import *
from numpy.fft import *
from matplotlib.pyplot import *
from rtlsdr import RtlSdr

# if these imports fail might need to install pyrtlsdr, pyModes
# pip3 install pyrtlsdr 
# pip3 install pyModeS 

sdr = RtlSdr()

# Set parameters
sampling_rate = 960000
center_freq = 99.701e6
# print(sdr.valid_gains_db)
gain = 40.2

sdr.set_sample_rate(sampling_rate)  
sdr.set_center_freq(center_freq)
sdr.set_gain(gain)

time_duration = 4 # if noisy plot try longer duration
N_Samples = sampling_rate * time_duration
y = sdr.read_samples(N_Samples) # comment out after collecting data
# y = np.load(str(int(center_freq)) + ".npy")	# uncomment after collecting data
sdr.close()

interval = 2048
chunks = N_Samples//interval
N = interval * chunks

y = y[:N]
# np.save(str(int(center_freq)), y)	# comment out after collecting data

# Calculate average power spectrum
y = y[:len(y//interval*interval)]
y = y.reshape(N//interval, interval)
y_windowed = y*np.kaiser(interval, 6)
Y = fftshift(fft(y_windowed,axis=1),axes=1)

Pspect = mean(abs(Y)*abs(Y),axis=0);

# Depending on the signal being a valid fm or invalid fm signal uncomment one of
# the next 2 lines

# np.save("pspect_invalid_fm_data/pspect_invalid_" + str(int(center_freq)), Pspect)
np.save("pspect_valid_fm_data/pspect_valid_" + str(int(center_freq)), Pspect)

# Plot
f = r_[-interval/2:interval/2]/interval*sampling_rate + center_freq
width, height = figaspect(0.2)
fig=plt.figure(figsize=(width,height))
p = plt.semilogy( f/1e6,Pspect)
plt.title("Frequency Spectrum")
plt.xlabel('frequency [MHz]')
plt.title('Average Power Spectrum')
plt.show()