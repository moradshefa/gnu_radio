import numpy as np
import matplotlib.pyplot as plt 
from numpy import *
from numpy.fft import *
from matplotlib.pyplot import *
import scipy.signal as signal

# example of how fm data was collected
# hackrf_transfer -r 88300000 -f 88300000 -s 2000000 -n 10000000 -a 1 -l 16 -g 20

# center_freq = 88.3e6
# sampling_rate = 2e6
# time_duration = 50

# example of how wifi data was collected
# hackrf_transfer -r 2407300000 -f 2407300000 -s 20000000 -n 60000000 -a 1 -l 16 -g 20
# center_freq = 2407300000
# sampling_rate = 20000000
# time_duration = 3


# open file and return signal
def open_file(filename, data_from_hackrf = False):
	if data_from_hackrf:
		dat = np.fromfile(filename, np.int8)
		iq = np.array(dat[::2] + 1j * dat[1::2])
		iq /= (255/2)
		return iq
	else:
		dat = np.fromfile(filename, np.uint8)
		iq = dat.astype(np.float32).view(np.complex64)
		iq /= (255/2)
		iq -= (1 + 1j)
		return iq

# returns the array you need to multiply a signal with a given sampling_rate by to modulate it by f0
def modulation_freq_array(length, f0, sampling_rate):
	return np.exp(1j*2*np.pi*f0* np.arange(0, length)/sampling_rate)

# modulates a signal in time such that the freq is shifted by f0
def modulate(sig, f0, sampling_rate):
	l = len(sig)
	arg = modulation_freq_array(l, f0, sampling_rate)
	return sig*arg


def get_freq_axis(fs, fc, interval = 2048):
	return r_[-interval/2:interval/2]/interval*fs + fc

# returns the freq axis and average powerspectrum of a signal
def average_pspect(y, interval = 2048):
	N_Samples = len(y)
	chunks = N_Samples // interval
	N = chunks * interval
	y = y[:N].reshape(chunks, interval)
	y_windowed = y * np.kaiser(interval, 6)
	Y = fftshift(fft(y_windowed,axis=1),axes=1)
	Pspect = mean(abs(Y)*abs(Y),axis=0)
	return Pspect

# returns the plot of average pspect of a signal
def plot_average_pspect(y, fs, fc, interval = 2048):
	f = get_freq_axis(fs = fs, fc = fc, interval = interval)
	Pspect = average_pspect(y, interval = interval)
	width, height = figaspect(0.2)
	fig=plt.figure(figsize=(width,height))
	p = plt.semilogy(f, Pspect)
	return p

# plot data passing in average power spectrum and freq axis
def plot_given_pspect(f, pspect):
	width, height = figaspect(0.2)
	fig=plt.figure(figsize=(width,height))
	p = plt.semilogy(f, pspect)
	return p

# plots of average pspect of a list of signals
def plot_average_pspect_multiple_signals(y_signals, fs, fc, interval = 2048):
	num_signals = len(y_signals)
	f = get_freq_axis(fs = fs, fc = fc, interval = interval)
	width, height = figaspect(0.2)
	fig=plt.figure(figsize=(width,height))

	pspect = []
	for i, signal in enumerate(y_signals):
		signal_ave_pspect = average_pspect(signal, interval = interval)
		pspect.append(signal_ave_pspect)
		plt.subplot(num_signals, 1, i+1)
		p = plt.semilogy(f, signal_ave_pspect)

	plt.show()

######################################
######################################




# Set parameters how data was collected
# center_freq = 92500000
# sampling_rate = 2000000


# time_duration = 3
# N_Samples = sampling_rate * time_duration


# filename_1 = str(center_freq)
# y1 = open_file(filename_1, data_from_hackrf = True)

# plot_average_pspect_multiple_signals([y1], fs = sampling_rate, fc = center_freq, interval = 1024)
######################################
######################################



