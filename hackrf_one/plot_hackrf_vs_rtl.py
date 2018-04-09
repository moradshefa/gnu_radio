import numpy as np
import matplotlib.pyplot as plt 
from numpy import *
from numpy.fft import *
from matplotlib.pyplot import *
import scipy.signal as signal

data_from_hackrf = False

# Set parameters how data was collected
center_freq = 99.7e6
sampling_rate = 2400000
time_duration = 10
N_Samples = sampling_rate * time_duration


# open file and return signal
def open_file(filename, data_from_hackrf = False):
	dat = []
	if data_from_hackrf:
		dat = np.fromfile(filename, np.int8)
		dat = np.uint8(dat)

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


def get_freq_axis(interval = 2048, fs = sampling_rate, fc = center_freq):
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
def plot_average_pspect(y, interval = 2048, fs = sampling_rate, fc = center_freq):
	f = get_freq_axis(interval = 1024, fs = sampling_rate, fc = center_freq)
	Pspect = average_pspect(y, interval = 1024)
	width, height = figaspect(0.2)
	fig=plt.figure(figsize=(width,height))
	p = plt.semilogy(f, Pspect)
	return p

# plots of average pspect of a list of signals
def plot_average_pspect_multiple_signals(y_signals, interval = 2048, fs = sampling_rate, fc = center_freq):
	num_signals = len(y_signals)
	f = get_freq_axis(interval = interval, fs = fs, fc = fc)
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

filename_rtl = str(int(center_freq)) + "rtl"
y1 = open_file(filename_rtl, data_from_hackrf = False)


filename_hackrf = "output.sb"
y2 = open_file(filename_hackrf, data_from_hackrf = False)

filename_hackrf = "output.ub"
y3 = open_file(filename_hackrf, data_from_hackrf = False)


plot_average_pspect_multiple_signals([y1, y2, y3], interval = 1024, fs = sampling_rate, fc = center_freq)
######################################
######################################



