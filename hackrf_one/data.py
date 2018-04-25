from utils import *
import os

# This file calculates ave power spectrum 
# for each signal that we collected data and saves
# them in a dict ave_pspect_signals. 
# The key is the name of what kind of signal it is
# and the value is an np 2D array where each
# row is a pspect of a center_freq of length interval

interval = 2048

# Save tuples of data folders and sampling rates
paths = []
paths.append(('unclassified_data', 2000000))
paths.append(('fm_data', 2000000))
paths.append(('wifi_data', 20000000))

ave_pspect_signals = {}
center_freqs = []
for path, sampling_rate in paths:
	print(path, sampling_rate)

	files = os.listdir(path)

	ave_pspect = np.zeros((len(files), interval))

	for i, filename in enumerate(files):
		try:
			center_freq = int(filename)
			print("     ", center_freq)
			y = open_file(path + '/' + filename, data_from_hackrf = True)
			ave_pspect[i] = average_pspect(y, interval = interval)
			center_freqs.append((path + '/' + filename, center_freq))
		except Exception as ex:
			print(filename, " not a correct file. Please delete and rerun.")
	ave_pspect_signals[path] = ave_pspect

for key in ave_pspect_signals.keys():
	print(key, ave_pspect_signals[key].shape)

np.save("ave_pspect_signals", (ave_pspect_signals, center_freqs))
