# takes all valid and invalid fm freqs that we have data collected
# and builds "fm_data.npy" file that the classifier uses

import numpy as np

interval_length = 2048

valid_center_freqs = [88.3e6, 88.4e6, 88.5e6, 88.7e6, 91.1e6, 92.3e6, 92.4e6, 92.5e6, 94.7e6, 94.8e6, 94.9e6, 95e6,95.1e6,96.3e6,96.4e6,96.5e6, 96.6e6,97.2e6, 97.3e6,97.4e6, 97.5e6, 98.5e6, 99.4e6, 99.5e6, 99.6e6, 99.7e6, 99.8e6, 99.9e6, 102.1e6, 102.8e6, 102.9e6, 103e6, 103.1e6,104.9e6, 106.1e6]										


Y = np.zeros((len(valid_center_freqs), interval_length))


for i, center_freq in enumerate(valid_center_freqs):
	Y[i] = np.load("pspect_valid_fm_data/pspect_valid_" + str(int(center_freq))+ ".npy")


# np.save("data/valid_fm_data", Y)

invalid_center_freqs = [600e5, 610e5, 615e5, 618e5, 619e5, 621e5, 623e5, 700e5, 800e5, 810e5, 1100e5, 1101e5, 1102e5, 1103e5, 1104e5, 1107e5, 1117e5, 1118e5, 1119e5, 1120e5, 1150e5, 1203e5, 1205e5, 1215e5, 1216e5, 1217e5, 1218e5, 1311e5, 1312e5, 1313e5, 1314e5, 1318e5, 1334e5]

Y_i = np.zeros((len(valid_center_freqs), interval_length))


for i, center_freq in enumerate(invalid_center_freqs):
	Y_i[i] = np.load("pspect_invalid_fm_data/pspect_invalid_" + str(int(center_freq))+ ".npy")

# np.save("data/invalid_fm_data", Y_i)

np.save("fm_data", (Y, Y_i))