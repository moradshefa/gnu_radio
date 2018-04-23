from utils import *


# wifi data
# sampling_rate = 20e6, 3 seconds

# fm and unclassified
# sampling_rate = 2e6, 5 seconds


interval = 2048
path = 'wifi_data/'


center_freq = int(2.394e9)
sampling_rate = 20e6


# open file
filename = str(center_freq)
y = open_file(path + filename, data_from_hackrf = True)


f = get_freq_axis(fs = sampling_rate, fc = center_freq, interval = interval)
pspect = average_pspect(y, interval = interval)

# plot
p = plot_given_pspect(f, pspect)
plt.title(filename)
plt.show()