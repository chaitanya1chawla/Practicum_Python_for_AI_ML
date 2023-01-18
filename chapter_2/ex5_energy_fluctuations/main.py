# imports

import numpy as np
import matplotlib.pyplot as plt


# Define moving average plus first values raw
def moving_average(window_size, arr):
    i = 0

    moving_averages = []

    while i < len(arr):
        i += 1
        i_start = i - window_size
        if i_start < 0:
            i_start = 0
        window = arr[i_start: i]

        window_average = round(sum(window) / len(window), 2)
        # window_average = sum(window) / len(window)
        moving_averages.append(window_average)

    return moving_averages


# Read text input
filename = "zif-nvt.csv"
with open(filename, "r") as f:
    lines = f.readlines()

# remove whitespace
for i, l in enumerate(lines):
    lines[i] = l.strip()

system = []
temp = []

for l in lines:
    [val0, val1] = l.split()
    val0 = float(val0)
    val1 = float(val1)

    system.append(val0)
    temp.append(val1)

# system_avg = moving_average(25, system)
temp_avg = moving_average(25, temp)

# output average file
avg_ar = np.asarray(temp_avg)
avg_ar = np.stack((system, avg_ar)).T
#np.savetxt('average.csv', avg_ar, fmt='%.2f')
np.savetxt('average.csv', avg_ar, delimiter=';', fmt='%.2f')

# plot
plt.figure()
plt.plot(temp, label="sensor")
plt.plot(temp_avg, label="filter")
plt.legend()

#title
plt.title("Average Temperature over Time")

# save pdf
plt.savefig("plot.pdf")
