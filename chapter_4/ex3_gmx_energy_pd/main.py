# imports

import numpy as np
import pandas as pd
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
file_name = "zif-nvt.csv"
db = pd.read_csv(file_name, delimiter='  ', header=None)
# print(db.head())
sensor_ar = db.iloc[:, 1].to_numpy()

temp_avg = moving_average(25, sensor_ar)

# output average file
avg_ar = np.asarray(temp_avg)
# avg_ar = np.stack((system, avg_ar)).T
np.savetxt('average.csv', avg_ar, delimiter='  ', fmt='%.2f')

# plot
plt.figure()
plt.plot(sensor_ar, label="sensor")
plt.plot(temp_avg, label="filter")
plt.legend()

#title
plt.title("Average Temperature over Time (pandas)")

# save pdf
plt.savefig("plot.pdf")
