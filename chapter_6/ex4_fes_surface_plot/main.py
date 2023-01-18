import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.spatial import distance_matrix
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib import mlab

# Read text input
file_name = "fes.csv"
db = pd.read_csv(file_name, delimiter='\t', header=None, skiprows=[0])
db.replace([np.inf, -np.inf], np.nan, inplace=True)
# db.fillna(0, inplace=True)
db.dropna(inplace=True)
db.columns = ['x', 'y', 'energy']
print(db.head())

# fig, (ax_1, ax_2) = plt.subplots(1, 2, subplot_kw={"projection": "3d"})
fig = plt.figure()
ax_1 = plt.subplot(1, 2, 1)
ax_2 = plt.subplot(1, 2, 2, projection="3d")

ax_1.tricontour(db.x, db.y, db.energy)
ax_1.set(xlabel='CV1', ylabel='CV2')
# xi = np.linspace(4, 8, 10)
# yi = np.linspace(1, 4, 10)
# z = mlab.griddata(db.x, db.y, db.energy, xi, yi, interp='linear')

x = np.asarray(db.x).reshape(-1, 55)
y = np.asarray(db.y).reshape(-1, 55)
z = -1 * np.asarray(db.energy).reshape(-1, 55)

ax_2.contour(x, y, z, cmap=cm.viridis)
surf = ax_2.plot_surface(x, y, z, cmap=cm.viridis)
fig.colorbar(surf, shrink=0.5, pad=0.2)

# plt.show()

plt.savefig('plot.pdf')
