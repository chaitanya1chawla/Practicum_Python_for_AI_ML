import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# Read text input
file_name = "aspirin.xyz"
db = pd.read_csv(file_name, delimiter='      ', header=None, skiprows=[0, 1])
db.columns = ['species', 'x', 'y', 'z']
# print(db.head())
# print(db.head())

species = db.species.to_numpy()
spec_to_size = {'O': 16, 'C': 12, 'H': 2}
spec_to_col = {'O': 'r', 'C': 'k', 'H': 'b'}
s = []
c = []
for spec in species:
    s.append(spec_to_size[spec])
    c.append(spec_to_col[spec])

matrix = np.asarray([db.x, db.y, db.z]).T

dm = distance_matrix(matrix, matrix)
dm += 10 * np.eye(len(dm))

bonds = np.argwhere(dm < 1.6)

conns = []
for b in bonds:
    conns.append([matrix[b[0]], matrix[b[1]]])
conns = np.asarray(conns)

conn_lines = Line3DCollection(conns, edgecolor='gray', linestyle='solid', linewidth=8)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.add_collection3d(conn_lines)
ax.scatter(db.x, db.y, db.z, s=s, c=c)
# plt.show()

figname = 'plot.pdf'
plt.savefig(figname)
