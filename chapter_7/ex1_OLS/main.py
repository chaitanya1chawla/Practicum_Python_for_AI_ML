from math import sin
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.linalg import inv
from collections import OrderedDict

def ols_fit(x,y,transformations: list):
    coeff=[]
    print(x)
    x_traf=np.zeros((len(transformations),len(x)))

    for idx,i in enumerate(transformations):
        
        x_tr=[]
        for xx in x:
            x_tr.append(i(xx))
        x_traf[idx,:] = x_tr        
    
    
    #coeff = inv(np.dot(x_traf, x_traf.transpose())) 
    coeff = np.dot( inv(np.dot(x_traf, x_traf.transpose())) , np.dot(x_traf,y) )

    return coeff


data = pd.read_csv("__files/data.csv")
x = data['x'].tolist()

y = data['y'].tolist()

res = {x[i]: y[i] for i in range(len(x))}
res = OrderedDict(sorted(res.items()))
x = list(res.keys())
y = list(res.values())

coeffs = ols_fit(x, y, [lambda x: x*x, lambda x: x**5, lambda x: sin(x)])

str = (f"{coeffs[0]} {coeffs[1]} {coeffs[2]}")
f = open("coeffs.txt", "w")
n = f.write(str)
f.close()


fig = plt.figure()
axes = fig.add_subplot()

plt.title(f"OLS fit, a={coeffs[0]}, b={coeffs[1]}, c={coeffs[2]},")

plotrange= np.linspace(-10,10,1000)
axes.plot(plotrange, list(coeffs[0]*np.square(plotrange) + coeffs[1]*np.power(plotrange,5) + coeffs[2]*np.sin(plotrange)) , color='r', label="OLS fit")
axes.scatter(x,y, color='b',label="Datapoints")

plt.legend(loc = "upper left")
plt.savefig("plot.pdf", dpi=300, bbox_inches='tight')
