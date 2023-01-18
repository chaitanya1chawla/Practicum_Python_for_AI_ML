import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import math
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

im = np.zeros([450,800,3],dtype=np.uint8)

#im = Image.open('__files/example.png', 'r')
print(type(im))

width= 800
height = 450
colors=[]
xx=0
for i in range(255):
    colors.append((i,0,0))
    xx+=1

# x == 0 to width     
# y == 0 to height    

for x in range(height):
    for y in range(width):
        #print(y)
        ix = (3.5*x/height) - 2.5
        iy = (2*y/width) - 1      #scaled y coordinate of pixel (scaled to lie in the Mandelbrot Y scale (-1, 1))

        zx = 0  #zx represents the real part of z
        zy = 0  #zy represents the imaginary part of z 

        iteration = 0
        max_iteration = 100
    
        while (zx*zx + zy*zy < 4 and iteration < max_iteration) :
            xtemp = zx*zx - zy*zy + ix 
            zy = abs(2*zx*zy) + iy   # abs returns the absolute value
            zx = xtemp
            iteration = iteration + 1

        if iteration == max_iteration: # Belongs to the set
            im[x,y,:] = (255,255,255)
            continue
        
        #val = colors[iteration].lstrip('#')
        #lv = len(val)
        im[x,y, :] = colors[iteration] #colors[iteration]

img = Image.fromarray(im, 'RGB')
print(img.size)
img.save('fractal.png',bbox_inches='tight')
#plt.savefig("fractal.png", dpi=300, bbox_inches='tight')
