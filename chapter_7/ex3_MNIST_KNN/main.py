from array import array
import struct
from sklearn.neighbors import KNeighborsClassifier
import os
from joblib import dump, load
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def load_mnist(folder='__files', train = True):

    if train == True:
        address = "train"
    else:
        address = "t10k"

    with open(os.path.join(folder,f"{address}-labels.idx1-ubyte"), 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049,'
                                 'got {}'.format(magic))
            labels = array("B", file.read())

    with open(os.path.join(folder,f"{address}-images.idx3-ubyte"), 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            image_no = size/(rows * cols)
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051,'
                                 'got {}'.format(magic))
            image_data = array("B", file.read())
            

    images = []
    for i in range(size):
        images.append([0] * rows * cols)
    for i in range(size):
        images[i][:] = image_data[i * rows * cols:(i + 1) * rows * cols]
        
    return images, labels

def plotting(images):
    arr = np.array(images[2], dtype=np.uint8).reshape(28,28)
    img1 = Image.fromarray(arr)
    arr = np.array(images[3], dtype=np.uint8).reshape(28,28)
    img2 = Image.fromarray(arr)
    arr = np.array(images[4], dtype=np.uint8).reshape(28,28)
    img3 = Image.fromarray(arr)
    arr = np.array(images[5], dtype=np.uint8).reshape(28,28)
    img4 = Image.fromarray(arr)
    dst = Image.new("L",size=(img1.width*2+5, img1.height*2+5))
    dst.paste(img1, (0,0))
    dst.paste(img2, (img1.width+5,0))
    dst.paste(img3, (0,img1.height+5))
    dst.paste(img4, (img1.width+5,img1.height+5))
    dst.save('numbers.pdf',bbox_inches='tight')


max = 0
test_accuracy=[]

#plotting 4 images
neigh = KNeighborsClassifier(n_neighbors = 3)
x,y = load_mnist("__files", True)
plotting(x)


for neighbour in [1,2,3,4,5,7,10,15,20]:

    neigh = KNeighborsClassifier(n_neighbors = neighbour)
    x,y = load_mnist("__files", True)
    neigh.fit(x,y)
    print("GOOD")
    w,z = load_mnist("__files", False)
    z_updated = np.frombuffer(z, dtype=np.uint8)
    pred = neigh.score(w,z_updated)
    
    
    if(pred > max):
        max = pred
        dump(neigh, 'model.sk') 
    test_accuracy.append(pred)


fig = plt.figure()
axes = fig.add_subplot()

axes.plot([1,2,3,4,5,7,10,15,20], test_accuracy, color='b')
axes.set_xticks([1,2,3,4,5,7,10,15,20])

axes.set_yticks([0.0,0.2,0.4,0.6,0.8,1.0])
plt.xlabel("number of NN")
plt.ylabel("test accuracy")
plt.savefig("plot.pdf", dpi=300, bbox_inches='tight')
