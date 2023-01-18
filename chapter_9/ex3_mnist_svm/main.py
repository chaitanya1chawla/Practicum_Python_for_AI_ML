from array import array
import struct
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import os
from joblib import dump, load
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import gzip

def load_mnist(folder='__files', train = True):

    if train == True:
        address = "train"
        num_images = "60000"
    else:
        address = "t10k"
        num_images = "10000"

    with gzip.open(os.path.join(folder,f"{address}-labels-idx1-ubyte.gz"), 'rb') as f:
        magic,size = struct.unpack(">II", f.read(8))
        if magic!=2049:
            raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
        labels = array("B", f.read())
    
    with gzip.open(os.path.join(folder,f"{address}-images-idx3-ubyte.gz"), 'rb') as f:
        magic,size, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic!=2051:
            raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
        image_data = array("B", f.read())
    images = []
    for i in range(size):
        images.append([0]*rows*cols)

    for i in range(size):
        images[i][:] = image_data[i*rows*cols:(i+1)*rows*cols]

    return images, labels

def plotting(images):
    img =[]

    arr = np.array(images[2], dtype=np.uint8).reshape(28,28)
    img.append(Image.fromarray(arr))
    arr = np.array(images[3], dtype=np.uint8).reshape(28,28)
    img.append(Image.fromarray(arr))
    arr = np.array(images[4], dtype=np.uint8).reshape(28,28)
    img.append(Image.fromarray(arr))
    arr = np.array(images[5], dtype=np.uint8).reshape(28,28)
    img.append(Image.fromarray(arr))
    
    dst = Image.new("L",size=(img[0].width*2+5, img[0].height*2+5))
    dst.paste(img[0], (0,0))
    dst.paste(img[1], (img[0].width+5,0))
    dst.paste(img[2], (0,img[0].height+5))
    dst.paste(img[3], (img[0].width+5,img[0].height+5))
    dst.save('plot.pdf',bbox_inches='tight')

if __name__ == "__main__":
    train_data = load_mnist()
    train_imgs = train_data[0]
    train_lbls = train_data[1]

    test_data = load_mnist(train = False)
    test_imgs = test_data[0]
    test_lbls = test_data[1]

    # hypermodel = GridSearchCV(SVC(kernel ="rbf"), {"C" : np.arange(0,1,0.5)}, cv=5, n_jobs=-1, scoring="accuracy")
    # svm_model = hypermodel.fit(trainings, trainlbls)
    svm_model = SVC(kernel = 'rbf', C=1.0)
    svm_model.fit(train_imgs, train_lbls)
    dump(svm_model, "model.joblib")

    print("C: ", svm_model.C)
    pred = svm_model.predict(test_imgs)
    acc = accuracy_score(test_lbls, pred)
    print(acc)

    fig = plt.figure()
    for ix in range(1,21):
        ax = fig.add_subplot(4,5,ix)
        ax.imshow(np.array(test_imgs[28*ix]).reshape(28,28))
        ax.set_title("pred: {} true: {}".format(pred[28*ix], test_lbls[28*ix]) )
    plt.tight_layout()
    fig.savefig("plot.pdf")