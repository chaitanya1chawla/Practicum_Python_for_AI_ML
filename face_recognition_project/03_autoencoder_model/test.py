from time import time
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('_mpl-gallery')
import scipy.misc

import numpy as np
import cv2
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

print("started")
lfw_people = fetch_lfw_people(min_faces_per_person=20, resize = 0.6,)
# only 2 people have more than 70 pictures, George W Bush and Gerhard Schroeder

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape
print("height = ", h)
print("width = ", w)


X = lfw_people.data
n_features = X.shape[1]
# features = h * w , ie number of pixels for each picture

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names

n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
Z = np.zeros((807,66,66))
#print("n_classes: %d" % n_classes)
print("X[0] shape: ", X[0].reshape((h,w)).shape)
for i in range(807):
    Z[i] = cv2.resize(X[i].reshape((h,w)), dsize=(66, 66), interpolation=cv2.INTER_CUBIC)
print("X reshaped: ", Z.shape)

ax = plt.subplot(2,1,1)
plt.imshow(X[0].reshape((h, w)), cmap=plt.cm.gray)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

ax = plt.subplot(2,1,2)
plt.imshow(Z[0], cmap=plt.cm.gray)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

plt.show()

#image0 = scipy.misc.imresize(Z[0], (i_height, i_width))
#res = cv2.resize(Z[0], dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
#print("Z reshaped: ", res.shape)


(x_train, _), (x_test, _) = fashion_mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

print (x_train.shape)
print (x_test.shape)