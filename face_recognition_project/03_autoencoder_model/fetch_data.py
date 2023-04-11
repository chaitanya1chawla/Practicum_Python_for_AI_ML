from sklearn.datasets import fetch_lfw_people
from PIL import ImageFile, ImageFilter, ImageEnhance
import numpy as np
import os
import matplotlib.pyplot as plt



lfw_people = fetch_lfw_people(min_faces_per_person=20, resize = 0.4)
# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape
print("height = ", h)
print("width = ", w)

X = lfw_people.data # data as numpy arrays
n_features = X.shape[1]
# features = h * w , ie number of pixels for each picture

# the label to predict is the Name of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

# you can use "imgs" instead of your img_data
imgs = lfw_people.images
print()
ax = plt.subplot(1,1,1)
plt.imshow(imgs[0], cmap=plt.cm.gray)
plt.show()


print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)