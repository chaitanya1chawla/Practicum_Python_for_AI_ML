from time import time
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('_mpl-gallery')

import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.utils.fixes import loguniform


print("started")
lfw_people = fetch_lfw_people(min_faces_per_person=20, resize=0.4)
# only 2 people have more than 70 pictures, George W Bush and Gerhard Schroeder

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape
print(h)
print(w)


X = lfw_people.data
n_features = X.shape[1]
# features = h * w , ie number of pixels for each picture

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
#print(target_names)
#count =0
#for i in y:
#    if (i==0):
#        count+=1
#print(count)
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)


## train test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

print(X_train[0])


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pca_model = PCA(n_components=150, svd_solver="randomized", whiten=True)
pca = pca_model.fit(X_test)
eigenfaces = pca.components_.reshape((150, h, w))


score=[]
n_comps = range(10, 400, 10)
for n_components in n_comps:

    pca_model = PCA(n_components=n_components, svd_solver="randomized", whiten=True)
    pca = pca_model.fit(X_train)

    #can be used for report to show how an eigenface or principle component looks like
    print(pca.components_.shape)
    
    eigenfaces = pca.components_.reshape((n_components, h, w))

    #print("Projecting the input data on the eigenfaces orthonormal basis")
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    print("X_train_pca = ",X_train_pca.shape)
    print("X_train = ",X_train.shape)


    # print("Fitting the classifier to the training set")
    param = {"C": loguniform(1e3, 1e5),"gamma": loguniform(1e-4, 1e-1), }

    clf = RandomizedSearchCV( SVC(kernel="rbf", class_weight="balanced"), param, n_jobs=-1, cv=10 )
    clf = clf.fit(X_train_pca, y_train)
    # print("n_components = ", clf.score(X_test_pca, y_test))
    score.append(clf.score(X_test_pca, y_test))

xmax = n_comps[np.argmax(score)]
ymax = max(score)

ax = plt.figure().add_subplot()
ax.set_xlim(10, 390)
ax.set_ylim(0.7, 1.0)
text= "x={:.1f}, y={:.2f}".format(xmax, ymax)
bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
ax.annotate(text, xy=(xmax, ymax), xytext=(0.94,0.96), **kw)
ax.set_xlabel("No. of Principal Components")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy vs No. of Principal Components")
ax.plot(n_comps, score, linewidth=2.0)
plt.savefig("plot.pdf", dpi=300, bbox_inches='tight')


n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
  # display original
  ax = plt.subplot(2, n, i + 1)
  plt.imshow(X_test[i].reshape((h, w)), cmap=plt.cm.gray)
  plt.title("original")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  # display reconstruction
  ax = plt.subplot(2, n, i + 1 + n)
  plt.imshow(eigenfaces[i].reshape((h, w)), cmap=plt.cm.gray)
  plt.title("Eigenfaces")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
plt.show()