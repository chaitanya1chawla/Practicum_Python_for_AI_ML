import keras
from keras import regularizers
from keras.datasets import mnist
from keras import layers, losses
from keras.datasets import fashion_mnist
from keras.models import Model

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.utils.fixes import loguniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2


# #--> This is the size of our encoded representations
# encoding_dim = 32  # 64 floats -> compression of factor 28.9, assuming the input is 1850 floats
# 
# # This is our input image
# #-->
# input_img = keras.Input(shape=(784,))
# # "encoded" is the encoded representation of the input
# encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
# # "decoded" is the lossy reconstruction of the input
# #-->
# decoded = layers.Dense(784, activation='sigmoid')(encoded)
# 
# # This model maps an input to its reconstruction
# autoencoder = keras.Model(input_img, decoded)
# 
# # This model maps an input to its encoded representation
# encoder = keras.Model(input_img, encoded)
# 
# # This is our encoded (64-dimensional) input
# encoded_input = keras.Input(shape=(encoding_dim,))
# # Retrieve the last layer of the autoencoder model
# decoder_layer = autoencoder.layers[-1]
# # Create the decoder model
# decoder = keras.Model(encoded_input, decoder_layer(encoded_input))
# 
# 
# autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
# 
# (x_train, _), (x_test, _) = mnist.load_data()
# lfw_people = fetch_lfw_people(min_faces_per_person=20, resize=0.4)
# n_samples, h, w = lfw_people.images.shape
# X = lfw_people.data
# # the label to predict is the id of the person
# y = lfw_people.target
# target_names = lfw_people.target_names
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.25, random_state=42
# )
# 
# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.
# x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
# x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
# #print(x_train.shape[1:])
# print(X_train.shape)
# #print(x_test.shape)
# print(X_test.shape)
# 
# 
# autoencoder.fit(x_train, x_train,
#                 epochs=50,
#                 batch_size=256,
#                 shuffle=True,
#                 validation_data=(x_test, x_test))
# 
# 
# # Encode and decode some digits
# # Note that we take them from the *test* set
# encoded_imgs = encoder.predict(x_test)
# decoded_imgs = decoder.predict(encoded_imgs)
# 
# 
# n = 10  # How many digits we will display
# plt.figure(figsize=(20, 4))
# for i in range(n):
#     # Display original
#     ax = plt.subplot(2, n, i + 1)
#     plt.imshow(x_test[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# 
#     # Display reconstruction
#     ax = plt.subplot(2, n, i + 1 + n)
#     plt.imshow(decoded_imgs[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()

"""
Delete following paragraph and use 01_preprocessing/fetch_data.get_ml_data()
Extract ML data:
  [X, h, w] = ml_input
  [y] = ml_output
"""
lfw_people = fetch_lfw_people(min_faces_per_person=20,resize=0.4)
n_samples, h, w = lfw_people.images.shape
print("h = ",h,", w = ",w)
X = lfw_people.data
n_features = X.shape[1]
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]
Z = np.zeros((807, 44,44))


#print("n_classes: %d" % n_classes)
print("X shape: ", X.shape)
for i in range(807):
        Z[i] = cv2.resize(X[i].reshape((h,w)), dsize=(44, 44), interpolation=cv2.INTER_CUBIC)
print("X reshaped: ", Z.shape)
X_train, X_test, y_train, y_test = train_test_split(
    Z, y, test_size=0.25, random_state=42)


#(x_train, _), (x_test, _) = fashion_mnist.load_data()

X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

#print (x_train.shape)
#print (x_test.shape)


latent_dim = 64

class Autoencoder(Model):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential([
      #layers.Flatten(),
      #layers.Dense(latent_dim, activation='relu'),

      # layers.Input(shape=(44,44,1)),
      # layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
      # layers.MaxPooling2D((2, 2), padding='same'),
      # layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
      # layers.MaxPooling2D((2, 2), padding='same'),
      # layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
      # layers.MaxPooling2D((2, 2), padding='same')

      layers.Input(shape=(44,44,1)),
      layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
      layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)
    ])
    self.decoder = tf.keras.Sequential([
      # layers.Dense(784, activation='sigmoid'),
      # layers.Reshape((28, 28))

      # layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
      # layers.UpSampling2D((2, 2)),
      # layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
      # layers.UpSampling2D((2, 2)),
      # layers.Conv2D(16, (3, 3), activation='relu'),
      # layers.UpSampling2D((2, 2)),
      # layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')

      layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = Autoencoder(latent_dim)

autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

autoencoder.fit(X_train, X_train,
                # change no. of epochs
                epochs=25, #batchsize = 8,
                shuffle=True,
                validation_data=(X_test, X_test))


encoded_test_imgs = autoencoder.encoder(X_test).numpy()
decoded_test_imgs = autoencoder.decoder(encoded_test_imgs).numpy()

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
  # display original
  ax = plt.subplot(2, n, i + 1)
  plt.imshow(X_test[i], cmap=plt.cm.gray)
  plt.title("original")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  # display reconstruction
  ax = plt.subplot(2, n, i + 1 + n)
  plt.imshow(decoded_test_imgs[i], cmap=plt.cm.gray)
  plt.title("reconstructed")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
plt.show()


encoded_train_imgs = autoencoder.encoder(X_train).numpy()
decoded_train_imgs = autoencoder.decoder(encoded_train_imgs).numpy()
print("decoded_train_imgs shape = ",decoded_train_imgs.shape)
print("decoded_train_imgs reshaped = ",decoded_train_imgs.reshape(605,1936).shape)


score = []
# print("Fitting the classifier to the training set")
param = {"C": loguniform(1e3, 1e5),"gamma": loguniform(1e-4, 1e-1), }

clf = RandomizedSearchCV( SVC(kernel="rbf", class_weight="balanced"), param, n_jobs=-1, cv=10 )
clf = clf.fit(decoded_train_imgs.reshape(605,1936), y_train)
print("score = ", clf.score(decoded_test_imgs.reshape(202,1936), y_test))
# score.append(clf.score(decoded_test_imgs, y_test))


#ax = plt.figure().add_subplot()
#ax.set_xlim(10, 390)
#ax.set_ylim(0.7, 1.0)
#ax.plot(n_comps, score, linewidth=2.0)
#plt.savefig("plot.pdf", dpi=300, bbox_inches='tight')