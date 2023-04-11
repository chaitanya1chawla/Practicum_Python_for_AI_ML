from sklearn.datasets import fetch_lfw_people
from PIL import ImageFile, ImageFilter, ImageEnhance
import numpy as np
import os
import matplotlib.pyplot as plt
import config
from fetch_own_images import add_imgs


def get_ml_data(min_faces=20, resize=0.4, add_image=True):
    lfw_people = fetch_lfw_people(min_faces_per_person=min_faces, resize=resize)
    # introspect the images arrays to find the shapes (for plotting)
    n_samples, h, w = lfw_people.images.shape
    print("height = ", h)
    print("width = ", w)

    X = lfw_people.data  # data as numpy arrays
    n_features = X.shape[1]
    # features = h * w , ie number of pixels for each picture

    # the label to predict is the Name of the person
    y = lfw_people.target
    target_names = lfw_people.target_names
    n_classes = target_names.shape[0]

    # you can use "imgs" instead of your img_data
    imgs = lfw_people.images
    print()
    ax = plt.subplot(1, 1, 1)
    plt.imshow(imgs[0], cmap=plt.cm.gray)
    plt.show()

    print("Total dataset size:")
    print("n_samples: %d" % n_samples)
    print("n_features: %d" % n_features)

    # add own images
    if add_image:
        assert config.im_height == h, "Height does not match with lfw dataset"
        assert config.im_width == w, "Width does not match with lfw dataset"
        imgs_new, y_new = add_imgs(config, max(y) + 1)

        # transform imgs to np array
        imgs_new = np.asarray(imgs_new)

        # color transform to b&w
        def rgb_to_gray(rgb):
            return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

        plt.imshow(imgs_new[0])
        plt.show()
        imgs_new = rgb_to_gray(imgs_new)
        plt.imshow(imgs_new[0], cmap=plt.cm.gray)
        plt.show()

        # flatten array
        imgs_new = imgs_new.reshape(len(imgs_new), -1)
        # add images and labels to dataset
        X = np.vstack((X, imgs_new))
        y = np.hstack((y, y_new))

        print("Total dataset size with own images:")
        print(f"n_samples: {len(X)}")
        print(f"n_features: {X.shape[1]}")

    ml_input = [X, h, w]
    ml_output = [y]
    return ml_input, ml_output


if __name__ == '__main__':
    get_ml_data()
