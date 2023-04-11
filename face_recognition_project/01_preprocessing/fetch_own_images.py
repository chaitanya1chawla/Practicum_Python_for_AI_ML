import numpy as np
import os
from PIL import Image, ImageFilter, ImageEnhance
import random

import config


def add_imgs(config, y_start) -> (list, list):
    img_data, ml_output = import_images(config.image_dir, y_start)

    # # plot 1 image as example
    # show_image(img_data[0])

    ml_input = img_data

    extra_input = []
    extra_output = []

    # add more images through data augmentation
    for m in config.augment_modes:
        if m == 'rotate':
            # rotate by 180 degree to keep dimensions
            extra_input = rotate_images(extra_input, img_data)
            extra_output.extend(ml_output)

        if m == 'flip':
            # mirror the image
            extra_input = flip_images(extra_input, img_data)
            extra_output.extend(ml_output)

        if m == 'noise':
            # add gaussian noise to the image
            extra_input = noise_image(extra_input, img_data)
            extra_output.extend(ml_output)

        if m == 'contrast':
            # increase contrast of image
            extra_input = contrast_image(extra_input, img_data)
            extra_output.extend(ml_output)

    # add augmented data to ml data
    ml_input.extend(extra_input)
    ml_output.extend(extra_output)

    if config.split_data:
        ml_input, ml_output = split_train(ml_input, ml_output)

    return ml_input, ml_output

# Augmentation code
###################


def rotate_images(extra_input, img_data):
    for im_ar in img_data:
        im = ar_to_image(im_ar)
        im.rotate(180)
        # save new image
        extra_input.append(np.asarray(im))
    return extra_input


def flip_images(extra_input, img_data):
    for im_ar in img_data:
        im = ar_to_image(im_ar)
        im.transpose(method=Image.FLIP_LEFT_RIGHT)
        # save new image
        extra_input.append(np.asarray(im))
    return extra_input


def noise_image(extra_input, img_data):
    for im_ar in img_data:
        im = ar_to_image(im_ar)
        im = im.filter(ImageFilter.GaussianBlur(radius=10))
        # save new image
        extra_input.append(np.asarray(im))
    return extra_input


def contrast_image(extra_input, img_data):
    for im_ar in img_data:
        im = ar_to_image(im_ar)
        # Create an enhancer for the image
        enhancer = ImageEnhance.Contrast(im)
        im = enhancer.enhance(1.5)
        # save new image
        extra_input.append(np.asarray(im))
    return extra_input

# Image processing
##################


def import_images(image_dir: str, y_start=0) -> list:
    # import jpg images and save as list
    # output image and label

    cl_dict = config.class_dict
    files = os.listdir(image_dir)
    image_list = []
    label_list = []
    for f in files:
        if f.endswith('.jpg'):
            # import image
            im = Image.open(f"{image_dir}{f}")
            # resize image
            im = resize_image(im)
            # save image as array
            image_list.append(np.asarray(im))

            # get label
            label = f[:-4]  # remove format header
            label = helper_string_without_num(label)
            label_key = cl_dict[label]
            label_key += y_start
            label_list.append(label_key)

        else:
            print(f"Ignore file <{f}> due to format.")
    return image_list, label_list


def resize_image(im: Image) -> Image:
    # The requested size in pixels, as a 2-tuple: (width, height).
    im = im.resize(size=(config.im_width, config.im_height))
    return im


def ar_to_image(im_ar: np.array) -> Image:
    im = Image.fromarray(im_ar)
    return im


def show_image(im_ar: np.array) -> None:
    # Plot image from array
    im = ar_to_image(im_ar)
    im.show()

# Helper functions
##################


def helper_string_without_num(string: str) -> str:
    # remove numbers from string
    result = ''.join(i for i in string if not i.isdigit())
    return result


def split_train(ml_in, ml_out, p_val=0.15, p_test=0.15):
    # split dataset into train, val, test
    # reshape data as np arrays

    # calculate bounds
    n_samples = len(ml_in)
    n_val = int(n_samples * p_val)
    n_test = int(n_samples * p_test)
    n_train = n_samples - n_val - n_test

    # shuffle dataset
    indices = list(range(n_samples))
    random.shuffle(indices)
    ml_in = [ml_in[i] for i in indices]
    ml_out = [ml_out[i] for i in indices]

    ml_in = np.asarray(ml_in)
    ml_out = np.asarray(ml_out)

    # split dataset
    ml_in_split = [ml_in[:n_train], ml_in[n_train:n_train + n_val],
                   ml_in[n_train + n_val:]]
    ml_out_split = [ml_out[:n_train], ml_out[n_train:n_train + n_val],
                    ml_out[n_train + n_val:]]

    return ml_in_split, ml_out_split


if __name__ == '__main__':
    image_dir = config.image_dir
    main(image_dir, config.augment_modes)
