import numpy as np
from PIL import Image
from scipy import signal


def subsample(img_ar) -> np.array:
    img_ar = img_ar[::2, ::2]
    # for x in range(image_ar.shape[0]):
    #     for y in range(image_ar.shape[1]):
    return img_ar


def boxblur(img_ar, size=5) -> np.array:
    blur_ar = np.ones((size, size)) / (size*size)
    for c in range(img_ar.shape[-1]):
        img_ar[:, :, c] = signal.convolve2d(img_ar[:, :, c], blur_ar, mode='same')
    return img_ar


def frame(img_ar, frame, color) -> np.array:
    left = frame[0]
    right = frame[1]
    top = frame[2]
    bottom = frame[3]
    color_ar = np.asarray(color)

    img_ar[:left] = color_ar
    img_ar[-right:] = color_ar

    img_ar[:, :top] = color_ar
    img_ar[:, -bottom:] = color_ar
    return img_ar


def save_img(img, file_name):
    img_new = Image.fromarray(img)
    img_new.save(file_name)


if __name__ == '__main__':
    file_name = "TUM_old.jpg"
    img = Image.open(file_name)
    # img.show()
    image_ar = np.asarray(img)
    image_new = subsample(np.copy(image_ar))
    # save picture
    file_name = 'TUM_small.png'
    save_img(image_new, file_name)

    img_blur = boxblur(np.copy(image_ar))
    # save blur
    file_name = 'TUM_blur.png'
    save_img(img_blur, file_name)

    color = (48, 112, 179)
    img_frame = frame(np.copy(image_ar), frame=(15, 15, 15, 15), color=color)
    # save frame
    file_name = 'TUM_frame.png'
    save_img(img_frame, file_name)
