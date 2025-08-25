import numpy as np
import cv2 as cv
from cv2 import COLOR_BGR2RGB
from matplotlib import pyplot as plt

img = cv.imread('lena.png')
assert img is not None, "file could not be read"

# II. Create functions for padding, cropping, resizing, copying, grayscale, hue-shift, HSV, smoothing, rotation

# Padding
def padding(image, border_width):
    return cv.copyMakeBorder(image, border_width, border_width, border_width, border_width,
    cv.BORDER_REFLECT)


# Cropping
def crop(image, x_0, x_1, y_0, y_1):
    return image[x_0:x_1,y_0:y_1]


# Resizing
def resize(image, width, height):
    return cv.resize(image, (width, height))


# Manual Copy
def copy(image, emptyPictureArray):
    for height in range(image.shape[0]):
        for width in range(image.shape[1]):
            pixel = image[height, width]
            emptyPictureArray[height, width] = pixel
    return emptyPictureArray


# Grayscale
def grayscale(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# HSV
def hsv(image):
    return cv.cvtColor(image, cv.COLOR_BGR2HSV)


# Hue shifting
def hue_shifted(image, emptyPictureArray, hue):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for c in range(image.shape[2]):
                shifted = image[i,j,c] + hue
                emptyPictureArray[i,j,c] = shifted
    return emptyPictureArray


# Smoothing
def smoothing(image):
    return cv.GaussianBlur(image, (15,15), cv.BORDER_DEFAULT)


# Rotation
def rotation(image, rotation_angle):
    if rotation_angle == 90:
        return cv.rotate(image, cv.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        return cv.rotate(image, cv.ROTATE_180)
    return image


def main():
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB) # Remove blue tint BGR -> RGB

    pad = padding(img_rgb, 100)
    crp = crop(img_rgb, 80, img_rgb.shape[0] - 130, 80, img_rgb.shape[1]-130)
    rsz = resize(img_rgb, 200, 200)

    emptyPictureArray1 = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype=np.uint8)
    emptyPictureArray2 = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype=np.uint8)
    new_img = copy(img, emptyPictureArray1)
    gry = grayscale(img)
    hs = hsv(img)
    hue = hue_shifted(img, emptyPictureArray2, 50)
    smt = smoothing(img)
    rot90 = rotation(img, 90)
    rot180 = rotation(img, 180)

    plt.figure(figsize=(10, 6))
    plt.subplot(231), plt.imshow(img_rgb), plt.title('ORIGINAL')
    plt.subplot(232), plt.imshow(pad), plt.title('PADDING')
    plt.subplot(233), plt.imshow(crp), plt.title('CROPPING')
    plt.subplot(234), plt.imshow(rsz), plt.title('RESIZING')
    plt.subplot(235), plt.imshow(cv.cvtColor(new_img, cv.COLOR_BGR2RGB)), plt.title('COPY')
    plt.subplot(236), plt.imshow(cv.cvtColor(gry, cv.COLOR_BGR2RGB)), plt.title('GRAYSCALE')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.subplot(231), plt.imshow(cv.cvtColor(hs, cv.COLOR_BGR2RGB)), plt.title('HSV')
    plt.subplot(232), plt.imshow(cv.cvtColor(hue, cv.COLOR_BGR2RGB)), plt.title('HUE SHIFT')
    plt.subplot(233), plt.imshow(cv.cvtColor(smt, cv.COLOR_BGR2RGB)), plt.title('SMOOTHING')
    plt.subplot(234), plt.imshow(cv.cvtColor(rot90, cv.COLOR_BGR2RGB)), plt.title('90')
    plt.subplot(235), plt.imshow(cv.cvtColor(rot180, cv.COLOR_BGR2RGB)), plt.title('180')
    plt.show()


if __name__ == "__main__":
    main()
