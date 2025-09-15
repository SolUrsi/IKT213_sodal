import os
import numpy as np
import cv2 as cv
from cv2 import COLOR_BGR2RGB, COLOR_RGB2BGR, COLOR_HSV2BGR

OUTPUT_DIR = 'images'
os.makedirs(OUTPUT_DIR, exist_ok=True)

img = cv.imread('lena.png')
assert img is not None, "File could not be read"


def padding(image, border_width):
    padded = cv.copyMakeBorder(image, border_width, border_width,
                               border_width, border_width,
                               cv.BORDER_REFLECT)
    cv.imwrite(os.path.join(OUTPUT_DIR, 'padded.png'),
               cv.cvtColor(padded, COLOR_RGB2BGR))
    return padded

def crop(image, x0, x1, y0, y1):
    crp = image[x0:x1, y0:y1]
    cv.imwrite(os.path.join(OUTPUT_DIR, 'cropped.png'),
               cv.cvtColor(crp, COLOR_RGB2BGR))
    return crp

def resize(image, width, height):
    rsz = cv.resize(image, (width, height))
    cv.imwrite(os.path.join(OUTPUT_DIR, 'resized.png'),
               cv.cvtColor(rsz, COLOR_RGB2BGR))
    return rsz

def copy_img(image):
    copy_arr = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            copy_arr[i, j] = image[i, j]
    cv.imwrite(os.path.join(OUTPUT_DIR, 'copied.png'), copy_arr)
    return copy_arr

def grayscale(image):
    gry = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imwrite(os.path.join(OUTPUT_DIR, 'grayscale.png'), gry)
    return gry

def hsv(image):
    hs = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    cv.imwrite(os.path.join(OUTPUT_DIR, 'hsv.png'),
               cv.cvtColor(hs, COLOR_HSV2BGR))
    return hs

def hue_shifted(image, hue):
    shifted = np.clip(image.astype(int) + hue, 0, 255).astype(np.uint8)
    cv.imwrite(os.path.join(OUTPUT_DIR, 'hue_shift.png'), shifted)
    return shifted

def smoothing(image):
    smt = cv.GaussianBlur(image, (15, 15), cv.BORDER_DEFAULT)
    cv.imwrite(os.path.join(OUTPUT_DIR, 'smoothed.png'), smt)
    return smt

def rotation(image, angle):
    if angle == 90:
        out = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)
        name = 'rotated_90.png'
    elif angle == 180:
        out = cv.rotate(image, cv.ROTATE_180)
        name = 'rotated_180.png'
    else:
        out = image.copy()
        name = 'rotated_0.png'
    cv.imwrite(os.path.join(OUTPUT_DIR, name), out)
    return out

def main():
    # for display we convert BGR → RGB
    img_rgb = cv.cvtColor(img, COLOR_BGR2RGB)

    pad     = padding(img_rgb, 100)
    crp     = crop(img_rgb, 80, img_rgb.shape[0] - 130, 80, img_rgb.shape[1] - 130)
    rsz     = resize(img_rgb, 200, 200)

    new_img = copy_img(img)

    gry     = grayscale(img)
    hs      = hsv(img)
    hue     = hue_shifted(img, 50)
    smt     = smoothing(img)
    rot90   = rotation(img, 90)
    rot180  = rotation(img, 180)

if __name__ == "__main__":
    main()
