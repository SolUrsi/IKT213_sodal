import numpy as np
import cv2 as cv

img = cv.imread('lena.png')
assert img is not None, "file could not be read"

# II. Create functions for padding, cropping, resizing, copying, grayscale, hue-shift, HSV, smoothing, rotation
