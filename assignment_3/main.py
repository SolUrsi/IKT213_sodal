import numpy as np
import cv2 as cv

img = cv.imread('lambo.png')
assert img is not None, "file could not be read"

# II. Create functions for edge detection, resampling, and template matching

# Sobel edge Detection
def sobel_edge_detection(image):
    # Grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray,(3,3),0)
    sobel = cv.Sobel(src=blur, ddepth=cv.CV_64F, dx=1, dy=1, ksize=1)
    return sobel


# Canny edge detection
def canny_edge_detection():
    return 0

# Template matching
def template_match():
    return 0

# Resizing
def resize():
    return 0

# Main
def main():
    cv.imshow('Original', img)
    cv.waitKey(0)
    cv.imshow('Sobel', sobel_edge_detection(img))
    cv.waitKey(1)

if __name__ == "__main__":
    main()
