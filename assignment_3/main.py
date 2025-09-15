import os
import cv2 as cv

OUTPUT_DIR = 'images'
os.makedirs(OUTPUT_DIR, exist_ok=True)

img = cv.imread('lambo.png')
assert img is not None, "File could not be read"

# II. Create functions for edge detection, resampling, and template matching

# Sobel edge Detection
def sobel_edge_detection(image):
    # Grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray,(3,3),0)
    sobel = cv.Sobel(src=blur, ddepth=cv.CV_64F, dx=1, dy=1, ksize=1)
    cv.imwrite(os.path.join(OUTPUT_DIR, 'sobel.png'), sobel)
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
    sobel = sobel_edge_detection(img)
    cv.imshow('Sobel', sobel)
    cv.waitKey(0)

if __name__ == "__main__":
    main()
