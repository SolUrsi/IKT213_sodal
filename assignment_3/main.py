import os
import cv2 as cv
import numpy as np

OUTPUT_DIR = 'images'
os.makedirs(OUTPUT_DIR, exist_ok=True)

img = cv.imread('lambo.png')
assert img is not None, "File could not be read"

img_init_template = cv.imread('shapes-1.png')
assert img_init_template is not None, "File could not be read"

img_template = cv.imread('shapes_template.jpg',0)
assert img_template is not None, "File could not be read"

# II. Create functions for edge detection, resampling, and template matching

# Sobel edge Detection
def sobel_edge_detection(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray,(3,3),0)
    sobel = cv.Sobel(src=blur, ddepth=cv.CV_64F, dx=1, dy=1, ksize=1)
    cv.imwrite(os.path.join(OUTPUT_DIR, 'sobel.png'), sobel)
    return sobel

# Canny edge detection
def canny_edge_detection(image, threshold1, threshold2):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (3, 3), 0)
    canny = cv.Canny(image=blur, threshold1=threshold1, threshold2=threshold2)
    cv.imwrite(os.path.join(OUTPUT_DIR, 'canny.png'), canny)
    return canny


# Template matching
def template_match(image, template):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    h, w = template.shape[::]
    match = cv.matchTemplate(gray, template, cv.TM_CCOEFF_NORMED)
    threshold = 0.9
    location = np.where( match >= threshold)
    for pt in zip(*location[::-1]):
        cv.rectangle(image, pt, (pt[0] + h, pt[1] + w), (0,0,255),2)
    cv.imwrite(os.path.join(OUTPUT_DIR, 'template_matching.png'), image)
    return image

# Resizing
def resize(image, scale_factor: int, up_or_down: str):
    rows, columns, _channels = map(int, image.shape)
    if up_or_down == "up":
        resized = cv.pyrUp(image, dstsize=(scale_factor * columns, scale_factor * rows))
        cv.imwrite(os.path.join(OUTPUT_DIR, 'resize.png'), resized)
        return resized
    elif up_or_down == "down":
        resized = cv.pyrDown(image, dstsize=(scale_factor // columns, scale_factor // rows))
        cv.imwrite(os.path.join(OUTPUT_DIR, 'resize.png'), resized)
        return resized
    return None


# Main
def main():
    sobel = sobel_edge_detection(img)
    canny = canny_edge_detection(img, 50, 50)
    template_matching = template_match(img_init_template, img_template)
    resized = resize(img, 2, "up")
if __name__ == "__main__":
    main()
