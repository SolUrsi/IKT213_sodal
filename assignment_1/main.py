import cv2 as cv

img = cv.imread('lena-1.png')
assert img is not None, "file could not be read"

print("Image height is", img.shape[1], "pixels")
print("Image height is", img.shape[0], "pixels")
print("The image has", img.shape[2], "channels")
print("The size of the total image is:", img.size, "pixels, for single channel:", img.size / 3)
print("The image is of type:", img.dtype)



