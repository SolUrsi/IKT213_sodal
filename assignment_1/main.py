import cv2 as cv

img = cv.imread('lena-1.png')
assert img is not None, "file could not be read"

print("Image height is", img.shape[1], "pixels")
print("Image height is", img.shape[0], "pixels")
print("The image has", img.shape[2], "channels")
print("The size of the total image is:", img.size, "pixels, for single channel:", img.size / 3)
print("The image is of type:", img.dtype)

cam = cv.VideoCapture(0)

fps = int(cam.get(cv.CAP_PROP_FPS))
width = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))

file_name = "camera_outputs.txt"

write_fps = "FPS: ", fps
write_width = "Width: ", width
write_height = "Height: ", height

