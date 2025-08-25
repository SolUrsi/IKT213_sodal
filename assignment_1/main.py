import cv2 as cv
import os

img = cv.imread('lena-1.png')
assert img is not None, "file could not be read"

print("Image height is", img.shape[1], "pixels")
print("Image width is", img.shape[0], "pixels")
print("The image has", img.shape[2], "channels")
print("The size of the total image is:", img.size, "pixels, for single channel:", img.size / 3)
print("The image is of type:", img.dtype)

cam = cv.VideoCapture(0)

fps = int(cam.get(cv.CAP_PROP_FPS))
width = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))

file_name = "camera_outputs.txt"
directory = "solutions"
os.makedirs(directory, exist_ok=True)
file_path = os.path.join(directory, "camera_outputs.txt")

with open(file_path, "w") as f:
    f.write(f"FPS: {fps}\n")
    f.write(f"Width: {width}\n")
    f.write(f"Height: {height}\n")



