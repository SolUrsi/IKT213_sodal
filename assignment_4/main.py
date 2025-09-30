import cv2 as cv
import numpy as np
import os


image = cv.imread("reference_img.png")
if image is None:
    print("Error: Could not load image 'reference_img.png'")

image2 = cv.imread("align_this.jpg")
if image2 is None:
    print("Error: Could not load image 'align_this.jpg'")

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)

def harris (reference_image):
    gray_float = np.float32(reference_image)
    img = cv.cornerHarris(gray_float, 2, 3, 0.04)
    # Dilation for more visible corners
    img = cv.dilate(img, None)
    normalized_img = cv.normalize(img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    final_img = np.uint8(normalized_img)
    return final_img


def sift (image_to_align, reference_image, max_features, good_match_precent):
    sift = cv.SIFT_create()

    kp1, des1 = sift.detectAndCompute(image_to_align, None)
    kp2, des2 = sift.detectAndCompute(reference_image, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < good_match_precent * n.distance:
            good.append(m)

    if len(good) > max_features:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = image_to_align.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, M)

        reference_image = cv.polylines(reference_image, [np.int32(dst)], True, 255, 3, cv.LINE_AA)

    else:
        print("Not enough matches are found - {}/{}".format(len(good), max_features))
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=None,
                       matchesMask=matchesMask,
                       flags=2)

    img = cv.drawMatches(image_to_align,kp1,reference_image,kp2,good,None,**draw_params)
    return img


def main():
   harris_img = harris(gray)
   sift_img = sift(gray2, gray, 10, 0.7)

   cv.imwrite("results/harris_img.jpg", harris_img)
   cv.imwrite("results/sift_img.jpg", sift_img)


if __name__ == '__main__':
    main()