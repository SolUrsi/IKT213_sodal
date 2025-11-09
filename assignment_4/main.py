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

def harris (reference_image_color):
   
    gray_for_harris = cv.cvtColor(reference_image_color, cv.COLOR_BGR2GRAY)
    
    gray_float = np.float32(gray_for_harris)
    img_harris = cv.cornerHarris(gray_float, 2, 3, 0.04)
    
    # Dilation for more visible corners
    img_harris = cv.dilate(img_harris, None)
    
    # Marks corners on the original COLOR image (B, G, R)
    # Marks pixels above a threshold in Red [0, 0, 255]
    reference_image_color[img_harris > 0.01 * img_harris.max()] = [0, 0, 255]
    
    return reference_image_color

def sift (image_to_align, reference_image, max_features, good_match_precent):
    sift = cv.SIFT_create()

    # Grayscale global images for detection and computation
    kp1, des1 = sift.detectAndCompute(gray2, None)
    kp2, des2 = sift.detectAndCompute(gray, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    # Find matches
    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < good_match_precent * n.distance:
            good.append(m)

    if len(good) > max_features:
        # Extract location of good matches
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # Finding the Homography matrix (M)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        # The output aligned image should have the same size as the reference image
        height, width, channels = reference_image.shape
        aligned_img = cv.warpPerspective(image_to_align, M, (width, height)) 
        
        # Defining the bounding box on the image to align and project it onto the reference image
        h_orig, w_orig, c_orig = image_to_align.shape
        pts = np.float32([[0, 0], [0, h_orig - 1], [w_orig - 1, h_orig - 1], [w_orig - 1, 0]]).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, M)

        # Drawing the projected bounding box on the reference image (using BGR color format)
        reference_image = cv.polylines(reference_image, [np.int32(dst)], True, (0, 0, 255), 3, cv.LINE_AA)

        # Drawing the feature matches
        draw_params = dict(matchColor=(0, 255, 0),  # Green matches
                           singlePointColor=None,
                           matchesMask=matchesMask,
                           flags=2) # Draw only good matches

        img_matches = cv.drawMatches(image_to_align, kp1, reference_image, kp2, good, None, **draw_params)
        
        return img_matches, aligned_img # Return required images

    else:
        print("Not enough matches are found - {}/{}".format(len(good), max_features))
        return None, None


def main():
   if not os.path.exists("results"):
      os.makedirs("results")

   harris_img = harris(image.copy())
   matches_img, aligned_img = sift(image2.copy(), image.copy(), 10, 0.7)

   cv.imwrite("results/harris_img.jpg", harris_img)
   print("Generated results/harris_img.jpg (Page 1)")

   if matches_img is not None and aligned_img is not None:
      cv.imwrite("results/matches_img.jpg", matches_img)
      print("Generated results/matches_img.jpg (Page 2)")
        
      cv.imwrite("results/aligned_img.jpg", aligned_img)
      print("Generated results/aligned_img.jpg (Page 3)")
   else:
      print("SIFT matching and alignment failed.")


if __name__ == '__main__':
    main()
