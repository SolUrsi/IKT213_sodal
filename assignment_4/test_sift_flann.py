import cv2 as cv
import os
import time

from test_orb_bf import results_folder


# Taken straight from example (Experiment 2): https://opencv.org/blog/fingerprint-matching-using-opencv/

def preprocess(path):
    img = cv.imread(path, 0)
    _, img_bin = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    return img_bin

def match(img1_path, img2_path):
    img1 = preprocess(img1_path)
    img2 = preprocess(img2_path)

    sift = cv.SIFT_create(nfeatures=1000)

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        return 0, None

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

    match_img = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return len(good_matches), match_img

def process_set(set_path, results_folder):
    threshold = 20
    y_true = []
    y_pred = []

    os.makedirs(results_folder, exist_ok=True)
    for folder in sorted(os.listdir(set_path)):
        folder_path = os.path.join(set_path, folder)
        if os.path.isdir(folder_path):  # Check if it's a valid directory
            image_files = [f for f in os.listdir(folder_path) if f.endswith(('.tif', '.png', '.jpg'))]
            if len(image_files) != 2:
                print(f"Skipping {folder}, expected 2 images but found {len(image_files)}")
                continue
            img1_path = os.path.join(folder_path, image_files[0])
            img2_path = os.path.join(folder_path, image_files[1])
            match_count, match_img = match(img1_path, img2_path)

            # Determine the ground truth
            actual_match = 1 if "same" in folder.lower() else 0  # 1 for same, 0 for different
            y_true.append(actual_match)

            # Decision based on good matches count
            predicted_match = 1 if match_count > threshold else 0
            y_pred.append(predicted_match)
            result = "sift_flann_matched" if predicted_match == 1 else "sift_flann_unmatched"
            print(f"{folder}: {result.upper()} ({match_count} good matches)")
            if match_img is not None:
                match_img_filename = f"{folder}_{result}.png"
                match_img_path = os.path.join(results_folder, match_img_filename)
                cv.imwrite(match_img_path, match_img)
                print(f"Saved match image at: {match_img_path}")


set_path = r"C:\Users\bjorn\ikt213g25h\IKT213_sodal\assignment_4\test\sets"
results = r"C:\Users\bjorn\ikt213g25h\IKT213_sodal\assignment_4\test\results\sift_flann"

start = time.perf_counter()
process_set(set_path, results)
end = time.perf_counter()

print("Elapsed time: ", end-start, "s")