import cv2 as cv
import time
import os

def preprocess(path):
    img = cv.imread(path, 0)
    _, img_bin = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    return img_bin

def match(img1_path, img2_path):
    img1 = preprocess(img1_path)
    img2 = preprocess(img2_path)

    orb = cv.ORB_create(nfeatures=1000)

    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)
    if descriptors1 is None or descriptors2 is None:
        return 0, None

    brute_force = cv.BFMatcher_create(cv.NORM_HAMMING, crossCheck=False)

    matches = brute_force.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

    match_img = cv.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return len(good_matches), match_img

def process_set(set_path, results_folder):
    threshold = 20
    y_true = []
    y_pred = []

    os.makedirs(results_folder, exist_ok=True)

    for folder in sorted(os.listdir(set_path)):
        folder_path = os.path.join(set_path, folder)
        if os.path.isdir(folder_path):
            image_files = [f for f in os.listdir(folder_path) if f.endswith(('.tif', '.png', '.jpg'))]
            if len(image_files) != 2:
                print(f"Skipping {folder}, expected 2 images but found {len(image_files)}")
                continue
            img1_path = os.path.join(folder_path, image_files[0])
            img2_path = os.path.join(folder_path, image_files[1])
            match_count, match_img = match(img1_path, img2_path)

            actual_match = 1 if "same" in folder.lower() else 0
            y_true.append(actual_match)

            predicted_match = 1 if match_count > threshold else 0
            y_pred.append(predicted_match)
            result = "orb_bf_matched" if predicted_match == 1 else "orb_bf_unmatched"
            print(f"{folder}: {result.upper()} ({match_count} good matches)")

            if match_img is not None:
                match_img_filename = f"{folder}_{result}.png"
                match_img_path = os.path.join(results_folder, match_img_filename)
                cv.imwrite(match_img_path, match_img)
                print(f"Saved match image at: {match_img_path}")

set_path = r"C:\Users\bjorn\ikt213g25h\IKT213_sodal\assignment_4\UiA_images\sets"
results_folder = r"C:\Users\bjorn\ikt213g25h\IKT213_sodal\assignment_4\UiA_images\results\orb_bf"

start = time.perf_counter()
process_set(set_path, results_folder)
end = time.perf_counter()

print("Elsapsed time:", end - start, "s")