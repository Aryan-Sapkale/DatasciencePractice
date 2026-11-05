
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Project folders
BASE_DIR = Path(__file__).resolve().parent.parent
IMAGE1_PATH = BASE_DIR / "dataset" / "images" / "my_photo.jpg"
IMAGE2_PATH = BASE_DIR / "dataset" / "images" / "my_photo2.jpg"
OUTPUT_DIR = BASE_DIR / "outputs" / "task2"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------
# STEP 1: LOAD IMAGES
# -----------------------------------------

img1 = cv2.imread(str(IMAGE1_PATH))
img2 = cv2.imread(str(IMAGE2_PATH))

if img1 is None or img2 is None:
    raise FileNotFoundError(
        "Check that my_photo.jpg and my_photo2.jpg "
        "are inside dataset/images"
    )

# Resize for easier processing
img1 = cv2.resize(img1, (600, 400))
img2 = cv2.resize(img2, (600, 400))

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


# -----------------------------------------
# STEP 2: HOG FEATURE EXTRACTION
# -----------------------------------------

hog = cv2.HOGDescriptor(
    (64, 128),
    (16, 16),
    (8, 8),
    (8, 8),
    9
)

# HOG requires a 64 x 128 image
hog_image = cv2.resize(gray1, (64, 128))

hog_features = hog.compute(hog_image)

print("\nHOG FEATURES")
print("Number of HOG features:", len(hog_features))

# Visualize HOG feature values
plt.figure(figsize=(10, 4))
plt.plot(hog_features.flatten())
plt.title("HOG Feature Visualization")
plt.xlabel("Feature Index")
plt.ylabel("Feature Value")
plt.tight_layout()

plt.savefig(
    str(OUTPUT_DIR / "hog_visualization.png")
)
plt.close()


# -----------------------------------------
# STEP 3: SIFT KEYPOINT DETECTION
# -----------------------------------------

sift = cv2.SIFT_create()

keypoints1, descriptors1 = sift.detectAndCompute(
    gray1, None
)

keypoints2, descriptors2 = sift.detectAndCompute(
    gray2, None
)

print("\nSIFT FEATURES")
print("Image 1 keypoints:", len(keypoints1))
print("Image 2 keypoints:", len(keypoints2))

# Draw keypoints
keypoints_image1 = cv2.drawKeypoints(
    img1,
    keypoints1,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

keypoints_image2 = cv2.drawKeypoints(
    img2,
    keypoints2,
    None,
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

cv2.imwrite(
    str(OUTPUT_DIR / "sift_keypoints1.jpg"),
    keypoints_image1
)

cv2.imwrite(
    str(OUTPUT_DIR / "sift_keypoints2.jpg"),
    keypoints_image2
)


# -----------------------------------------
# STEP 4: SIFT FEATURE MATCHING
# -----------------------------------------

def match_features(des1, des2, image1, image2, filename):

    if des1 is None or des2 is None:
        print("No descriptors found for matching.")
        return

    # Match features using Brute Force Matcher
    matcher = cv2.BFMatcher()

    matches = matcher.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good_matches = []

    for pair in matches:
        if len(pair) == 2:
            m, n = pair

            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

    print("Good matches:", len(good_matches))

    # Display the best 50 matches
    result = cv2.drawMatches(
        image1,
        keypoints1,
        image2,
        keypoints2,
        good_matches[:50],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    cv2.imwrite(
        str(OUTPUT_DIR / filename),
        result
    )

    return len(good_matches)


print("\nORIGINAL IMAGE MATCHING")

original_matches = match_features(
    descriptors1,
    descriptors2,
    img1,
    img2,
    "original_matching.jpg"
)


# -----------------------------------------
# STEP 5: ROTATE THE SECOND IMAGE
# -----------------------------------------

height, width = gray2.shape

rotation_matrix = cv2.getRotationMatrix2D(
    (width // 2, height // 2),
    30,
    1
)

rotated_image = cv2.warpAffine(
    img2,
    rotation_matrix,
    (width, height)
)

rotated_gray = cv2.cvtColor(
    rotated_image,
    cv2.COLOR_BGR2GRAY
)

rotated_keypoints, rotated_descriptors = (
    sift.detectAndCompute(rotated_gray, None)
)

print("\nROTATED IMAGE MATCHING")

rotated_matches = match_features(
    descriptors1,
    rotated_descriptors,
    img1,
    rotated_image,
    "rotated_matching.jpg"
)


# -----------------------------------------
# STEP 6: RESIZE THE SECOND IMAGE
# -----------------------------------------

resized_image = cv2.resize(
    img2,
    None,
    fx=0.75,
    fy=0.75
)

resized_gray = cv2.cvtColor(
    resized_image,
    cv2.COLOR_BGR2GRAY
)

resized_keypoints, resized_descriptors = (
    sift.detectAndCompute(resized_gray, None)
)

print("\nRESIZED IMAGE MATCHING")

resized_matches = match_features(
    descriptors1,
    resized_descriptors,
    img1,
    resized_image,
    "resized_matching.jpg"
)


# -----------------------------------------
# STEP 7: COMPARE RESULTS
# -----------------------------------------

print("\nMATCHING COMPARISON")
print("Original image matches:", original_matches)
print("Rotated image matches:", rotated_matches)
print("Resized image matches:", resized_matches)

print("\nAll Task 2 outputs saved in:", OUTPUT_DIR)
print("Task 2 completed successfully!")
