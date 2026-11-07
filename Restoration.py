
import cv2
import numpy as np
from pathlib import Path


# Project folders
BASE_DIR = Path(__file__).resolve().parent.parent
IMAGE_PATH = BASE_DIR / "dataset" / "images" / "my_photo.jpg"
OUTPUT_DIR = BASE_DIR / "outputs" / "task1"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Read the original photograph
image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(
        "Image not found. Check dataset/images/my_photo.jpg"
    )

# Resize large images for easier processing
image = cv2.resize(image, (600, 400))

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# -----------------------------------------
# PART A: IMAGE RESTORATION
# -----------------------------------------

# 1. Add Gaussian noise
np.random.seed(42)

gaussian_noise = np.random.normal(
    0, 25, gray.shape
).astype(np.float32)

noisy_gaussian = np.clip(
    gray.astype(np.float32) + gaussian_noise,
    0, 255
).astype(np.uint8)

# 2. Add salt-and-pepper noise
noisy_sp = gray.copy()

random_values = np.random.random(gray.shape)

noisy_sp[random_values < 0.02] = 0
noisy_sp[random_values > 0.98] = 255

# 3. Apply Gaussian and median filters
gaussian_restored = cv2.GaussianBlur(
    noisy_gaussian, (5, 5), 0
)

median_restored = cv2.medianBlur(
    noisy_sp, 5
)

# 4. Calculate PSNR and SSIM
def compare_images(original, restored):

    # Calculate PSNR using OpenCV
    psnr = cv2.PSNR(original, restored)

    # Calculate SSIM using NumPy
    original = original.astype(np.float64)
    restored = restored.astype(np.float64)

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu1 = cv2.GaussianBlur(original, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(restored, (11, 11), 1.5)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(
        original * original, (11, 11), 1.5
    ) - mu1_sq

    sigma2_sq = cv2.GaussianBlur(
        restored * restored, (11, 11), 1.5
    ) - mu2_sq

    sigma12 = cv2.GaussianBlur(
        original * restored, (11, 11), 1.5
    ) - mu1_mu2

    numerator = (
        (2 * mu1_mu2 + C1) *
        (2 * sigma12 + C2)
    )

    denominator = (
        (mu1_sq + mu2_sq + C1) *
        (sigma1_sq + sigma2_sq + C2)
    )

    ssim = np.mean(numerator / denominator)

    return psnr, ssim

print("\nIMAGE RESTORATION RESULTS")

for name, restored in [
    ("Gaussian Filter", gaussian_restored),
    ("Median Filter", median_restored)
]:
    psnr, ssim = compare_images(gray, restored)

    print(f"\n{name}")
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim:.4f}")


# -----------------------------------------
# PART B: SEGMENTATION
# -----------------------------------------

# 5. Binary thresholding using Otsu's method
_, binary = cv2.threshold(
    gray, 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# 6. Morphological operations
kernel = np.ones((5, 5), np.uint8)

erosion = cv2.erode(binary, kernel, iterations=1)

dilation = cv2.dilate(binary, kernel, iterations=1)

opening = cv2.morphologyEx(
    binary, cv2.MORPH_OPEN, kernel
)

closing = cv2.morphologyEx(
    binary, cv2.MORPH_CLOSE, kernel
)

# 7. Canny edge detection
edges = cv2.Canny(gray, 100, 200)

# 8. Watershed segmentation
# Use thresholding to identify foreground objects.
# This works best when objects are separated from
# the background.

_, sure_background = cv2.threshold(
    gray, 0, 255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

sure_background = cv2.dilate(
    sure_background, kernel, iterations=3
)

distance = cv2.distanceTransform(
    binary, cv2.DIST_L2, 5
)

_, sure_foreground = cv2.threshold(
    distance,
    0.4 * distance.max(),
    255,
    cv2.THRESH_BINARY
)

sure_foreground = np.uint8(sure_foreground)

unknown = cv2.subtract(
    sure_background, sure_foreground
)

_, markers = cv2.connectedComponents(
    sure_foreground
)

markers = markers + 1
markers[unknown == 255] = 0

# Watershed requires a 3-channel image
watershed_image = image.copy()

cv2.watershed(watershed_image, markers)

# Mark watershed boundaries in red
watershed_result = image.copy()

watershed_result[markers == -1] = [0, 0, 255]


# -----------------------------------------
# SAVE ALL OUTPUTS
# -----------------------------------------

outputs = {
    "original.jpg": image,
    "gaussian_noisy.jpg": noisy_gaussian,
    "salt_pepper_noisy.jpg": noisy_sp,
    "gaussian_restored.jpg": gaussian_restored,
    "median_restored.jpg": median_restored,
    "binary.jpg": binary,
    "erosion.jpg": erosion,
    "dilation.jpg": dilation,
    "opening.jpg": opening,
    "closing.jpg": closing,
    "canny_edges.jpg": edges,
    "watershed.jpg": watershed_result
}

for filename, result in outputs.items():
    cv2.imwrite(str(OUTPUT_DIR / filename), result)

print("\nAll output images saved in:", OUTPUT_DIR)
print("Task 1 completed successfully!")
