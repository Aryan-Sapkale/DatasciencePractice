
import cv2
import numpy as np
import os
import sys

# Input and output folders
INPUT_FOLDER = "input"
OUTPUT_FOLDER = "output"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def process_image(image_path):

    # Read image
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Image not found!")
        return

    print("Image loaded successfully!")

    # 1. Grayscale conversion
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Canny Edge Detection
    edges = cv2.Canny(blurred, 100, 200)

    # 4. Thresholding
    _, threshold = cv2.threshold(
        gray, 127, 255, cv2.THRESH_BINARY
    )

    # 5. Contour Detection
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contour_image = image.copy()

    cv2.drawContours(
        contour_image,
        contours,
        -1,
        (0, 255, 0),
        2
    )

    # Save processed images
    results = {
        "grayscale": gray,
        "blurred": blurred,
        "edges": edges,
        "threshold": threshold,
        "contours": contour_image
    }

    for name, result in results.items():
        output_path = os.path.join(
            OUTPUT_FOLDER, name + ".jpg"
        )
        cv2.imwrite(output_path, result)
        print("Saved:", output_path)

    print("\nComputer Vision Processing Completed!")
    print("Total contours detected:", len(contours))


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python src/main.py input/sample.jpg")
    else:
        process_image(sys.argv[1])
