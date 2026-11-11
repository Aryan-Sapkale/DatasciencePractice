
import cv2
from ultralytics import YOLO
import os
from datetime import datetime

# Create output folder
os.makedirs("output", exist_ok=True)

# Load pretrained YOLO model
model = YOLO("yolov8n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access webcam!")
    exit()

print("Real-Time Object Detection Started!")
print("Press Q to quit.")
print("Press S to save screenshot.")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Unable to read webcam frame.")
        break

    # Detect objects
    results = model(
        frame,
        conf=0.45,
        verbose=False
    )

    # Draw bounding boxes and labels
    annotated_frame = results[0].plot()

    # Display detection window
    cv2.imshow(
        "Real-Time Object Detection",
        annotated_frame
    )

    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    elif key == ord("s"):

        filename = datetime.now().strftime(
            "detection_%Y%m%d_%H%M%S.jpg"
        )

        save_path = os.path.join(
            "output", filename
        )

        cv2.imwrite(save_path, annotated_frame)

        print("Screenshot saved:", save_path)

# Release resources
cap.release()
cv2.destroyAllWindows()

print("Object Detection Stopped!")
