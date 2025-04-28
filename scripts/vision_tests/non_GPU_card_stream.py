# Use this script to run the YOLOv11 model on a system without GPU support
# It will run if there is a GPU, but it is recommended to run NVIDIA_card_stream.py instead

import os
import time
import cv2
import numpy as np
from ultralytics import YOLO
from card_tracker import CardTracker

# Define all possible cards in the deck
ALL_CARDS = [
    "Ace of Spades", "2 of Spades", "3 of Spades", "4 of Spades", "5 of Spades", "6 of Spades", "7 of Spades", "8 of Spades", "9 of Spades", "10 of Spades", "Jack of Spades", "Queen of Spades", "King of Spades",
    "Ace of Hearts", "2 of Hearts", "3 of Hearts", "4 of Hearts", "5 of Hearts", "6 of Hearts", "7 of Hearts", "8 of Hearts", "9 of Hearts", "10 of Hearts", "Jack of Hearts", "Queen of Hearts", "King of Hearts",
    "Ace of Diamonds", "2 of Diamonds", "3 of Diamonds", "4 of Diamonds", "5 of Diamonds", "6 of Diamonds", "7 of Diamonds", "8 of Diamonds", "9 of Diamonds", "10 of Diamonds", "Jack of Diamonds", "Queen of Diamonds", "King of Diamonds",
    "Ace of Clubs", "2 of Clubs", "3 of Clubs", "4 of Clubs", "5 of Clubs", "6 of Clubs", "7 of Clubs", "8 of Clubs", "9 of Clubs", "10 of Clubs", "Jack of Clubs", "Queen of Clubs", "King of Clubs"
]

# Hardcoded values
MODEL_PATH = "card_weights.pt"
CAMERA_INDEX = 0

# Camera Resolution
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

# Check if model file exists
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model file '{MODEL_PATH}' not found!")
    exit(1)

# Load YOLO model
model = YOLO(MODEL_PATH)

# Open camera
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)


if not cap.isOpened():
    print("ERROR: Could not open laptop camera. Make sure it's not being used by another application.")
    exit(1)

# Set camera resolution
cap.set(3, FRAME_WIDTH)
cap.set(4, FRAME_HEIGHT)

# Initialize CardTracker with all cards
card_tracker = CardTracker(ALL_CARDS, min_frames=5)

print("Starting camera stream... Press 'q' to exit, 'r' to show remaining cards, 'x' to clear memory")

# Start Camera Loop
while True:
    t_start = time.perf_counter()

    ret, frame = cap.read()
    if not ret:
        print("ERROR: Failed to capture image from webcam.")
        break

    # Run YOLO inference
    results = model(frame, verbose=False)

    # Extract detections
    detections = results[0].boxes

    # Object counting
    object_count = 0
    detected_cards = []

    for i in range(len(detections)):
        xyxy_tensor = detections[i].xyxy.cpu()
        xyxy = xyxy_tensor.numpy().squeeze()
        xmin, ymin, xmax, ymax = xyxy.astype(int)

        classidx = int(detections[i].cls.item())
        classname = model.names[classidx]
        conf = detections[i].conf.item()

        if conf > 0.5:
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            label = f"{classname}: {int(conf * 100)}%"
            cv2.putText(frame, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            object_count += 1

            detected_cards.append(classname)  # Store detected card name

    # Update tracker with detected cards
    card_tracker.update(detected_cards)

    # Display FPS and object count
    fps = 1 / (time.perf_counter() - t_start)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Objects: {object_count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("YOLO Detection", frame)

    # Check for key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        remaining_cards = card_tracker.get_remaining_cards()
        print(f"Cards Left in Deck: {remaining_cards if remaining_cards else 'None! All cards have been shown!'}")
    elif key == ord('x'):  # Reset the deck when 'x' is pressed
        card_tracker.reset()

cap.release()
cv2.destroyAllWindows()
print("Camera stream closed.")
