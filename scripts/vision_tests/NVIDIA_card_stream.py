# This script is designed to run YOLOv11 on a camera stream using multiple GPUs for card detection.
# This script will not work if the computer does not have an NVIDIA GPU.

import os
import time
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from card_tracker import CardTracker
from threading import Thread

# Ensure YOLO uses all available GPUs
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.set_device(0)  # Default to first GPU

# Multi-GPU setup
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for inference!")
    device = torch.device("cuda")

# Load YOLO model on GPU
MODEL_PATH = "card_weights.pt"
model = YOLO(MODEL_PATH).to(device)

# Define all possible cards in the deck
ALL_CARDS = [
    "Ace of Spades", "2 of Spades", "3 of Spades", "4 of Spades", "5 of Spades", "6 of Spades", "7 of Spades", "8 of Spades", "9 of Spades", "10 of Spades", "Jack of Spades", "Queen of Spades", "King of Spades",
    "Ace of Hearts", "2 of Hearts", "3 of Hearts", "4 of Hearts", "5 of Hearts", "6 of Hearts", "7 of Hearts", "8 of Hearts", "9 of Hearts", "10 of Hearts", "Jack of Hearts", "Queen of Hearts", "King of Hearts",
    "Ace of Diamonds", "2 of Diamonds", "3 of Diamonds", "4 of Diamonds", "5 of Diamonds", "6 of Diamonds", "7 of Diamonds", "8 of Diamonds", "9 of Diamonds", "10 of Diamonds", "Jack of Diamonds", "Queen of Diamonds", "King of Diamonds",
    "Ace of Clubs", "2 of Clubs", "3 of Clubs", "4 of Clubs", "5 of Clubs", "6 of Clubs", "7 of Clubs", "8 of Clubs", "9 of Clubs", "10 of Clubs", "Jack of Clubs", "Queen of Clubs", "King of Clubs"
]

# Camera settings
CAMERA_INDEX = 0
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
BATCH_SIZE = 4

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit(1)

# Set camera resolution
cap.set(3, FRAME_WIDTH)
cap.set(4, FRAME_HEIGHT)

# Initialize CardTracker
card_tracker = CardTracker(ALL_CARDS, min_frames=5)

# Buffer for batch processing
frame_buffer = []
print("🎥 Starting camera stream... Press 'q' to exit, 'r' to show remaining cards, 'x' to reset.")

def capture_frames():
    """ Continuously captures frames in a separate thread to reduce latency. """
    global frame_buffer
    while True:
        ret, frame = cap.read()
        if ret:
            frame_buffer.append(frame)
            if len(frame_buffer) > BATCH_SIZE:
                frame_buffer.pop(0)

# Start frame capture in a separate thread
Thread(target=capture_frames, daemon=True).start()

# Start YOLO inference loop
while True:
    t_start = time.perf_counter()

    if len(frame_buffer) == 0:
        continue

    # Process a batch of frames
    frames_to_process = frame_buffer.copy()
    frame_buffer.clear()

    # Run YOLO on all frames in batch
    results = model(frames_to_process, verbose=False, device=device)

    # Track detected cards
    detected_cards = []

    for result in results:
        detections = result.boxes
        for i in range(len(detections)):
            xyxy_tensor = detections[i].xyxy.cpu()
            xyxy = xyxy_tensor.numpy().squeeze()
            xmin, ymin, xmax, ymax = xyxy.astype(int)

            classidx = int(detections[i].cls.item())
            classname = model.names[classidx]
            conf = detections[i].conf.item()

            if conf > 0.5:
                cv2.rectangle(frames_to_process[0], (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                label = f"{classname}: {int(conf * 100)}%"
                cv2.putText(frames_to_process[0], label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                detected_cards.append(classname)

    # Update tracker with detected cards
    card_tracker.update(detected_cards)

    # Calculate FPS
    fps = 1 / (time.perf_counter() - t_start)
    cv2.putText(frames_to_process[0], f"FPS: {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frames_to_process[0], f"Objects: {len(detected_cards)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("YOLO Detection", frames_to_process[0])

    # Key press actions
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        remaining_cards = card_tracker.get_remaining_cards()
        print(f"Cards Left in Deck: {remaining_cards if remaining_cards else 'None! All cards have been shown!'}")
    elif key == ord('x'):
        card_tracker.reset()

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Camera stream closed.")
