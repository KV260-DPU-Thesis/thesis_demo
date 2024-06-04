import numpy as np
import cv2
import os
import time
import argparse
import tensorflow as tf

# For using GPU or CPU
parser = argparse.ArgumentParser(description='Set GPU usage for TensorFlow')
parser.add_argument('--use_gpu', action='store_true', help='Use GPU if available')
args = parser.parse_args()

if args.use_gpu:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU is enabled and configured.")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU devices found.")
else:
    print("GPU usage is disabled. Running on CPU.")

# Constants
MODEL_PATH = "flower_model.h5"
CAMERA_INDEX = 0  # Default camera index

flower_labels = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully")

# Open the camera
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

all_process_start_time = time.time()
total_inference_duration_ms = 0
num_frames = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    image = cv2.resize(frame, (224, 224))
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # normalization

    # Start timing for inference
    inference_start_time = time.time()

    # Predict
    predictions = model.predict(image, verbose=0)
    idx = np.argmax(predictions[0])

    # Inference duration
    inference_end_time = time.time()
    inference_duration_ms = (inference_end_time - inference_start_time) * 1000
    total_inference_duration_ms += inference_duration_ms
    num_frames += 1

    # Display the prediction on the frame
    label = f"{flower_labels[idx]} ({predictions[0][idx]:.5f})"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Real-time Inference', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Summarize the process
average_inference_time_ms = total_inference_duration_ms / num_frames
print(f"Average inference duration: {average_inference_time_ms:.5f} ms")
print(f"Total inference duration: {total_inference_duration_ms:.5f} ms")

all_process_end_time = time.time()
all_process_duration_ms = (all_process_end_time - all_process_start_time) * 1000
print(f"All process duration: {all_process_duration_ms:.5f} ms")

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
