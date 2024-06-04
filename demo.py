import sys
import numpy as np
import cv2
import vitis_ai_library
import xir
import time
import os

# Check if the model name argument is provided
if len(sys.argv) != 2:
    print("Usage: python script.py <model_name>")
    sys.exit(1)

DPU_CONFIG = sys.argv[1]

# File path
MODEL_PATH = f"./xmodel_outputs/{DPU_CONFIG}/{DPU_CONFIG}.xmodel"

# Initialize camera
CAMERA_INDEX = 0  # Default camera index
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("Error: Could not open camera.")
    sys.exit(1)

all_process_start_time = time.time()

g = xir.Graph.deserialize(MODEL_PATH)
runner = vitis_ai_library.GraphRunner.create_graph_runner(g)

# Input buffer
inputDim = tuple(runner.get_inputs()[0].get_tensor().dims)
inputData = [np.empty(inputDim, dtype=np.int8)]

# Variables to calculate FPS
frame_count = 0
fps = 0
start_time = time.time()
total_inference_duration_ms = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Normalization
    image = frame / 255.0

    # Quantization
    fix_point = runner.get_input_tensors()[0].get_attr("fix_point")
    scale = 2 ** fix_point
    image = (image * scale).round()
    image = image.astype(np.int8)

    # Set input data
    inputData[0][0] = image

    # Start timing for inference
    inference_start_time = time.time()

    # Output buffer
    outputData = runner.get_outputs()

    # Prediction
    job_id = runner.execute_async(inputData, outputData)
    runner.wait(job_id)

    inference_end_time = time.time()

    # Inference duration for this frame
    inference_duration_ms = (inference_end_time - inference_start_time) * 1000
    total_inference_duration_ms += inference_duration_ms

    resultList = np.asarray(outputData[0])[0]
    resultIdx = resultList.argmax()
    resultVal = resultList[0][0][resultIdx]

    # Calculate FPS
    frame_count += 1
    end_time = time.time()
    elapsed_time = end_time - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()

    # Display the FPS on the frame
    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Real-time Video Stream', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Summarize the process
num_frames = frame_count
average_inference_time_ms = total_inference_duration_ms / num_frames
print(f"Average inference duration: {average_inference_time_ms:.5f} ms")
print(f"Total inference duration: {total_inference_duration_ms:.5f} ms")

all_process_end_time = time.time()
all_process_duration_ms = (all_process_end_time - all_process_start_time) * 1000
print(f"All process duration: {all_process_duration_ms:.5f} ms")

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()

del runner
