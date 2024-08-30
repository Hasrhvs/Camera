import cv2
import numpy as np
from datetime import datetime
import os

# Load the YOLO model configuration and weights (using YOLOv3-tiny for faster processing)
MODEL_CONFIG = "../models/yolov3.cfg"
MODEL_WEIGHTS = "../models/yolov3.weights"

# Initialize the YOLO model
net = cv2.dnn.readNet(MODEL_WEIGHTS, MODEL_CONFIG)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Set up the video capture using your phone's IP camera feed
# Replace the URL with the one provided by your IP Camera app
cap = cv2.VideoCapture("http://192.168.1.103:8080/video")

# Define the codec and create VideoWriter object for saving detected segments
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None

# Variables to track detection state and frame count
person_detected = False
frame_count = 0
max_frames = 100  # Stop after processing 100 frames
processing_fps = 5  # Limit processing to 5 frames per second

# Create output directories if they don't exist
output_dirs = ["../output/detected_videos", "../output/screenshots"]
for directory in output_dirs:
    os.makedirs(directory, exist_ok=True)

# Main loop for processing video frames
while frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # Resize frame to 320x320 for faster processing
    resized_frame = cv2.resize(frame, (320, 320))

    # Prepare the image for YOLO model
    blob = cv2.dnn.blobFromImage(resized_frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Lists to hold detection data
    class_ids = []
    confidences = []
    boxes = []
    count_people = 0

    # Iterate through each detection and filter for persons
    for out_ in outs:
        for detection in out_:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if class_id == 0 and confidence > 0.5:  # Class ID 0 is for 'person'
                count_people += 1
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # If a person is detected, save the video segment and screenshot
    if len(indexes) > 0:
        if not person_detected:
            person_detected = True
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            out = cv2.VideoWriter(f'../output/detected_videos/detected_{timestamp}.avi', fourcc, 20.0, (width, height))
            cv2.imwrite(f'../output/screenshots/screenshot_{timestamp}.jpg', frame)

        # Draw bounding boxes around detected persons
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(count_people)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if out is not None:
            out.write(frame)  # Write the frame to the video file
        frame_count += 1

    # If no person is detected, close the video writer and log the event
    else:
        if person_detected:
            person_detected = False
            if out is not None:
                out.release()
            with open("../output/detection_log.txt", "a") as log_file:
                log_file.write(f"{timestamp}, People detected: {count_people}\n")

    # Display the video feed with detections
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
