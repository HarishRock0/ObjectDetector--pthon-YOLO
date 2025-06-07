import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO  # For YOLOv8

# Initialize YOLOv8 model
model = YOLO('yolov8n.pt')

# Initialize video capture (0 for webcam, or path to video file)
cap = cv2.VideoCapture(0)  # Change to your video source

# Store the tracking history
track_history = {}

# Dictionary to map class IDs to class names (using COCO dataset classes)
class_names = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Color palette for different tracks
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
          (255, 0, 255), (0, 255, 255), (255, 128, 0),
          (0, 255, 128), (128, 0, 255), (128, 255, 0)]

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    # Run YOLOv8 tracking on the frame
    results = model.track(frame, persist=True, tracker="bytetrack.yaml")
    
    # Visualize the results on the frame
    annotated_frame = results[0].plot()
    
    # Get the boxes and track IDs
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        
        # Plot the tracks and class labels
        for box, track_id, class_id in zip(boxes, track_ids, class_ids):
            x, y, w, h = box
            class_name = class_names[class_id]
            
            # Store tracking history
            if track_id not in track_history:
                track_history[track_id] = deque(maxlen=30)
            
            track_history[track_id].append((float(x), float(y)))
            
            # Draw the tracking lines
            points = np.array(track_history[track_id], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=colors[track_id % len(colors)], thickness=2)
            
            # Display class name and track ID
            label = f"{class_name} ID:{track_id}"
            cv2.putText(annotated_frame, label, (int(x - w/2), int(y - h/2 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[track_id % len(colors)], 2)
    
    # Display the annotated frame
    cv2.imshow("Object Detection and Tracking", annotated_frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()