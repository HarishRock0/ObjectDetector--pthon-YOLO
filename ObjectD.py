import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

# Load COCO class names
COCO_CLASSES = [
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

# Color palette for track lines
TRACK_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 0, 128), (255, 128, 0), (0, 128, 255), (128, 255, 0)
]

def draw_tracking_info(frame, track_history, results):
    if results[0].boxes.id is None:
        return frame

    boxes = results[0].boxes.xywh.cpu().numpy()
    ids = results[0].boxes.id.int().cpu().numpy()
    class_ids = results[0].boxes.cls.int().cpu().numpy()

    for box, track_id, class_id in zip(boxes, ids, class_ids):
        x, y, w, h = box
        class_name = COCO_CLASSES[class_id]
        color = TRACK_COLORS[track_id % len(TRACK_COLORS)]

        # Save position history
        if track_id not in track_history:
            track_history[track_id] = deque(maxlen=30)
        track_history[track_id].append((int(x), int(y)))

        # Draw tracking path
        points = np.array(track_history[track_id], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [points], False, color, 2)

        # Draw label and bounding box
        cv2.rectangle(frame, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), color, 2)
        label = f"{class_name} ID:{track_id}"
        cv2.putText(frame, label, (int(x - w/2), int(y - h/2) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame

def main():
    print("Loading YOLOv8 model...")
    model = YOLO('yolov8n.pt')

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open camera.")
        return

    track_history = {}

    print("Starting tracking... Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame.")
                break

            # Optional: resize for performance
            frame = cv2.resize(frame, (640, 480))

            results = model.track(frame, persist=True, tracker="bytetrack.yaml", imgsz=320)
            annotated_frame = results[0].plot()

            annotated_frame = draw_tracking_info(annotated_frame, track_history, results)

            cv2.imshow("YOLOv8 Object Tracking", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty("YOLOv8 Object Tracking", cv2.WND_PROP_VISIBLE) < 1:
                break

    except Exception as e:
        print(f"Exception occurred: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Tracking stopped. Resources released.")

if __name__ == "__main__":
    main()
