import cv2
import torch
from ultralytics import YOLO

# Load YOLOv11 model
model_path = "yolo11n-obb.pt"  # Ensure this file is in the same directory
model = YOLO(model_path)

# Initialize webcam (camera index 0 for default webcam)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set the camera to fullscreen mode
cv2.namedWindow("YOLOv11 OBB Detection", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("YOLOv11 OBB Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    # Run YOLOv11 model on the frame
    results = model(frame)

    # Parse the results for Oriented Bounding Boxes
    annotated_frame = results[0].plot()  # This function overlays detected boxes on the image

    # Display the frame with detections
    cv2.imshow("YOLOv11 OBB Detection", annotated_frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
