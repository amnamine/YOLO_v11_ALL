import cv2
import torch
from ultralytics import YOLO

# Load the YOLO model
model_path = "./yolo11n-pose.pt"  # Path to your YOLOv11n model
model = YOLO(model_path)  # Load the YOLOv11 model

# Initialize the camera
camera_index = 0  # Default camera index
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print("Error: Cannot open the camera.")
    exit()

# Set the window to fullscreen
cv2.namedWindow("YOLO Pose Detection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("YOLO Pose Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Real-time pose detection loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture frame.")
        break

    # Perform pose detection
    results = model.predict(source=frame, save=False, save_txt=False, conf=0.5)

    # Annotate the frame with the detections
    annotated_frame = results[0].plot()  # Draw detections on the frame

    # Display the annotated frame
    cv2.imshow("YOLO Pose Detection", annotated_frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
