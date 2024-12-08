import cv2
from ultralytics import YOLO

# Load the YOLO segmentation model
model = YOLO("yolo11n-seg.pt")

# Initialize the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set the window to full screen
cv2.namedWindow("YOLO Segmentation", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("YOLO Segmentation", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    # Perform segmentation
    results = model(frame)

    # Retrieve and visualize the segmentation mask on the frame
    segmented_frame = results[0].plot()

    # Display the frame
    cv2.imshow("YOLO Segmentation", segmented_frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
