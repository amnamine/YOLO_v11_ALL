from ultralytics import YOLO
import cv2

# Load the YOLO model (replace 'yolo11n.pt' with the correct path to your model file)
model = YOLO('yolo11n.pt')

# Open webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Set the screen resolution to 1366x768
screen_width = 1366
screen_height = 768

# Create a fullscreen window
cv2.namedWindow("YOLO Real-Time Detection", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("YOLO Real-Time Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Loop through frames from the webcam
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Run YOLO detection
    results = model(frame, stream=False)  # Run inference on the current frame

    # Annotate the frame with detection results
    annotated_frame = results[0].plot()

    # Resize the frame to fit the 1366x768 resolution
    frame_resized = cv2.resize(annotated_frame, (screen_width, screen_height))

    # Display the annotated frame
    cv2.imshow("YOLO Real-Time Detection", frame_resized)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
