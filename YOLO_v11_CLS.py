import cv2
from ultralytics import YOLO

# Load the YOLO11 classification model
model = YOLO("yolo11n-cls.pt")

# Open the webcam (default camera index is 0)
cap = cv2.VideoCapture(0)

# Set fullscreen mode
cv2.namedWindow("YOLO Real-Time Classification", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("YOLO Real-Time Classification", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from camera.")
        break

    # Convert the frame from BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform classification
    results = model(rgb_frame)

    # Extract classification probabilities and class names
    if results and results[0].probs is not None:
        probs = results[0].probs  # Classification probabilities
        class_id = probs.top1  # Index of the highest probability class
        confidence = probs.top1conf.item()  # Confidence score for the top class
        class_name = model.names[class_id]  # Access class names from the model

        label = f"{class_name}: {confidence:.2f}"

        # Display the label on the frame
        cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the output in fullscreen
    cv2.imshow("YOLO Real-Time Classification", frame)

    # Break on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
