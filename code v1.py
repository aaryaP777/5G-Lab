import time
import cv2
import torch
import requests
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Load YOLOv5 pre-trained model (from Ultralytics)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.3  # Lower confidence threshold for better detection

# Define anomaly classes 
ANOMALY_CLASSES = ['knife', 'scissors', 'fire', 'fight', 'Suspecious_object']  # Added person for testing

# Add person class filter
def filter_detections(detections):
    return detections[detections['name'] != 'person']

# Initialize video capture
cap = cv2.VideoCapture(0)

# 5G camera
# RTSP_URL = "rtsp://admin:admin123@192.168.128.10:554/avstream/channel=1/stream=1.sdp"  

# Initialize video capture with RTSP
# cap = cv2.VideoCapture(RTSP_URL)

print("Starting surveillance... Press 'q' to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    
    # Run inference
    results = model(frame, size=640)
    detections = results.pandas().xyxy[0]

    # Filter out person detections
    detections = filter_detections(detections)
    
    # Debug: Print all detections
    if not detections.empty:
        print("\nAll detected objects:")
        for _, row in detections.iterrows():
            print(f"- {row['name']} (confidence: {row['confidence']:.2f})")

    # Check for anomalies
    for _, row in detections.iterrows():
        if row['name'] in ANOMALY_CLASSES and row['confidence'] > 0.3:
            # Alert message
            print("\n" + "="*50)
            print(f"ALERT: {row['name']} detected!")
            print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Confidence: {row['confidence']:.2f}")
            print("="*50 + "\n")

            # Save frame
            filename = f"anomaly_{row['name']}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)

    # Display frame with detection boxes
    annotated_frame = results.render()[0]
    cv2.imshow("5G Surveillance Feed", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()