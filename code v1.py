import time
import cv2
import torch
import psycopg2
from psycopg2 import sql
import requests
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

load_dotenv()

# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname="5g_surveillance",
    user="postgres",
    password=os.getenv("sqlPass"), 
    host="localhost",
    port="5432"
)
cursor = conn.cursor()

# Function to save anomaly into DB
def save_anomaly_to_db(anomaly_name, confidence, frame):
    try:
        # Convert frame to binary
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        insert_query = """
            INSERT INTO anomalies (anomaly_name, confidence, timestamp, frame)
            VALUES (%s, %s, NOW(), %s)
        """
        cursor.execute(insert_query, (anomaly_name, float(confidence), psycopg2.Binary(frame_bytes)))
        conn.commit()
        print(f"Saved {anomaly_name} with {confidence:.2f} confidence to DB")
    except Exception as e:
        print(f"Error saving to DB: {e}")

# Load YOLOv5 pre-trained model (from Ultralytics)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.3  # Lower confidence threshold for better detection

# Define anomaly classes 
ANOMALY_CLASSES = ['knife', 'scissors', 'fire', 'fight', 'Suspecious_object']

# Add person class filter
def filter_detections(detections):
    return detections[detections['name'] != 'person']
    # return detections

# Initialize video capture (webcam or 5G camera RTSP URL)
cap = cv2.VideoCapture(0)

# RTSP_URL = "rtsp://admin:admin123@192.168.128.10:554/avstream/channel=1/stream=1.sdp"
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

            # Save anomaly to DB instead of local file
            save_anomaly_to_db(row['name'], row['confidence'], frame)

    # Display frame with detection boxes
    annotated_frame = results.render()[0]
    cv2.imshow("5G Surveillance Feed", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cursor.close()
conn.close()
cv2.destroyAllWindows()
