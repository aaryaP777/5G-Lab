# Smart Surveillance using 5G Technology

## 🔹 Problem Statement
Traditional surveillance systems face challenges such as:
- High latency in live video streaming.
- Limited ability to process data in real time.
- Inefficient anomaly detection requiring manual monitoring.
- High bandwidth consumption when transmitting raw video continuously.
- Security concerns with unencrypted or poorly managed camera feeds.

👉 To address these challenges, we propose **Smart Surveillance powered by 5G**.  
Our system leverages **edge computing, AI/ML-based anomaly detection, and secure data storage** to detect threats in real time and notify administrators immediately — transmitting only anomalous frames instead of continuous video.

---

## 🔹 Key Technological Aspects
1. **5G Integration**
   - High bandwidth and ultra-low latency.
   - Supports real-time anomaly detection and rapid alerts.

2. **Edge Computing**
   - Video frames are processed locally (on the camera or nearby edge server).
   - Reduces cloud dependency and improves response time.

3. **AI/ML Models**
   - Uses **YOLOv5 (PyTorch)** for object detection.
   - Detects anomalies such as weapons, fire, suspicious objects, and fights.
   - Configurable anomaly classes (`ANOMALY_CLASSES`).

4. **Database Integration**
   - **PostgreSQL** is used to log anomalies.
   - Stores anomaly name, confidence score, timestamp, and frame (as binary data).
   - Can be extended to store file paths or integrate with dashboards.

5. **Cybersecurity**
   - Secure storage of database credentials using **.env** files.
   - Future scope: add encryption for data transmission (TLS/SSL).

6. **Visualization & Alerts**
   - Annotated video feed shows real-time detections.
   - Console alerts highlight anomalies with timestamps.
   - Frames containing anomalies are stored in PostgreSQL for auditing.

---

## 🔹 Project Workflow
1. **5G-enabled cameras** capture live footage.
2. **Edge computing (local inference with YOLOv5)** processes video in real time.
3. **AI/ML anomaly detection** identifies threats instantly.
4. **System triggers alerts** and saves anomalies into PostgreSQL.
5. **Cloud/DB storage** preserves frames for audit and retraining.
6. **Admins** review anomalies without needing to sift through hours of video.

---

## 🔹 Tech Stack
- **Programming Language**: Python
- **Deep Learning Framework**: PyTorch
- **Object Detection Model**: YOLOv5 (Ultralytics)
- **Database**: PostgreSQL 17
- **Environment Management**: dotenv
- **Libraries**: OpenCV, psycopg2, requests

---

## 🔹 Installation & Setup

### 1️⃣ Clone Repository
```bash
git clone https://github.com/your-username/5g-smart-surveillance.git
cd 5g-smart-surveillance
