AI-Driven Predictive Maintenance for CNC Machines using Cloud and IoT

📌 Project Overview

This project presents an AI-driven predictive maintenance system for CNC machines using IoT sensors, Machine Learning, and AWS Cloud services.
The system continuously monitors machine parameters such as vibration, temperature, voltage, current, and RPM and predicts machine health in real time.

The goal is to detect anomalies, predict machine failure, and estimate Remaining Useful Life (RUL) to reduce downtime and maintenance costs in industrial environments.

---

🧠 Key Features

- Real-time machine monitoring using IoT sensors
- Anomaly detection from vibration signals
- Failure prediction using machine learning
- Remaining Useful Life (RUL) estimation
- Cloud-based data processing
- Real-time visualization through a dashboard

---

⚙️ Technologies Used

- Python
- Machine Learning
- IoT (ESP32 + Sensors)
- Streamlit Dashboard
- Cloud Computing

---

☁️ AWS Services Used

- AWS IoT Core – receives real-time sensor data from ESP32 devices
- AWS Lambda – processes incoming data and runs ML inference
- Amazon DynamoDB – stores sensor readings and prediction results
- Amazon API Gateway – exposes REST APIs for data retrieval
- Amazon EC2 – hosts the Streamlit dashboard

---

🔄 System Architecture

IoT Sensors (Vibration, Temperature, Current, Voltage)
⬇
ESP32 Microcontroller
⬇
AWS IoT Core
⬇
AWS Lambda (ML Prediction Engine)
⬇
Amazon DynamoDB (Data Storage)
⬇
Amazon API Gateway
⬇
Streamlit Dashboard (Hosted on EC2)

---

🧠 Machine Learning Models

Model| Purpose
Isolation Forest| Anomaly Detection
Decision Tree / Random Forest| Failure Prediction
Random Forest Regressor| Remaining Useful Life (RUL) Prediction

---

📊 Datasets Used

CMAPSS Dataset (NASA)

Used for training the Remaining Useful Life prediction model.

AI4I Predictive Maintenance Dataset

Used for failure prediction modeling based on industrial machine parameters.

---

📈 Parameters Monitored

- Vibration
- Temperature
- Current
- Voltage
- Rotational Speed (RPM)

These parameters help determine machine health and degradation patterns.

---

📊 Dashboard Features

The Streamlit dashboard displays:

- Real-time sensor readings
- Machine health status
- Failure probability
- Remaining Useful Life estimation
- Anomaly detection alerts

---

📂 Project Structure

AI-Driven-Predictive-Maintenance-CNC
│
├── architecture
│   └── system_architecture.png
│
├── lambda
│   └── lambda_function.py
│
├── streamlit_app
│   └── app.py
│
├── iot_device
│   └── esp32_sensor_code.ino
│
├── models
│   ├── rul_model.pkl
│   ├── failure_model.pkl
│   └── anomaly_model.pkl
│
├── requirements.txt
├── README.md
└── LICENSE

---

🚀 How to Run the Dashboard

Install dependencies:

pip install -r requirements.txt

Run the Streamlit dashboard:

streamlit run app.py

Access the dashboard in your browser:

http://localhost:8501

---

🔮 Future Improvements

- Integration with real industrial CNC machines
- Edge AI deployment for faster predictions
- Advanced deep learning models
- Mobile monitoring application
- Industrial alert systems

---

👨‍💻 Author

Sabarish
AI | Cloud | IoT Enthusiast

---

⭐ Acknowledgment

This project demonstrates how Artificial Intelligence, IoT, and Cloud Computing can work together to enable smart manufacturing and predictive maintenance systems.
