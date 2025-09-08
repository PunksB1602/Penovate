# Penovate: A Sensor-Equipped Pen for Writing on Paper with Digital Conversion using CNN-LSTM Model

This project implements a smart pen system that captures handwriting on paper and converts it into digital text in real time. The system integrates motion sensors, a microcontroller, and a CNN-LSTM deep learning model to recognize characters from pen movements.

---

## Features

- **Hardware Integration:**
  - **MPU-6050 IMU:** Captures pen motion, orientation, and tilt.
  - **Force Sensitive Resistor (FSR):** Detects writing pressure for stroke detection.
  - **Arduino Nano:** Microcontroller for data acquisition and preprocessing.
  - **HC-05 Bluetooth Module:** Transmits sensor data to the computer.
  - **Li-ion Battery:** Portable power source.

- **Data Collection & Preprocessing:**
  - Real-time reading of sensor data from IMU and FSR.
  - Noise filtering using low-pass Butterworth filter.
  - Calculation of relative motion between multiple sensors.
  - Normalization and padding of sequences for model input.

- **Machine Learning Model:**
  - **CNN:** Extracts spatial stroke features.
  - **BiLSTM:** Captures temporal patterns of handwriting.
  - **Softmax Classifier:** Outputs predicted characters.

- **Real-Time Prediction:**
  - Console or GUI-based prediction of handwritten characters.
  - Processes incoming sensor sequences and displays recognized text instantly.

- **Dataset Management:**
  - Stores character sequences in JSON files.
  - Combines individual character files into structured datasets for training.
  - Supports train-test split and label encoding for model training.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/PunksB1602/Penovate_Minor_Project.git
cd Penovate_Minor_Project
