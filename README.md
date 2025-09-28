# Penovate: A Sensor-Equipped Pen for Digital Handwriting Recognition

This repository contains the implementation of **Penovate**, a hardware–software system designed to capture handwriting on ordinary paper using motion and pressure sensors, and convert it into digital text using a **CNN–BiLSTM model**.

The project integrates **embedded hardware** for data acquisition, a **signal processing pipeline** for preprocessing, and a **deep learning model** for recognition. This README provides the technical details, experimental results, and instructions to reproduce the system.

---

## 1. Background and Motivation

Handwriting recognition has been studied extensively, with approaches ranging from **optical character recognition (OCR)** on scanned documents to **stylus-based digitizers** on tablets. However, most existing systems have limitations:

- OCR requires scanning or imaging, which is not real-time.  
- Stylus-based systems require specialized touchscreens or tablets.  
- Existing smart pens are often proprietary and expensive.

**Objective**: Develop an **open, low-cost, portable pen device** that enables handwriting capture on ordinary paper and converts it into digital characters in real time.  

**Key idea**: Use **inertial measurement units (IMUs)** to capture motion, an **FSR sensor** to detect strokes, and **deep sequence models** to recognize characters from the resulting time-series data.

---

## 2. System Overview

The Penovate system consists of four layers:

1. **Hardware Layer**  
   - A pen prototype equipped with sensors and Bluetooth module.  
   - Handles data acquisition during handwriting.  

2. **Firmware Layer**  
   - Arduino Nano firmware for synchronizing and transmitting IMU + FSR data.  

3. **Data Pipeline Layer**  
   - Preprocessing: filtering, segmentation, normalization.  
   - Converts raw sensor streams into fixed-length sequences.  

4. **Recognition Layer**  
   - A CNN–BiLSTM deep learning model trained to classify characters A–Z.  

---

### 2.1 Hardware Components

- **Arduino Nano (ATmega328p)** – microcontroller for acquisition.  
- **Two MPU-6050 IMUs** – capture accelerometer and gyroscope signals.  
- **Force-Sensitive Resistor (FSR)** – detects pen–paper contact and stroke boundaries.  
- **HC-05 Bluetooth module** – wireless transmission to host machine.  
- **Li-ion battery (2S, 7.4V)** – portable power source.  

---

### 2.2 Firmware Functionality

- Initializes I²C communication with two MPU-6050 sensors (addresses `0x68` and `0x69`).  
- Reads accelerometer and gyroscope data at fixed frequency (100 Hz).  
- Reads pressure data from FSR.  
- Formats sensor data into structured packets.  
- Streams packets over serial/Bluetooth to the host computer.  

---

### 2.3 Data Processing Pipeline

1. **Acquisition**: Sensor streams (accelerometer, gyroscope, pressure).  
2. **Filtering**: Low-pass Butterworth filter removes high-frequency noise.  
3. **Segmentation**: Pressure threshold from FSR marks stroke start/end.  
4. **Normalization**: Sensor values scaled to unit range.  
5. **Padding**: Sequences zero-padded to fixed length for batching.  

---

## 3. Dataset

- **Classes**: 26 uppercase English letters (A–Z).  
- **Format**: JSON (per character sequence) → converted to NumPy `.npy`.  
- **Signals recorded**:  
  - Accelerometer (x, y, z)  
  - Gyroscope (x, y, z)  
  - Pressure (scalar)  
- **Sampling frequency**: 100 Hz.  
- **Samples**: ~130 per class, multiple writers.  

---

## 4. Model

### 4.1 Architecture

- **Input**: Sequence of 7 features (acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, pressure).  
- **CNN layers**:  
  - 1D convolutions extract local spatial/temporal features.  
- **BiLSTM layers**:  
  - Capture sequential handwriting dynamics in both forward and backward directions.  
- **Fully Connected Layer + Softmax**:  
  - Outputs class probabilities for 26 characters.  

### 4.2 Training Setup

- Loss: Categorical cross-entropy.  
- Optimizer: Adam (lr=0.001, decay after 10 epochs).  
- Epochs: 30.  
- Batch size: 32.  
- Framework: PyTorch.  

---

## 5. Experiments and Results

Two main experiments were conducted:

- **Result 1**: Training with batch size 32, **without batch normalization**  
  - Accuracy: ~78.7%  
  - Issues: Slower convergence, overfitting on some classes.  

- **Result 2**: Training with batch size 32, **with batch normalization**  
  - Accuracy: ~93.2%  
  - Improvement: Faster convergence, better generalization.  

### Metrics

- Accuracy, precision, recall, F1-score.  
- Confusion matrices for all 26 classes.  

### Observations

- Batch normalization significantly improved stability and accuracy.  
- Most misclassifications occurred between visually or motion-similar letters (e.g., M vs N, C vs G).  
- Recognition works reliably for isolated characters, but continuous words/sentences remain challenging.  

---

## 6. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/PunksB1602/Penovate_Minor_Project.git
cd Penovate_Minor_Project
pip install -r requirements.txt
