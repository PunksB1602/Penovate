# Penovate: A Sensor-Equipped Pen for Digital Handwriting Recognition

**Penovate** is a smart pen system that captures handwriting on ordinary paper and converts it into digital text in real time.  
It integrates **motion and pressure sensors**, a **microcontroller with Bluetooth**, and a **CNN–BiLSTM deep learning model** to recognize handwritten characters from sensor streams.

---

## 📌 Background & Motivation

Digital handwriting recognition has traditionally relied on **touchscreens, graphics tablets, or optical scanning (OCR)**.  
While effective, these approaches require specialized surfaces or post-processing.  

Penovate addresses this gap by enabling **direct handwriting capture on any surface using a pen-shaped device**.  
The system combines **low-cost IMUs + pressure sensing** with a **neural sequence model**, making it portable, low-power, and extensible.

---

## 🖼 System Overview

**Architecture Workflow**:

1. **Hardware Layer**  
   - Two MPU-6050 IMUs record 6-DOF motion/orientation.  
   - An FSR detects pen–paper contact.  
   - Arduino Nano collects and transmits data via HC-05 Bluetooth.  

2. **Preprocessing Layer**  
   - Noise filtering (Butterworth low-pass).  
   - Stroke segmentation using pressure thresholding.  
   - Normalization & zero-padding for fixed-length input.  

3. **Recognition Layer**  
   - CNN extracts local spatial features.  
   - BiLSTM models temporal dependencies.  
   - Softmax classifier outputs characters (A–Z).  

4. **Application Layer**  
   - Real-time display of predicted characters in console or GUI.  
   - Data storage in structured datasets (JSON/CSV).  

---

## 🔧 Hardware Design

- **Arduino Nano (ATmega328p)** – microcontroller for acquisition  
- **MPU-6050 IMU × 2** – motion/orientation capture  
- **FSR sensor** – pressure-based stroke detection  
- **HC-05 Bluetooth module** – wireless communication  
- **Li-ion battery (7.4V, 2S)** – portable power supply  

📂 See [`hardware/`](hardware) for:
- Schematics & wiring tables  
- AD0/I²C addressing clarification (0x68 for AD0=GND, 0x69 for AD0=VCC)  
- Power design notes  

---

## 📊 Dataset & Preprocessing

- **Collection protocol**  
  - Recorded at fixed sampling rate (100 Hz)  
  - Multiple writers contributed samples for all 26 letters (A–Z)  
  - Stroke segmentation triggered by FSR pressure  

- **Preprocessing steps**  
  - Low-pass Butterworth filter for noise removal  
  - Relative motion calculation between dual IMUs  
  - Sequence normalization and zero-padding  

- **Dataset statistics**  
  - ~130 samples per character per writer  
  - Stored as JSON sequences (accelerometer, gyroscope, pressure)  

📂 Available in [`data/`](data).

---

## 🤖 Model Architecture

- **CNN layers** – extract spatial features from motion sequences  
- **BiLSTM layers** – capture temporal handwriting patterns  
- **Fully Connected + Softmax** – classify 26 English letters  

**Training setup**:  
- Optimizer: Adam, lr=0.001, decay every 10 epochs  
- Loss: Categorical Cross-Entropy  
- Epochs: 30  
- Batch size: 32  
- Framework: PyTorch  

📂 Implementation in [`models/`](models).  
📓 Jupyter notebooks in [`notebooks/`](notebooks).  

---

## 📈 Results

- **Character-level accuracy**: 93.2% (with batch normalization)  
- **Without batch normalization**: 78.7%  
- **Metrics reported**: Accuracy, Precision, Recall, F1-score  

**Key Observations**:  
- CNN–BiLSTM significantly outperformed CNN-only and LSTM-only baselines  
- Batch normalization stabilized training and improved generalization  
- Errors mainly occurred on visually similar characters (e.g., M vs N, C vs G)  

📂 See [`results/`](results) for plots, training curves, and confusion matrices.

---

## 📂 Repository Structure

