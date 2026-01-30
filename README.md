# Penovate: Real-Time Handwriting Recognition with a Sensor-Equipped Pen

Penovate is an open-source hardware–software system for capturing handwriting on ordinary paper using a custom sensor-equipped pen and converting it into digital text using a deep learning model (CNN–BiLSTM). The project covers the full stack: embedded hardware, firmware, data processing, and neural network recognition.

**Key Features:**
- Low-cost, portable pen device with IMU and FSR sensors
- Real-time data acquisition and Bluetooth transmission
- Signal processing and segmentation pipeline
- Deep learning model for character recognition (a–z)
- Reproducible experiments and results

This README provides a comprehensive overview, including hardware details, data pipeline, model architecture, training setup, results, and usage instructions.

---

## Project Team
- **Pankaj Bhatt (THA078BEI025)**
- **Pratik Pokharel (THA078BEI027)**
- **Subham Gautam (THA078BEI042)**

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

## Project Folder Structure

```
Penovate_Minor_Project/
├── data/                  # Processed and raw sensor data
│   └── processed_imu/     # Final .npy and .json files for training
├── results/               # Model checkpoints, logs, and plots
├── model.ipynb            # Main model development notebook
├── hybrid_CNN_BiLSTM.ipynb# Full experiment notebook
├── combine_json.py        # Data conversion scripts
├── predict.py, predict_gui.py # Inference scripts
├── README.md, LICENSE     # Documentation and license
└── ...
```

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

**Data Preprocessing Steps:**
- Raw sensor streams are filtered (Butterworth low-pass)
- Segmentation using FSR pressure threshold
- Normalization to unit range
- Zero-padding to fixed sequence length
- Label encoding and mapping (see `label_map.json`)

---

## 3. Dataset

  - Accelerometer (x, y, z)  
  - Gyroscope (x, y, z)  
  - Pressure (scalar)  

**Classes:** 26 lowercase English letters (a–z), single character recognition.
**Format:** Each character's samples are stored in separate JSON files (e.g., `a_lower.json`, `B_upper.json`) in the `imu_dataset` directory. Each file contains a list of preprocessed sensor sequences for that character.
**Signals recorded:**
   - Accelerometer (x, y, z) from IMU 1 and IMU 2
   - Gyroscope (x, y, z) from IMU 1 and IMU 2
   - Pressure (scalar)
**Sampling frequency:** 100 Hz
**Samples:** ~130 per class, single writer

**Data Collection & Preprocessing Pipeline:**
- Data is collected via a serial connection from the pen hardware.
- Each sample is a time-series sequence, recorded between `START` and `END` signals.
- Preprocessing steps for each sequence:
   1. **Low-pass Butterworth filtering** (order 2, cutoff 20 Hz) is applied to all sensor channels.
   2. **Relative motion** is computed between the two IMUs (imu1 - imu2), and concatenated with the original IMU data.
   3. **Normalization**: Each channel is normalized (zero mean, unit variance) per sequence.
   4. The processed sequence is appended to the character's dataset.
- Data is saved in files named `{char}_lower.json` or `{char}_upper.json` (for lowercase/uppercase), containing lists of sequences.
- The dataset is later converted to `.npy` format for model training.

**Note:** The current model and dataset are designed for single lowercase character recognition (a–z). Recognition of continuous words or uppercase characters is not supported in this version.

**Data Example:**
Each character sample is stored as a JSON file with synchronized sensor readings and label. Data is converted to `.npy` arrays for efficient training.

---

## 4. Model

### 4.1 Architecture

- **Input**: Sequence of 18 features (sensor channels: acc, gyro, pressure, etc.)  
- **CNN layers**:  
  - 1D convolutions extract local spatial/temporal features.  
- **BiLSTM layers**:  
  - Capture sequential handwriting dynamics in both forward and backward directions.  
- **Fully Connected Layer + Softmax**:  
  - Outputs class probabilities for 26 characters.  

**Model Code Reference:**
See `hybrid_CNN_BiLSTM.ipynb` and `model.ipynb` for full PyTorch implementation, including:
- Model class: `CNN_BiLSTM`
- Training loop: `Trainer` class
- Data loading: `make_loaders`, `load_raw_splits`

### 4.2 Training Setup

**Data directory:** `data/processed_imu`  
**Output directory:** `results/exp_{bn/no_bn}_bs{batch_size}_seed{seed}_{drop}`  
**Model:** Hybrid CNN-BiLSTM (2 Conv1D layers, 2-layer BiLSTM, optional BatchNorm, Dropout)  
**Input features:** 18 (sensor channels)  
**Batch size:** 32  
**Epochs:** 30  
**Hidden size:** 128 (LSTM)  
**Dropout:** 0.5  
**Batch normalization:** enabled/disabled (experimented both)  
**Optimizer:** AdamW (`lr=1e-4`, `weight_decay=1e-3`)  
**Learning rate scheduler:** StepLR (`step_size=8`, `gamma=0.7`)  
**Early stopping patience:** 7 epochs (on macro-F1)  
**Random seeds:** 42, 123, 7 (for reproducibility)  
**Top-k accuracy:** k=3 (optional)  
**Deterministic training:** True (for reproducibility)  
**Framework:** PyTorch  

---

## 5. Experiments and Results

### Latest Experiments (Batch Size 32, Dropout 0.5, Seeded)

| Experiment | BatchNorm | Test Loss | Test Accuracy | Macro F1 |
|-----------|-----------|-----------|--------------|----------|
| exp_bn_bs32_seed42_drop05     | True  | 0.0608 | 0.9862 | 0.9863 |
| exp_no_bn_bs32_seed42_drop05  | False | 0.1118 | 0.9803 | 0.9803 |
| exp_bn_bs32_seed123_drop05    | True  | 0.0652 | 0.9842 | 0.9841 |
| exp_no_bn_bs32_seed123_drop05 | False | 0.1228 | 0.9822 | 0.9824 |
| exp_bn_bs32_seed7_drop05      | True  | 0.0670 | 0.9862 | 0.9863 |
| exp_no_bn_bs32_seed7_drop05   | False | 0.1049 | 0.9842 | 0.9843 |

#### Metrics

- Accuracy, macro F1-score, and loss are reported for each experiment.
- Confusion matrices for all 26 classes are available in the results folder.

#### Observations


- **Batch normalization** consistently improved both accuracy and macro F1-score across all random seeds.
- The **best test accuracy achieved** was 98.62% (with batch normalization).
- Most misclassifications occurred between visually or motion-similar letters (e.g., M vs N, C vs G).
- The model is **highly reliable for isolated character recognition**.
- Recognition of continuous words or sentences remains a challenge and is a direction for future work.
- Training was stable and reproducible due to deterministic settings and fixed seeds.
- The data pipeline and preprocessing steps (filtering, segmentation, normalization) were crucial for robust model performance.

---

## 6. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/PunksB1602/Penovate_Minor_Project.git
cd Penovate_Minor_Project
pip install -r requirements.txt
```

---

## 7. Usage

### Training

1. Prepare your data in `data/processed_imu/` (see dataset section).
2. Run the main notebook or script:
   - `python model.ipynb` (or run all cells in Jupyter)
   - Or use the provided training functions in your own script.

### Inference

Use `predict.py` or `predict_gui.py` to run inference on new sensor data. See script comments for usage.

---

## 8. Contribution Guidelines

Contributions are welcome! Please open issues or pull requests for bug fixes, improvements, or new features. For major changes, discuss with the maintainers first.

---

## 9. License

This project is licensed under the MIT License. See LICENSE for details.

---

## 10. Contact & Acknowledgments

For any inquiries or feedback, please reach out to:
- **Pankaj Bhatt**: pbecie16@gmail.com
- **Pratik Pokharel**: pratikpokhrel14@gmail.com
- **Subham Gautam**: gautamsubham65@gmail.com

Special thanks to our advisors and all open-source contributors whose tools and libraries made this project possible.


