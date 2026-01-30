import serial
import json
import numpy as np
from scipy.signal import butter, filtfilt
import torch
import torch.nn as nn


# Model definition
class CNN_BiLSTM(nn.Module):
    def __init__(self, num_features, num_classes, hidden_size=128, dropout=0.3):
        super().__init__()
        self.num_features = num_features
        self.conv1 = nn.Conv1d(num_features, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x, lengths):
        x = x.permute(0, 2, 1)  # (B, F, T)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        lengths = torch.clamp((lengths + 1) // 2, min=1, max=x.shape[2])
        x = x.permute(0, 2, 1)  # (B, T', 128)
        x, _ = self.lstm(x)
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, x.size(2))
        x = x.gather(1, idx).squeeze(1)
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)

# Predictor class
class IMUPredictor:
    def __init__(
        self,
        model_path="best.pth",
        label_map_path="label_map.json",
        port="COM6",
        baud_rate=115200,
        fixed_seq_len=128,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open(label_map_path, "r") as f:
            lm = json.load(f)
        self.id2label = {int(k): v for k, v in lm["id2label"].items()}
        num_classes = len(self.id2label)
        self.model = CNN_BiLSTM(num_features=18, num_classes=num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.serial = serial.Serial(port, baud_rate)
        # Preprocessing params (must match data collection)
        self.filter_order = 2
        self.cutoff_freq = 20
        self.sampling_rate = 100
        self.fixed_seq_len = fixed_seq_len

    def butter_lowpass(self):
        nyq = 0.5 * self.sampling_rate
        b, a = butter(self.filter_order, self.cutoff_freq / nyq, btype="low")
        return b, a

    def preprocess_sequence(self, sequence):
        data = np.asarray(sequence, dtype=np.float32)
        # Low-pass filter
        b, a = self.butter_lowpass()
        for i in range(data.shape[1]):
            data[:, i] = filtfilt(b, a, data[:, i])
        imu1 = data[:, :6]
        imu2 = data[:, 6:]
        rel = imu1 - imu2
        data = np.concatenate([imu1, imu2, rel], axis=1)
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        std[std == 0] = 1
        data = (data - mean) / std
        return data

    def pad_or_trim(self, x):
        T = x.shape[0]
        if T >= self.fixed_seq_len:
            return x[:self.fixed_seq_len]
        pad = np.zeros((self.fixed_seq_len - T, x.shape[1]), dtype=np.float32)
        return np.vstack([x, pad])

    def run(self):
        print("IMU Predictor (PyTorch) started")
        print("Write a character...")
        sequence = []
        recording = False
        while True:
            if self.serial.in_waiting:
                line = self.serial.readline().decode().strip()
                if line == "START":
                    recording = True
                    sequence = []
                    print("Recording...")
                elif line == "END":
                    if recording and len(sequence) > 0:
                        x = self.preprocess_sequence(sequence)
                        x = self.pad_or_trim(x)
                        x = torch.tensor(x).unsqueeze(0).to(self.device)
                        lengths = torch.tensor([min(len(sequence), self.fixed_seq_len)]).to(self.device)
                        with torch.no_grad():
                            logits = self.model(x, lengths)
                            probs = torch.softmax(logits, dim=1)[0]
                            idx = int(torch.argmax(probs))
                            conf = float(probs[idx])
                        print(f"\nPredicted: {self.id2label[idx]}  (conf={conf:.2f})")
                        print("Ready...\n")
                    recording = False
                elif recording:
                    try:
                        vals = [float(v) for v in line.split(",")]
                        if len(vals) == 12:
                            sequence.append(vals)
                    except ValueError:
                        pass

if __name__ == "__main__":
    predictor = IMUPredictor(
        model_path="best.pth",
        label_map_path="label_map.json",
        port="COM6",
    )
    predictor.run()
