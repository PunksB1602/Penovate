import serial
import json
import numpy as np
from scipy.signal import butter, filtfilt
import tkinter as tk
from tkinter import filedialog
from threading import Thread
import torch
import torch.nn as nn

# Model definition
class CNN_BiLSTM(nn.Module):
    def __init__(self, num_features: int, num_classes: int, hidden_size: int = 128, dropout: float = 0.3):
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

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)  # (B, F, T)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        lengths2 = torch.clamp((lengths + 1) // 2, min=1, max=x.shape[2])
        x = x.permute(0, 2, 1)  # (B, T', 128)
        x, _ = self.lstm(x)
        idx = (lengths2 - 1).view(-1, 1, 1).expand(-1, 1, x.size(2))
        x = x.gather(dim=1, index=idx).squeeze(1)
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)

# GUI Predictor class
class IMUPredictorGUI:
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
        self.num_classes = len(self.id2label)
        self.model = CNN_BiLSTM(num_features=18, num_classes=self.num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.serial = serial.Serial(port, baud_rate)
        self.filter_order = 2
        self.cutoff_freq = 20
        self.sampling_rate = 100
        self.fixed_seq_len = fixed_seq_len
        self.root = tk.Tk()
        self.root.title("IMU Character Predictor (PyTorch)")
        self.root.configure(bg="white")
        self.text_display = tk.Text(
            self.root,
            font=("Helvetica", 60),
            bg="white",
            fg="black",
            wrap=tk.WORD,
            state=tk.DISABLED,
        )
        self.text_display.pack(expand=True, fill="both")
        button_frame = tk.Frame(self.root, bg="white")
        button_frame.pack(fill="x", pady=10)
        clear_button = tk.Button(
            button_frame,
            text="Clear",
            font=("Helvetica", 20),
            command=self.clear_text,
        )
        clear_button.pack(side=tk.LEFT, padx=20)
        save_button = tk.Button(
            button_frame,
            text="Save",
            font=("Helvetica", 20),
            command=self.save_text,
        )
        save_button.pack(side=tk.RIGHT, padx=20)
        self.running = True
        self.thread = Thread(target=self.collect_and_predict, daemon=True)
        self.thread.start()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def butter_lowpass(self):
        nyquist = 0.5 * self.sampling_rate
        normalized_cutoff_freq = self.cutoff_freq / nyquist
        b, a = butter(self.filter_order, normalized_cutoff_freq, btype="low", analog=False)
        return b, a

    def preprocess_sequence(self, sequence):
        if not sequence:
            return None
        data = np.asarray(sequence, dtype=np.float32)
        b, a = self.butter_lowpass()
        for i in range(data.shape[1]):
            data[:, i] = filtfilt(b, a, data[:, i])
        imu1 = data[:, :6]
        imu2 = data[:, 6:]
        rel = imu1 - imu2
        data18 = np.concatenate([imu1, imu2, rel], axis=1)
        mean = data18.mean(axis=0)
        std = data18.std(axis=0)
        std[std == 0] = 1.0
        data18 = (data18 - mean) / std
        return data18

    def pad_or_trim(self, x18):
        T = x18.shape[0]
        if T >= self.fixed_seq_len:
            return x18[: self.fixed_seq_len], self.fixed_seq_len
        pad = np.zeros((self.fixed_seq_len - T, x18.shape[1]), dtype=np.float32)
        return np.vstack([x18, pad]), T

    def append_prediction(self, text):
        self.text_display.config(state=tk.NORMAL)
        self.text_display.insert(tk.END, text)
        self.text_display.config(state=tk.DISABLED)

    def clear_text(self):
        self.text_display.config(state=tk.NORMAL)
        self.text_display.delete(1.0, tk.END)
        self.text_display.config(state=tk.DISABLED)

    def save_text(self):
        text = self.text_display.get(1.0, tk.END).strip()
        if text:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt")],
            )
            if file_path:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(text)

    def collect_and_predict(self):
        sequence = []
        recording = False
        while self.running:
            if self.serial.in_waiting:
                line = self.serial.readline().decode("utf-8", errors="ignore").strip()
                if line == "START":
                    recording = True
                    sequence = []
                elif line == "END":
                    if recording and len(sequence) > 0:
                        x = self.preprocess_sequence(sequence)
                        if x is not None:
                            x, true_len = self.pad_or_trim(x)
                            x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)
                            lengths = torch.tensor([min(true_len, self.fixed_seq_len)], dtype=torch.long).to(self.device)
                            with torch.no_grad():
                                logits = self.model(x_t, lengths)
                                probs = torch.softmax(logits, dim=1)[0]
                                pred_idx = int(torch.argmax(probs))
                                pred_char = self.id2label[pred_idx].lower()
                            self.root.after(0, self.append_prediction, pred_char)
                    recording = False
                elif recording:
                    try:
                        data = [float(x) for x in line.split(",")]
                        if len(data) == 12:
                            sequence.append(data)
                    except ValueError:
                        pass

    def on_close(self):
        self.running = False
        try:
            self.serial.close()
        except Exception:
            pass
        self.root.destroy()

    def run(self):
        self.root.mainloop()

def main():
    MODEL_PATH = "best.pth"
    LABEL_MAP_PATH = "label_map.json"
    PORT = "COM6"
    app = IMUPredictorGUI(
        model_path=MODEL_PATH,
        label_map_path=LABEL_MAP_PATH,
        port=PORT,
    )
    app.run()

if __name__ == "__main__":
    main()
