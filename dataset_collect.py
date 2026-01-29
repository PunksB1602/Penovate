"""
IMU Data Collector

Collects and preprocesses IMU data for handwritten character recognition.
Each sequence is filtered, normalized, relative motion is calculated, and
data is saved in JSON files per character.
"""

import serial
import os
import json
import numpy as np
from scipy.signal import butter, filtfilt


class IMUDataCollector:
    def __init__(self, port='COM13', baud_rate=115200):
        # connect to the IMU device
        self.serial = serial.Serial(port, baud_rate)

        self.dataset = {}
        self.initial_samples = {}
        self.data_dir = 'imu_dataset'

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        # load any previous data if available
        self.load_existing_dataset()

        # filter configuration
        self.filter_order = 2
        self.cutoff_freq = 20
        self.sampling_rate = 100

    def butter_lowpass(self):
        """design a low-pass Butterworth filter"""
        nyquist = 0.5 * self.sampling_rate
        norm_cutoff = self.cutoff_freq / nyquist
        return butter(self.filter_order, norm_cutoff, btype='low')

    def apply_lowpass_filter(self, data):
        """apply filter to each sensor axis"""
        b, a = self.butter_lowpass()
        arr = np.array(data)
        filtered = np.zeros_like(arr)

        for i in range(arr.shape[1]):
            filtered[:, i] = filtfilt(b, a, arr[:, i])

        return filtered.tolist()

    def calculate_relative_motion(self, data):
        """
        subtract IMU2 from IMU1 and return
        concatenated [imu1, imu2, relative]
        """
        arr = np.array(data)
        imu1, imu2 = arr[:, :6], arr[:, 6:]
        rel = imu1 - imu2
        return np.concatenate([imu1, imu2, rel], axis=1).tolist()

    def normalize_data(self, data):
        """normalize each axis (zero mean, unit variance)"""
        arr = np.array(data)
        mean, std = np.mean(arr, axis=0), np.std(arr, axis=0)
        std = np.where(std == 0, 1, std)
        return ((arr - mean) / std).tolist()

    def preprocess_sequence(self, sequence):
        """filter -> relative motion -> normalize"""
        if not sequence:
            return None
        return self.normalize_data(
            self.calculate_relative_motion(
                self.apply_lowpass_filter(sequence)
            )
        )

    def load_existing_dataset(self):
        """load dataset from json files if they exist"""
        self.dataset, self.initial_samples = {}, {}

        if os.path.exists(self.data_dir):
            for file in os.listdir(self.data_dir):
                if not file.endswith('.json'):
                    continue
                char = file[:-5]
                path = os.path.join(self.data_dir, file)
                try:
                    with open(path, 'r') as f:
                        self.dataset[char] = json.load(f)
                        self.initial_samples[char] = len(self.dataset[char])
                    print(f"Loaded {len(self.dataset[char])} samples for '{char}'")
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    self.dataset[char] = []
                    self.initial_samples[char] = 0

        self.show_dataset_stats()

    def show_dataset_stats(self):
        if not self.dataset:
            print("Dataset is empty")
            return
        total = sum(len(v) for v in self.dataset.values())
        print(f"\nTotal samples: {total}")
        for char, samples in sorted(self.dataset.items()):
            print(f"{char}: {len(samples)}")

    def collect_character(self, character, num_samples):
        if character not in self.dataset:
            self.dataset[character] = []
            self.initial_samples[character] = 0

        print(f"\nCollecting {num_samples} samples for '{character}'")
        collected = 0

        while collected < num_samples:
            print(f"\nSample {collected + 1}/{num_samples}")
            input("Press Enter when ready...")

            seq = self.collect_single_sample()
            if seq:
                self.dataset[character].append(seq)
                collected += 1
                print(f"Recorded {len(seq)} timesteps")
            else:
                print("Recording failed, retry.")

            if collected < num_samples:
                if input("Continue? (y/n): ").lower() != 'y':
                    break

    def collect_single_sample(self):
        """record one sequence between START and END markers"""
        seq, recording = [], False

        while True:
            if self.serial.in_waiting:
                line = self.serial.readline().decode('utf-8').strip()

                if line == "START":
                    recording, seq = True, []
                elif line == "END":
                    return self.preprocess_sequence(seq) if recording and seq else None
                elif recording:
                    try:
                        data = [float(x) for x in line.split(',')]
                        if len(data) == 12:
                            seq.append(data)
                    except ValueError:
                        continue

    def save_dataset(self):
        for char, sequences in self.dataset.items():
            path = os.path.join(self.data_dir, f"{char}.json")
            try:
                initial = self.initial_samples.get(char, 0)
                new_samples = sequences[initial:]

                existing = []
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        existing = json.load(f)

                all_sequences = existing + new_samples
                with open(path, 'w') as f:
                    json.dump(all_sequences, f, indent=2)

                self.initial_samples[char] = len(all_sequences)
                print(f"Saved {len(new_samples)} new samples for '{char}'")
            except Exception as e:
                print(f"Error saving '{char}': {e}")

    def close(self):
        self.serial.close()


def main():
        collector = IMUDataCollector()
        try:
            while True:
                print("\n1. Collect data")
                print("2. Show stats")
                print("3. Save and exit")

                choice = input("Choice (1-3): ")
                if choice == '1':
                    char = input("Enter character: ").lower()
                    if len(char) != 1:
                        print("Enter a single character")
                        continue
                    try:
                        num = int(input("Number of samples: "))
                        collector.collect_character(char, num)
                        collector.save_dataset()
                    except ValueError:
                        print("Invalid number")
                elif choice == '2':
                    collector.show_dataset_stats()
                elif choice == '3':
                    break
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            collector.save_dataset()
            collector.close()
            print("Data collection completed")


if __name__ == "__main__":
    main()
