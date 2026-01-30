import serial
import csv
import os
import json
import numpy as np
from datetime import datetime
from scipy.signal import butter, filtfilt

class IMUDataCollector:
    def __init__(self, port='COM13', baud_rate=115200):
        self.serial = serial.Serial(port, baud_rate)
        self.dataset = {}
        self.initial_samples = {}
        self.data_dir = 'imu_dataset'
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.load_existing_dataset()
        self.filter_order = 2
        self.cutoff_freq = 20  # Hz
        self.sampling_rate = 100  # Hz
        
    def butter_lowpass(self):
        # Create a lowpass Butterworth filter
        nyquist = 0.5 * self.sampling_rate
        normalized_cutoff_freq = self.cutoff_freq / nyquist
        b, a = butter(self.filter_order, normalized_cutoff_freq, btype='low', analog=False)
        return b, a
        
    def apply_lowpass_filter(self, data):
        # Apply lowpass filter to sensor data
        b, a = self.butter_lowpass()
        data_array = np.array(data)
        filtered_data = np.zeros_like(data_array)
        for i in range(data_array.shape[1]):
            filtered_data[:, i] = filtfilt(b, a, data_array[:, i])
        return filtered_data.tolist()
    
    def calculate_relative_motion(self, data):
        # Calculate relative motion between the two IMUs
        data_array = np.array(data)
        imu1_data = data_array[:, :6]
        imu2_data = data_array[:, 6:]
        rel_motion = imu1_data - imu2_data
        return np.concatenate([imu1_data, imu2_data, rel_motion], axis=1).tolist()
    
    def normalize_data(self, data):
        # Normalize sensor data per axis
        data_array = np.array(data)
        mean = np.mean(data_array, axis=0)
        std = np.std(data_array, axis=0)
        std = np.where(std == 0, 1, std)
        normalized_data = (data_array - mean) / std
        return normalized_data.tolist()
    
    def preprocess_sequence(self, sequence):
        # Apply all preprocessing steps to a sequence
        if not sequence:
            return None
        filtered_data = self.apply_lowpass_filter(sequence)
        rel_motion_data = self.calculate_relative_motion(filtered_data)
        normalized_data = self.normalize_data(rel_motion_data)
        return normalized_data
    
    def load_existing_dataset(self):
        # Load existing dataset from individual character files, handling _upper/_lower
        self.dataset = {}
        self.initial_samples = {}
        if os.path.exists(self.data_dir):
            files = [f for f in os.listdir(self.data_dir) if f.endswith('.json')]
            for file in files:
                if "_upper" in file:
                    character = file.replace("_upper.json", "")
                elif "_lower" in file:
                    character = file.replace("_lower.json", "")
                else:
                    character = file[:-5]
                filepath = os.path.join(self.data_dir, file)
                try:
                    with open(filepath, 'r') as f:
                        self.dataset[character] = json.load(f)
                        self.initial_samples[character] = len(self.dataset[character])
                    print(f"Loaded existing data for character '{character}' from {filepath}")
                except Exception as e:
                    print(f"Error loading data for character '{character}': {e}")
                    self.dataset[character] = []
                    self.initial_samples[character] = 0
        self.show_dataset_stats()
    
    def show_dataset_stats(self):
        # Show statistics about the current dataset
        if not self.dataset:
            print("Dataset is empty")
            return
        print("\nCurrent Dataset Statistics:")
        print("---------------------------")
        total_samples = sum(len(samples) for samples in self.dataset.values())
        print(f"Total samples: {total_samples}")
        print("\nSamples per character:")
        for char, samples in sorted(self.dataset.items()):
            print(f"'{char}': {len(samples)} samples")
    
    def collect_character(self, character, num_samples):
        # Collect multiple samples for a single character
        if character not in self.dataset:
            self.dataset[character] = []
            self.initial_samples[character] = 0
        print(f"\nCollecting data for character: {character}")
        print(f"Please write the character {num_samples} times")
        existing_samples = len(self.dataset[character])
        print(f"Currently have {existing_samples} samples for '{character}'")
        samples_collected = 0
        while samples_collected < num_samples:
            print(f"\nSample {samples_collected + 1}/{num_samples}")
            input("Press Enter when ready to write...")
            sequence = self.collect_single_sample()
            if sequence:
                self.dataset[character].append(sequence)
                samples_collected += 1
                print("Sample recorded successfully!")
                print(f"Sequence length: {len(sequence)} timesteps")
            else:
                print("Error recording sample, please try again")
            if samples_collected < num_samples:
                retry = input("Continue collecting? (y/n): ").lower()
                if retry != 'y':
                    break
    
    def collect_single_sample(self):
        # Collect one sample of writing a character with preprocessing
        sequence = []
        recording = False
        while True:
            if self.serial.in_waiting:
                line = self.serial.readline().decode('utf-8').strip()
                if line == "START":
                    recording = True
                    sequence = []
                elif line == "END":
                    if recording and len(sequence) > 0:
                        preprocessed_sequence = self.preprocess_sequence(sequence)
                        return preprocessed_sequence
                    return None
                elif recording:
                    try:
                        data = [float(x) for x in line.split(',')]
                        if len(data) == 12:
                            sequence.append(data)
                    except ValueError:
                        continue
    
    def save_dataset(self):
        # Save the collected dataset to JSON files, safely handling Windows filenames
        for character, sequences in self.dataset.items():
            if character.isupper():
                filename = f"{character}_upper.json"
            elif character.islower():
                filename = f"{character}_lower.json"
            else:
                filename = f"{character}.json"
            filepath = os.path.join(self.data_dir, filename)
            try:
                initial_count = self.initial_samples.get(character, 0)
                new_samples = sequences[initial_count:]
                existing_sequences = []
                if os.path.exists(filepath):
                    with open(filepath, 'r') as f:
                        existing_sequences = json.load(f)
                all_sequences = existing_sequences + new_samples
                with open(filepath, 'w') as f:
                    json.dump(all_sequences, f, indent=2)
                print(f"Saved data for character '{character}' to {filename}")
                self.initial_samples[character] = len(all_sequences)
            except Exception as e:
                print(f"Error saving data for character '{character}': {e}")
    
    def close(self):
        # Close the serial connection
        self.serial.close()

def main():
    collector = IMUDataCollector()
    
    try:
        while True:
            print("\nIMU Data Collection Menu:")
            print("1. Collect data for a specific character")
            print("2. Show current dataset statistics")
            print("3. Save and exit")
            
            choice = input("\nEnter your choice (1-3): ")
            
            if choice == '1':
                char = input("Enter the character to collect: ")
                if len(char) != 1:
                    print("Please enter a single character")
                    continue
                    
                try:
                    num_samples = int(input("Enter number of samples to collect: "))
                    collector.collect_character(char, num_samples)
                    collector.save_dataset()  # Auto-save after collection
                except ValueError:
                    print("Please enter a valid number")
                    
            elif choice == '2':
                collector.show_dataset_stats()
                
            elif choice == '3':
                break
                
    except KeyboardInterrupt:
        print("\nData collection interrupted by user")
    finally:
        collector.save_dataset()
        collector.close()
        print("\nData collection completed")

if __name__ == "__main__":
    main()