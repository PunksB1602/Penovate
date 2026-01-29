
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def prepare_training_data(json_file="combined_dataset.json", random_state=42, save_dir="./"):
    """
    Load IMU dataset from JSON, pad sequences, encode labels,
    split into train/val/test sets (70/15/15), and save to .npy files.
    """
    with open(json_file, "r") as f:
        dataset = json.load(f)


    # Extract sequences and labels
    sequences = [sample["sequence"] for sample in dataset["data"]]
    labels = [sample["character"] for sample in dataset["data"]]

    # Pad sequences to the same length
    max_length = max(len(seq) for seq in sequences)
    feature_dim = len(sequences[0][0]) if sequences and sequences[0] else 0
    X = np.array([
        seq + [[0.0] * feature_dim] * (max_length - len(seq))
        for seq in sequences
    ])

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)


    # Split: 70% train, 15% val, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
    )

    # Save splits
    np.save(f"{save_dir}/X_train.npy", X_train)
    np.save(f"{save_dir}/X_val.npy", X_val)
    np.save(f"{save_dir}/X_test.npy", X_test)
    np.save(f"{save_dir}/y_train.npy", y_train)
    np.save(f"{save_dir}/y_val.npy", y_val)
    np.save(f"{save_dir}/y_test.npy", y_test)
    np.save(f"{save_dir}/label_encoder.npy", label_encoder.classes_)

    # Print dataset info
    print("\nDataset Info")
    print(f"Features per timestep: {feature_dim}")
    print(f"Max sequence length: {max_length}")
    print(f"Classes: {label_encoder.classes_}")

    print("\nShapes")
    print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}, y_val: {y_val.shape}, y_test: {y_test.shape}")

    print("\nClass distribution")
    for idx, label in enumerate(label_encoder.classes_):
        train_count = np.sum(y_train == idx)
        val_count = np.sum(y_val == idx)
        test_count = np.sum(y_test == idx)
        print(f"{label}: train={train_count}, val={val_count}, test={test_count}")

    return X_train, X_val, X_test, y_train, y_val, y_test, label_encoder


if __name__ == "__main__":
    prepare_training_data()
