import json
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split

DATA_DIR = Path("data/imu_dataset")
OUTPUT_DIR = Path("data/processed_imu")

NUM_FEATURES = 18

FIXED_SEQ_LEN = 184
AUTO_LEN_PERCENTILE = 95

MIN_SAMPLES_PER_CLASS = 5
RANDOM_STATE = 42

AUGMENT_DATA = True
AUGMENT_COPIES = 2
NOISE_LEVEL = 0.05
SCALE_RANGE = (0.8, 1.2)

def extract_label(stem: str) -> str:
    """Remove optional suffixes like _upper/_lower from filename stem."""
    if stem.endswith("_upper"):
        return stem[:-6]
    if stem.endswith("_lower"):
        return stem[:-6]
    return stem

def load_sequences():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Directory not found: {DATA_DIR}")

    json_files = sorted(DATA_DIR.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No .json files found in {DATA_DIR}")

    seqs, labels, lengths = [], [], []

    for fp in json_files:
        label = extract_label(fp.stem)

        try:
            with open(fp, "r") as f:
                data = json.load(f)

            for seq in data:
                arr = np.asarray(seq, dtype=np.float32)

                if arr.ndim != 2 or arr.shape[1] != NUM_FEATURES:
                    continue
                T = int(arr.shape[0])
                if T <= 0:
                    continue

                seqs.append(arr)
                labels.append(label)
                lengths.append(T)

        except Exception as e:
            print(f"[WARN] Failed reading {fp.name}: {e}")

    return seqs, np.asarray(labels, dtype=object), np.asarray(lengths, dtype=np.int32)

def filter_low_sample_classes(seqs, labels, lengths):
    counts = Counter(labels.tolist())

    if MIN_SAMPLES_PER_CLASS <= 1:
        return seqs, labels, lengths, []

    keep_idx = [i for i, y in enumerate(labels) if counts[y] >= MIN_SAMPLES_PER_CLASS]
    dropped = sorted([c for c, n in counts.items() if n < MIN_SAMPLES_PER_CLASS])

    if dropped:
        print(f"[WARN] Dropping classes with <{MIN_SAMPLES_PER_CLASS} samples: {dropped}")

    seqs_f = [seqs[i] for i in keep_idx]
    labels_f = labels[keep_idx]
    lengths_f = lengths[keep_idx]
    return seqs_f, labels_f, lengths_f, dropped

def choose_fixed_len(lengths: np.ndarray) -> int:
    if FIXED_SEQ_LEN is not None:
        return int(FIXED_SEQ_LEN)
    return int(np.percentile(lengths, AUTO_LEN_PERCENTILE))

def pad_truncate(seqs, lengths, fixed_len: int):
    X = np.zeros((len(seqs), fixed_len, NUM_FEATURES), dtype=np.float32)
    out_len = np.zeros((len(seqs),), dtype=np.int32)
    truncated = 0
    for i, (arr, T) in enumerate(zip(seqs, lengths)):
        if T >= fixed_len:
            X[i] = arr[:fixed_len]
            out_len[i] = fixed_len
            truncated += 1
        else:
            X[i, :T] = arr
            out_len[i] = T
    return X, out_len, truncated

def compute_norm_stats(X, lengths):
    feats = []
    for i in range(X.shape[0]):
        feats.append(X[i, :lengths[i]])
    feats = np.concatenate(feats, axis=0)
    mean = feats.mean(axis=0)
    std = feats.std(axis=0)
    std[std == 0] = 1.0
    return mean, std

def normalize_X(X, lengths, mean, std):
    Xn = X.copy()
    for i in range(X.shape[0]):
        T = lengths[i]
        Xn[i, :T] = (Xn[i, :T] - mean) / std
    return Xn

def encode_labels(labels: np.ndarray):
    classes = sorted(np.unique(labels).tolist())
    label2id = {c: i for i, c in enumerate(classes)}
    y = np.asarray([label2id[c] for c in labels], dtype=np.int64)
    return y, label2id

def split_data(X, y, lengths):
    X_temp, X_test, y_temp, y_test, len_temp, len_test = train_test_split(
        X, y, lengths, test_size=0.15, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val, len_train, len_val = train_test_split(
        X_temp, y_temp, len_temp, test_size=0.1765, stratify=y_temp, random_state=RANDOM_STATE
    )
    return (X_train, y_train, len_train), (X_val, y_val, len_val), (X_test, y_test, len_test)

def augment_training_data(X, y, lengths):
    print(f"\n[Complex Augmentation] Generating {AUGMENT_COPIES} augmented copies per sample...")
    X_aug_list = [X]
    y_aug_list = [y]
    len_aug_list = [lengths]
    N, T, F = X.shape
    for _ in range(AUGMENT_COPIES):
        X_new = X.copy()
        scales = np.random.uniform(SCALE_RANGE[0], SCALE_RANGE[1], size=(N, 1, 1))
        X_new = X_new * scales
        noise = np.random.normal(0, NOISE_LEVEL, size=(N, T, F))
        for i in range(N):
            valid_len = lengths[i]
            X_new[i, :valid_len, :] += noise[i, :valid_len, :]
        X_aug_list.append(X_new)
        y_aug_list.append(y)
        len_aug_list.append(lengths)
    X_final = np.concatenate(X_aug_list, axis=0)
    y_final = np.concatenate(y_aug_list, axis=0)
    len_final = np.concatenate(len_aug_list, axis=0)
    return X_final, y_final, len_final

def save_outputs(train_pack, val_pack, test_pack, label2id, stats, norm_stats=None):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train, len_train = train_pack
    X_val,   y_val,   len_val   = val_pack
    X_test,  y_test,  len_test  = test_pack

    np.save(OUTPUT_DIR / "X_train.npy", X_train)
    np.save(OUTPUT_DIR / "y_train.npy", y_train)
    np.save(OUTPUT_DIR / "len_train.npy", len_train)

    np.save(OUTPUT_DIR / "X_val.npy", X_val)
    np.save(OUTPUT_DIR / "y_val.npy", y_val)
    np.save(OUTPUT_DIR / "len_val.npy", len_val)

    np.save(OUTPUT_DIR / "X_test.npy", X_test)
    np.save(OUTPUT_DIR / "y_test.npy", y_test)
    np.save(OUTPUT_DIR / "len_test.npy", len_test)

    with open(OUTPUT_DIR / "label_map.json", "w") as f:
        json.dump(
            {
                "label2id": label2id,
                "id2label": {int(v): k for k, v in label2id.items()},
            },
            f,
            indent=2
        )

    with open(OUTPUT_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    if norm_stats is not None:
        with open(OUTPUT_DIR / "norm_stats.json", "w") as f:
            json.dump({"mean": norm_stats[0].tolist(), "std": norm_stats[1].tolist()}, f, indent=2)

    print("Saved to the directory:", OUTPUT_DIR.resolve())
    print(f"Fixed sequence length: {stats['fixed_seq_len']}")
    print(f"Total samples (after filtering & augmentation): {stats['num_samples']}")
    print(f"Total classes: {stats['num_classes']}")
    print(f"Truncated: {stats['truncated_count']} ({stats['truncated_pct']:.2f}%)")
    print("\nLength stats:")
    print(f"  mean={stats['length_mean']:.2f}, std={stats['length_std']:.2f}, "
          f"min={stats['length_min']}, p50={stats['length_p50']:.2f}, "
          f"p95={stats['length_p95']:.2f}, max={stats['length_max']}")

    print("\nSplit shapes:")
    print(f"  Train: X={X_train.shape}, y={y_train.shape}, len={len_train.shape}")
    print(f"  Val:   X={X_val.shape},   y={y_val.shape},   len={len_val.shape}")
    print(f"  Test:  X={X_test.shape},  y={y_test.shape},  len={len_test.shape}")

    print("\nClass counts (after filtering):")
    for k in sorted(stats["class_counts"].keys()):
        print(f"  {k}: {stats['class_counts'][k]}")

def main():
    print(f"Loading data from {DATA_DIR.resolve()} ...")
    seqs, labels, lengths = load_sequences()
    class_counts = Counter(labels.tolist())
    print(f"Loaded {len(seqs)} sequences.")
    print("Classes and sample counts:")
    for cls, count in sorted(class_counts.items()):
        print(f"  {cls}: {count}")
    if len(seqs) == 0:
        print("No valid sequences found.")
        return

    seqs, labels, lengths, dropped = filter_low_sample_classes(seqs, labels, lengths)

    if len(seqs) == 0:
        print("No sequences left after filtering.")
        return

    fixed_len = choose_fixed_len(lengths)

    X, out_len, truncated = pad_truncate(seqs, lengths, fixed_len)
    y, label2id = encode_labels(labels)

    train_pack, val_pack, test_pack = split_data(X, y, out_len)

    if AUGMENT_DATA:
        X_train, y_train, len_train = train_pack
        X_train, y_train, len_train = augment_training_data(X_train, y_train, len_train)
        train_pack = (X_train, y_train, len_train)

    X_train, y_train, len_train = train_pack
    X_val, y_val, len_val = val_pack
    X_test, y_test, len_test = test_pack

    mean, std = compute_norm_stats(X_train, len_train)
    X_train_norm = normalize_X(X_train, len_train, mean, std)
    X_val_norm = normalize_X(X_val, len_val, mean, std)
    X_test_norm = normalize_X(X_test, len_test, mean, std)

    train_pack = (X_train_norm, y_train, len_train)
    val_pack = (X_val_norm, y_val, len_val)
    test_pack = (X_test_norm, y_test, len_test)

    counts = Counter(y_train.tolist())
    str_counts = {label2id_rev: count for label2id_rev, count in counts.items()}
    
    stats = {
        "data_dir": str(DATA_DIR),
        "num_samples": int(len(X_train) + len(X_val) + len(X_test)),
        "num_classes": int(len(label2id)),
        "classes": sorted(label2id.keys()),
        "fixed_seq_len": int(fixed_len),
        "min_samples_per_class": int(MIN_SAMPLES_PER_CLASS),
        "dropped_classes": dropped,
        "length_mean": float(np.mean(lengths)),
        "length_std": float(np.std(lengths)),
        "length_min": int(np.min(lengths)),
        "length_p50": float(np.percentile(lengths, 50)),
        "length_p95": float(np.percentile(lengths, 95)),
        "length_max": int(np.max(lengths)),
        "truncated_count": int(truncated),
        "truncated_pct": float(truncated) / float(len(X)) * 100.0,
        "class_counts": {k: int(v) for k, v in counts.items()},
        "random_state": int(RANDOM_STATE),
        "split": {"train": 0.70, "val": 0.15, "test": 0.15},
        "augmentation": {
            "enabled": AUGMENT_DATA,
            "copies": AUGMENT_COPIES,
            "noise": NOISE_LEVEL,
            "scale": SCALE_RANGE
        }
    }

    print("\nSaving processed dataset...")
    save_outputs(train_pack, val_pack, test_pack, label2id, stats, norm_stats=(mean, std))

if __name__ == "__main__":
    main()