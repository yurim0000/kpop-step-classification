
from pathlib import Path
import json
import pandas as pd
import numpy as np

from features import FEATURE_SET, make_features, feature_meta, video_fps, resample_fps, load_root, TARGET_FPS

ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "data" / "raw_keypoints"
OUTPUT_DIR = ROOT / "data" / "dataset"

SEQ_LENGTH = 30
STRIDE = 15
VAL_RATIO = 0.2
SEED = 42
MAX_WINDOWS_PER_VIDEO = 20   # training only; None = keep all windows

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_raw_keypoints(path):
    if path.suffix == ".npy":
        # Processed keypoints: (T, 33, 3)
        motion = np.load(path)

        if motion.ndim == 3 and motion.shape[1:] == (33, 3):
            return motion

        raise ValueError(f"Invalid processed data shape: {motion.shape}")

    elif path.suffix == ".npz":
        # Raw keypoints: (T * 33, 6)
        data = np.load(path, allow_pickle=True)["data"]

        df = pd.DataFrame(
            data,
            columns=["frame", "landmark", "x", "y", "z", "visibility"]
        )

        pts_seq = []

        for _, group in df.groupby("frame"):
            group = group.sort_values("landmark")
            pts = group[["x", "y", "z"]].values
            pts_seq.append(pts)

        return np.array(pts_seq)

def create_sequences(motion):
    """
    motion: (T, J, 3)  (J = number of selected joints, see features.py)
    return: (N, SEQ_LENGTH, J*3)
    """
    if motion.ndim != 3 or motion.shape[2] != 3:
        raise ValueError(f"Invalid motion shape: {motion.shape}")
    frame_dim = motion.shape[1] * 3

    if not np.isfinite(motion).all():
        raise ValueError("Motion contains NaN or Inf")

    sequences = []

    for start in range(0, len(motion) - SEQ_LENGTH + 1, STRIDE):
        clip = motion[start:start + SEQ_LENGTH]
        sequences.append(clip.reshape(SEQ_LENGTH, frame_dim))

    return sequences


MIRROR_PAIRS_PATH = ROOT / "data" / "mirror_pairs.json"   # written by p0_mirror_videos.py
MIRROR_PAIRS = (json.loads(MIRROR_PAIRS_PATH.read_text(encoding="utf-8"))
                if MIRROR_PAIRS_PATH.exists() else {})


def source_of(path):
    """'crab/crab_9' (mirror) -> 'crab/crab_1' ; any other video -> itself."""
    key = f"{path.parent.name}/{path.stem}"
    return MIRROR_PAIRS.get(key, key)


def split_video_paths(paths, rng):
    """
    Split by video, not by individual sequence (windows of one video never land in both sets).
    A video and its mirrored copy form one group, so they always land on the same side.
    """
    groups = {}
    for p in paths:
        groups.setdefault(source_of(p), []).append(p)
    keys = sorted(groups)
    rng.shuffle(keys)

    if len(keys) < 2:
        raise ValueError(
            "Each class needs at least two videos "
            "for train/validation splitting."
        )

    n_val = max(1, round(len(keys) * VAL_RATIO))
    n_val = min(n_val, len(keys) - 1)

    train = [p for k in keys[n_val:] for p in groups[k]]
    val = [p for k in keys[:n_val] for p in groups[k]]
    return train, val


def cap_windows(sequences, rng, k=None):
    """
    Class-imbalance fix: keep at most k windows of one TRAINING video (random subset, order kept),
    so a few long videos (e.g. a 48 s crab clip) do not dominate their class.
    Validation windows are never dropped.
    """
    k = MAX_WINDOWS_PER_VIDEO if k is None else k
    if k is None or len(sequences) <= k:
        return sequences
    keep = np.sort(rng.choice(len(sequences), k, replace=False))
    return [sequences[i] for i in keep]


def build_dataset():
    rng = np.random.default_rng(SEED)
    cap_rng = np.random.default_rng(SEED + 1)   # separate stream: the split stays the same

    class_dirs = sorted(
        p for p in INPUT_DIR.iterdir() if p.is_dir()
    )

    if not class_dirs:
        raise RuntimeError(f"No class folders found: {INPUT_DIR}")

    labels = {
        class_dir.name: i
        for i, class_dir in enumerate(class_dirs)
    }

    X_train, y_train = [], []
    X_val, y_val = [], []

    for class_dir in class_dirs:
        label = labels[class_dir.name]
        # paths = sorted(class_dir.glob("*.npy"))
        paths = sorted(list(class_dir.glob("*.npz")) +
    list(class_dir.glob("*.npy")))

        # drop videos in which no pose was detected at all (empty npz)
        empty = [p for p in paths
                 if p.suffix == ".npz" and np.load(p, allow_pickle=True)["data"].size == 0]
        for p in empty:
            print(f"  Skipped (no pose detected): {class_dir.name}/{p.name}")
        paths = [p for p in paths if p not in empty]

        train_paths, val_paths = split_video_paths(paths, rng)

        print(f"\nClass: {class_dir.name}")
        print(f"  Train videos: {len(train_paths)}")
        print(f"  Val videos:   {len(val_paths)}  {[p.name for p in val_paths]}")

        for split, selected_paths in [
            ("train", train_paths),
            ("val", val_paths),
        ]:
            for path in selected_paths:
                motion = load_raw_keypoints(path)
                motion = resample_fps(motion, video_fps(path))  # same time scale for every video
                
                # TODO (Optional):
                # 1. Design custom input features from motion data
                #    (e.g., joint angles, distances, or velocities).
                #
                # 2. Generate labels directly from motion data
                #    instead of using folder names.

                # 1. -> joint selection + pelvis-centered coordinates (features.py)
                motion = make_features(motion, FEATURE_SET, root=load_root(path, motion))

                sequences = create_sequences(motion)

                if not sequences:
                    print(f"  Skipped (too short): {path.name}")
                    continue

                if split == "train":
                    sequences = cap_windows(sequences, cap_rng)
                    X_train.extend(sequences)
                    y_train.extend([label] * len(sequences))
                else:
                    X_val.extend(sequences)
                    y_val.extend([label] * len(sequences))

    if not X_train or not X_val:
        raise RuntimeError(
            "No valid training or validation sequences were created."
        )

    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    y_val = np.asarray(y_val, dtype=np.int64)

    np.save(OUTPUT_DIR / "X_train.npy", X_train)
    np.save(OUTPUT_DIR / "y_train.npy", y_train)
    np.save(OUTPUT_DIR / "X_val.npy", X_val)
    np.save(OUTPUT_DIR / "y_val.npy", y_val)

    with open(OUTPUT_DIR / "labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)

    # read by p3 (model input size) and p4 (same features at inference)
    meta = {**feature_meta(FEATURE_SET), "seq_length": SEQ_LENGTH, "stride": STRIDE, "fps": TARGET_FPS,
            "max_windows_per_video": MAX_WINDOWS_PER_VIDEO}
    with open(OUTPUT_DIR / "feature_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\nDataset created successfully!")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_val:  ", X_val.shape)
    print("y_val:  ", y_val.shape)
    print("Labels: ", labels)


if __name__ == "__main__":
    build_dataset()
