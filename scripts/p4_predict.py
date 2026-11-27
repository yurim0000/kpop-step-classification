
from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
import torch
import os

ROOT = Path(__file__).resolve().parent.parent

# 프로젝트 루트와 scripts 폴더 등록
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from utils.viser_test import PoseViser
from importlib import import_module

# Import the same model used during training
ActionMLP = import_module("p3_train_mlp").ActionMLP
from features import make_features, video_fps, resample_fps

MODEL_PATH = ROOT / "checkpoints" / "best_mlp.pth"
LABEL_PATH = ROOT / "data" / "dataset" / "labels.json"
META_PATH = ROOT / "data" / "dataset" / "feature_meta.json"  # written by p2_create_dataset.py

# Keypoint file to test: give it on the command line, e.g.
#   python scripts/p4_predict.py data/raw_keypoints/wave/wave_140_t03.npz
INPUT_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "test_keypoints" / "test.npz"

SEQ_LENGTH = 30
STRIDE = 15


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
    """Convert (T, J, 3) to (N, 30, J*3)."""

    sequences = []
    frame_dim = motion.shape[1] * 3

    for start in range(0, len(motion) - SEQ_LENGTH + 1, STRIDE):
        clip = motion[start:start + SEQ_LENGTH]
        sequences.append(clip.reshape(SEQ_LENGTH, frame_dim))

    if not sequences:
        raise ValueError("Video is shorter than 30 detected frames.")

    return np.asarray(sequences, dtype=np.float32)

def export_unity_csv(motion, output_path):
    """
    motion: (T, 33, 3)
    output: Unity-compatible CSV
    """
    rows = []

    T, n_bones, _ = motion.shape

    for t in range(T):
        for b in range(n_bones):
            x, y, z = motion[t, b]
            rows.append([t, b, x, y, z, 1.0])

    df = pd.DataFrame(
        rows,
        columns=[
            "frame", "landmark",
            "x", "y", "z", "visibility"
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Unity CSV saved: {output_path}")

def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load class labels
    with open(LABEL_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)

    id_to_label = {v: k for k, v in labels.items()}

    # Same input features as training
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Load trained model
    model = ActionMLP(
        input_size=meta["seq_length"] * meta["frame_dim"],
        num_classes=len(labels),
    ).to(device)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device, weights_only=True)
    )
    model.eval()

    # Load and prepare input
    motion = load_raw_keypoints(INPUT_PATH)
    sequences = create_sequences(make_features(resample_fps(motion, video_fps(INPUT_PATH)), meta["feature_set"], meta.get("normalize", "none")))

    X = torch.from_numpy(sequences).to(device)

    # Predict
    with torch.no_grad():
        logits = model(X)
        probabilities = torch.softmax(logits, dim=1)

    # Average probabilities across all windows
    avg_probabilities = probabilities.mean(dim=0)

    predicted_id = avg_probabilities.argmax().item()
    predicted_label = id_to_label[predicted_id]

    print(f"\nInput: {INPUT_PATH.name}")
    print(f"Frames: {len(motion)}")
    print(f"Sequences: {len(sequences)}")

    print("\nPrediction probabilities:")
    for i, probability in enumerate(avg_probabilities):
        print(f"  {id_to_label[i]}: {probability.item():.4f}")

    print(f"\nPredicted action: {predicted_label}")


    # Export keypoints for Unity
    OUTPUT_PATH = (
        ROOT / "data" / "output" /
        f"{INPUT_PATH.stem}_{predicted_label}.csv"
    )

    export_unity_csv(motion, OUTPUT_PATH)

    vis = PoseViser(fps=30)

    vis.server.gui.add_text(
        "Predicted Action",
        initial_value=predicted_label,
        disabled=True
    )

    vis.server.gui.add_number(
        "Confidence",
        initial_value=float(avg_probabilities[predicted_id].item()),
        disabled=True
    )

    vis.play_sequence(motion)


if __name__ == "__main__":
    main()
