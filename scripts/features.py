"""
scripts/features.py

Input-feature design shared by p2_create_dataset.py (training) and p4_predict.py (inference),
so both always build the same input.

FEATURE_SET
  "lower": hips(23, 24), knees(25, 26), ankles(27, 28), heels(29, 30), toes(31, 32)
           -> 10 joints x 3 = 30 values / frame
  "full" : all 33 BlazePose joints -> 99 values / frame   (comparison experiment)
Both are expressed relative to the hip center (mean of 23, 24) in every frame.
Note: this removes whole-body travel (crab moves sideways) and whole-body height (jumps).
"""
import numpy as np
from pathlib import Path

LOWER_BODY = list(range(23, 33))
FEATURE_SETS = {
    "lower": LOWER_BODY,
    "full": list(range(33)),
}
FEATURE_SET = "lower"  # change to "full" for the lower-body vs full-body experiment


def make_features(motion, feature_set=FEATURE_SET):
    """motion: (T, 33, 3) -> (T, J, 3), hip-centered, selected joints only."""
    if motion.ndim != 3 or motion.shape[1:] != (33, 3):
        raise ValueError(f"Invalid motion shape: {motion.shape}")
    pelvis = (motion[:, 23] + motion[:, 24]) / 2
    rel = motion - pelvis[:, None, :]
    return rel[:, FEATURE_SETS[feature_set]]


def feature_meta(feature_set=FEATURE_SET):
    joints = FEATURE_SETS[feature_set]
    return {"feature_set": feature_set, "joints": joints, "frame_dim": len(joints) * 3}


TARGET_FPS = 30  # internet videos come at 24/30/60 fps: resample so a 30-frame window is always ~1 s


def video_fps(path):
    """fps stored by p1 in the npz (None for older files / .npy)."""
    if Path(path).suffix != ".npz":
        return None
    d = np.load(path, allow_pickle=True)
    return float(d["fps"]) if "fps" in d.files and float(d["fps"]) > 0 else None


def resample_fps(motion, fps, target=TARGET_FPS):
    """(T,...) -> frames picked at target fps (nearest frame). No-op if fps unknown or ~target."""
    if fps is None or abs(fps - target) < 1:
        return motion
    idx = np.round(np.arange(0, len(motion), fps / target)).astype(int)
    return motion[idx[idx < len(motion)]]
