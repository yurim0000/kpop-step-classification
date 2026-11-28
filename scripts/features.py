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
NORMALIZE (camera-angle alignment) and VELOCITY (add joint speeds, doubles frame_dim) below.
"""
import numpy as np
from pathlib import Path

LOWER_BODY = list(range(23, 33))
FEATURE_SETS = {
    "lower": LOWER_BODY,
    "full": list(range(33)),
}
FEATURE_SET = "lower"  # change to "full" for the lower-body vs full-body experiment

# NORMALIZE (preprocessing experiment): videos are filmed from different camera angles,
# so the same step appears rotated in world coordinates.
#   "none" : hip-centered only
#   "video": rotate about the vertical (y) axis so the video's mean left->right hip direction
#            is +x, then divide by leg length (hip center -> ankle center, median)
#   "frame": same, but rotation recomputed every frame (also removes body turning)
NORMALIZE = "video"


def _align(rel, motion, mode):
    hv = motion[:, 24] - motion[:, 23]                      # left hip -> right hip
    yaw = np.arctan2(hv[:, 2], hv[:, 0])
    if mode == "video":
        yaw = np.full_like(yaw, np.arctan2(np.sin(yaw).mean(), np.cos(yaw).mean()))
    c, s = np.cos(-yaw)[:, None], np.sin(-yaw)[:, None]
    out = rel.copy()
    out[..., 0] = c * rel[..., 0] - s * rel[..., 2]
    out[..., 2] = s * rel[..., 0] + c * rel[..., 2]
    pelvis = (motion[:, 23] + motion[:, 24]) / 2
    leg = np.median(np.linalg.norm((motion[:, 27] + motion[:, 28]) / 2 - pelvis, axis=1))
    return out / leg


# VELOCITY: also give the model how fast each joint moves (frame-to-frame difference x fps,
# i.e. leg-lengths per second). Step identity is mostly in how the feet move, not where they are.
VELOCITY = True


# ROOT_MOTION: MediaPipe world landmarks always have the pelvis at the origin, so pelvis bounce,
# jumps and sideways travel are lost. The fixed camera still sees them: root_from_image() turns the
# 2D pelvis track into meters. With ROOT_MOTION = True
#   positions: joint heights are measured from the floor (pelvis bounce and jumps kept)
#   velocities: also include the whole-body sideways / vertical movement (crab travel)
# The absolute sideways position is NOT used (it depends on where in the clip a window starts).
ROOT_MOTION = False   # tested: same accuracy as True (5-fold CV 77.7% vs 78.3%) but less stable; jumps are kept in p1_3 instead


def load_image(path):
    """2D image landmarks of a p1 .npz -> (T, 33, 2) in pixels, y up; same frames as the world data.
    None if the file has no image landmarks (older extraction)."""
    d = np.load(path, allow_pickle=True)
    if "image" not in d.files or d["image"].size == 0:
        return None
    rows = d["image"].reshape(-1, 6)
    rows = rows[np.lexsort((rows[:, 1], rows[:, 0]))]          # by frame, then landmark
    img = rows[:, 2:4].reshape(-1, 33, 2)
    return np.stack([img[..., 0] * float(d["width"]), -img[..., 1] * float(d["height"])], axis=-1)


def root_from_image(motion, image):
    """(T,33,3) world + (T,33,2) image -> (T,3) pelvis position in meters:
    x = sideways from the clip's median position, y = pelvis height above the clip floor, z = 0.
    Pixels -> meters with the ratio of 3D to 2D leg length (fixed camera, roughly constant distance)."""
    pel3 = (motion[:, 23] + motion[:, 24]) / 2
    leg3 = np.median(np.linalg.norm((motion[:, 27] + motion[:, 28]) / 2 - pel3, axis=1))
    pel2 = (image[:, 23] + image[:, 24]) / 2
    leg2 = np.median(np.linalg.norm((image[:, 27] + image[:, 28]) / 2 - pel2, axis=1))
    s = leg3 / leg2
    lowest = image[:, 29:33, 1].min(axis=1)                    # lowest heel / toe in the image
    floor = np.percentile(lowest, 5)                           # ONE floor for the whole clip
    root = np.zeros((len(motion), 3))
    root[:, 0] = s * (pel2[:, 0] - np.median(pel2[:, 0]))
    root[:, 1] = s * (pel2[:, 1] - floor)
    return root


def make_features(motion, feature_set=FEATURE_SET, normalize=NORMALIZE, velocity=VELOCITY,
                  fps=30, root=None):
    """
    motion: (T, 33, 3) of ONE video (already at TARGET_FPS)
    root:   (T, 3) from root_from_image(), or None for pelvis-fixed coordinates
    -> (T, J, 3) positions, or (T, 2J, 3) positions + velocities if velocity=True
    """
    if motion.ndim != 3 or motion.shape[1:] != (33, 3):
        raise ValueError(f"Invalid motion shape: {motion.shape}")
    pelvis = (motion[:, 23] + motion[:, 24]) / 2
    rel = motion - pelvis[:, None, :]
    scale = 1.0
    if normalize != "none":
        rel = _align(rel, motion, normalize)
        scale = 1.0 / np.median(np.linalg.norm((motion[:, 27] + motion[:, 28]) / 2 - pelvis, axis=1))
    joints = FEATURE_SETS[feature_set]
    pos = rel.copy()
    if root is not None:
        pos[..., 1] += root[:, None, 1] * scale                # heights from the floor
    feat = pos[:, joints]
    if velocity:
        moving = pos.copy()
        if root is not None:
            moving[..., 0] += root[:, None, 0] * scale         # + whole-body sideways travel
        vel = np.diff(moving[:, joints], axis=0, prepend=moving[:1, joints]) * fps
        feat = np.concatenate([feat, vel], axis=1)
    return feat


def feature_meta(feature_set=FEATURE_SET, normalize=NORMALIZE, velocity=VELOCITY, root_motion=ROOT_MOTION):
    joints = FEATURE_SETS[feature_set]
    return {"feature_set": feature_set, "normalize": normalize, "velocity": velocity,
            "root_motion": root_motion, "joints": joints,
            "frame_dim": len(joints) * 3 * (2 if velocity else 1)}


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


def load_root(path, motion, root_motion=ROOT_MOTION):
    """pelvis track (T,3) for make_features(root=...), matching the already resampled motion.
    None when root motion is off or the npz has no image landmarks."""
    if not root_motion or Path(path).suffix != ".npz":
        return None
    image = load_image(path)
    if image is None:
        return None
    image = resample_fps(image, video_fps(path))
    if len(image) != len(motion):
        raise ValueError(f"image / world frame count differ in {path}: {len(image)} vs {len(motion)}")
    return root_from_image(motion, image)
