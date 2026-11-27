"""
scripts/p1_3_analyze_postprocessing.py

Assignment 02 helper (lower-body K-pop steps): compare raw BlazePose 3D keypoints with
  (1) the provided post-processing  (create_height_corrected_target in p1_2_post_processing.py)
  (2) bonus: provided post-processing + Savitzky-Golay smoothing
and report lower-body motion metrics.

Run from final_project/ :
    python scripts/p1_3_analyze_postprocessing.py                       # every .npz under data/raw_keypoints
    python scripts/p1_3_analyze_postprocessing.py --file data/raw_keypoints/crab/crab_1.npz

Outputs (data/output/):  <name>_metrics.csv, <name>_analysis.png, all_metrics.csv

World-coordinate metrics are divided by the leg length (hip center -> ankle center, median)
so performers / camera distances are comparable. No thresholds are applied: metrics are
continuous (percentiles), decide thresholds after looking at the collected data.

World landmarks are hip-centered (to be confirmed: see the "hip center" line printed per file),
so whole-body travel and jump height are measured from the 2D image landmarks instead
(saved as "image" in the npz by p1_create_raw_keypoints.py; older npz files have none).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from scripts.p1_2_post_processing import create_height_corrected_target
except ImportError as e:  # e.g. viser not installed in this environment
    print(f"[info] could not import provided post-processing ({e}); using a copy of it.")

    def create_height_corrected_target(raw_seq):
        # copied from p1_2_post_processing.py
        seq = raw_seq.copy()
        foot_y = np.minimum(seq[:, 31, 1], seq[:, 32, 1])
        seq[:, :, 1] -= foot_y[:, None]
        seq[:, :, 1] = np.maximum(seq[:, :, 1], 0)
        return seq

OUT_DIR = ROOT / "data" / "output"
SG_WINDOW, SG_POLY = 9, 3          # smoothing used as the bonus step
LOWER = list(range(23, 33))        # hips, knees, ankles, heels, toes
HEEL = {"L": 29, "R": 30}
TOE = {"L": 31, "R": 32}
ANKLE = {"L": 27, "R": 28}
# step-cycle search range: covers ~60-180 BPM dance tempos (a left-right cycle is 1-4 beats)
CYCLE_RANGE_S = (0.25, 2.5)


# ---------------------------------------------------------------- loading
def _rows_to_array(rows):
    df = pd.DataFrame(rows, columns=["frame", "landmark", "x", "y", "z", "visibility"])
    df = df.astype({"frame": int, "landmark": int}).sort_values(["frame", "landmark"])
    n = df["frame"].nunique()
    if len(df) != n * 33:
        raise ValueError("expected 33 landmarks per frame")
    return (df[["x", "y", "z"]].to_numpy().reshape(n, 33, 3),
            df["visibility"].to_numpy().reshape(n, 33),
            df["frame"].to_numpy()[::33])


def load_raw(path):
    """Return dict: world (T,33,3), vis (T,33), frames (T,), image (T,33,2) in pixels with y up, or None, fps."""
    z = np.load(path, allow_pickle=True)
    out = {"fps": float(z["fps"]) if "fps" in z.files and float(z["fps"]) > 0 else None, "image": None}
    if z["data"].size == 0:  # no pose detected in the whole video
        out.update(world=np.zeros((0, 33, 3)), vis=np.zeros((0, 33)), frames=np.zeros(0, int))
        return out
    out["world"], out["vis"], out["frames"] = _rows_to_array(z["data"])
    if "image" in z.files and z["image"].size:
        img, _, _ = _rows_to_array(z["image"])
        w, h = float(z["width"]), float(z["height"])
        out["image"] = np.stack([img[:, :, 0] * w, -img[:, :, 1] * h], axis=2)  # pixels, y up
    return out


# ---------------------------------------------------------------- metrics
def foot_heights(P):
    """Lowest point of each foot (min of heel, toe y) minus the sequence ground (5th percentile of both feet)."""
    yL = np.minimum(P[:, HEEL["L"], 1], P[:, TOE["L"], 1])
    yR = np.minimum(P[:, HEEL["R"], 1], P[:, TOE["R"], 1])
    ground = np.percentile(np.minimum(yL, yR), 5)
    return yL - ground, yR - ground


def toe_yaw(P, side):
    """Heel->toe direction in the floor plane (x, z), degrees, relative to the clip's mean direction,
    wrapped to [-180, 180] (no unwrapping: short / noisy vectors would make it drift without bound)."""
    v = P[:, TOE[side]] - P[:, HEEL[side]]
    a = np.arctan2(v[:, 0], v[:, 2])
    mean = np.arctan2(np.sin(a).mean(), np.cos(a).mean())
    return np.degrees(np.angle(np.exp(1j * (a - mean))))


def yaw_speed(yaw, fps):
    """Mean absolute frame-to-frame direction change (deg/s), with wrap-around handled."""
    d = (np.diff(yaw) + 180) % 360 - 180
    return np.abs(d).mean() * fps


def step_cycle(hL, hR, fps):
    """Period (s) of the left-right foot-height difference, by autocorrelation; and its peak value."""
    s = (hL - hR) - (hL - hR).mean()
    s = s / (np.linalg.norm(s) + 1e-9)
    ac = np.correlate(s, s, mode="full")[len(s) - 1:]
    lo, hi = int(CYCLE_RANGE_S[0] * fps), min(int(CYCLE_RANGE_S[1] * fps), len(s) - 1)
    if hi <= lo:
        return np.nan, np.nan
    lag = lo + int(np.argmax(ac[lo:hi + 1]))
    return lag / fps, ac[lag]


def p_range(x):
    return np.percentile(x, 95) - np.percentile(x, 5)


def compute_metrics(pts, leg, fps):
    P = pts / leg
    hL, hR = foot_heights(P)
    yaw = {k: toe_yaw(P, k) for k in "LR"}
    cycle_s, cycle_strength = step_cycle(hL, hR, fps)
    ankles = P[:, [ANKLE["L"], ANKLE["R"]]]
    jerk = np.diff(ankles, n=3, axis=0) * fps**3
    res = P[:, LOWER] - savgol_filter(P[:, LOWER], SG_WINDOW, SG_POLY, axis=0)
    hip_y = (P[:, 23, 1] + P[:, 24, 1]) / 2
    return {
        "foot_lift_p95": max(np.percentile(hL, 95), np.percentile(hR, 95)),
        "both_feet_off_p95": np.percentile(np.minimum(hL, hR), 95),    # jump-like: both feet up together
        "lr_height_corr": np.corrcoef(hL, hR)[0, 1],                   # < 0: feet alternate
        "step_cycle_s": cycle_s,
        "step_cycle_strength": cycle_strength,                         # autocorrelation at that period
        "foot_z_sep_range": p_range(P[:, ANKLE["L"], 2] - P[:, ANKLE["R"], 2]),  # front-back foot separation
        "toe_yaw_range_deg": max(p_range(yaw["L"]), p_range(yaw["R"])),
        "toe_yaw_speed_deg_s": max(yaw_speed(yaw[k], fps) for k in "LR"),
        "ankle_jerk_rms": np.sqrt(np.mean(np.sum(jerk**2, axis=2))),
        "jitter_lower": np.sqrt(np.mean(np.sum(res**2, axis=2))),
        "hip_y_range_world": p_range(hip_y),
    }, (hL, hR), yaw


def image_metrics(img):
    """Whole-body travel and jump from 2D image landmarks (fixed camera), in image leg lengths."""
    hip = (img[:, 23] + img[:, 24]) / 2
    ank = (img[:, 27] + img[:, 28]) / 2
    leg_px = np.median(np.linalg.norm(hip - ank, axis=1))
    yL = np.minimum(img[:, HEEL["L"], 1], img[:, TOE["L"], 1])
    yR = np.minimum(img[:, HEEL["R"], 1], img[:, TOE["R"], 1])
    low = np.minimum(yL, yR)
    return {
        "img_hip_x_range": p_range(hip[:, 0]) / leg_px,                       # sideways travel (crab)
        "img_hip_y_range": p_range(hip[:, 1]) / leg_px,                       # bounce / jump of the body
        "img_both_feet_off_p95": (np.percentile(low, 95) - np.percentile(low, 5)) / leg_px,
    }, hip / leg_px


# ---------------------------------------------------------------- per file
def analyze(path, fps=None, plot=True):
    d = load_raw(path)
    raw, vis = d["world"], d["vis"]
    if len(raw) < 30:
        print(f"skip (too short or no pose): {path.name}")
        return None
    fps = fps or d["fps"] or 30.0
    name = path.stem
    missing = int(d["frames"][-1] + 1 - len(d["frames"]))
    hip_c = (raw[:, 23] + raw[:, 24]) / 2
    leg = np.median(np.linalg.norm(hip_c - (raw[:, 27] + raw[:, 28]) / 2, axis=1))

    corrected = create_height_corrected_target(raw)
    smoothed = savgol_filter(corrected, SG_WINDOW, SG_POLY, axis=0)
    smoothed[:, :, 1] = np.maximum(smoothed[:, :, 1], 0)
    conds = {"raw": raw, "height_corrected (provided)": corrected, "corrected + smoothing (bonus)": smoothed}

    img_m, img_hip = ({}, None) if d["image"] is None else image_metrics(d["image"])
    rows, heights, yaws = [], {}, None
    for cname, seq in conds.items():
        m, heights[cname], yaw = compute_metrics(seq, leg, fps)
        yaws = yaw if yaws is None else yaws
        rows.append({"file": name, "step": path.parent.name, "condition": cname, **m, **img_m})
    table = pd.DataFrame(rows)

    print(f"\n=== {path.parent.name}/{name}  (frames={len(raw)}, fps={fps:g}) ===")
    print(f"missing frames (no pose detected): {missing}")
    print(f"hip center, first frame (x, y, z): {np.round(hip_c[0], 4)} | max |hip center|: "
          f"{np.abs(hip_c).max():.4f}  (~0 if world landmarks are hip-centered)")
    print(f"foot (heel/toe) visibility < 0.5: {np.mean(vis[:, 29:33].min(axis=1) < 0.5):.1%} of frames")
    foot_y = np.minimum(raw[:, 31, 1], raw[:, 32, 1]) / leg
    print(f"per-frame ground used by provided method, jitter (RMS 2nd diff / leg): "
          f"{np.sqrt(np.mean(np.diff(foot_y, n=2) ** 2)):.4f}")
    if d["image"] is None:
        print("no image landmarks in this npz (re-extract with the current p1 to get travel / jump metrics)")
    print(table.drop(columns=["file", "step"] + list(img_m)).round(3).to_string(index=False))
    if img_m:
        print("image (2D):", {k: round(v, 3) for k, v in img_m.items()})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / f"{name}_metrics.csv", index=False)

    if plot:
        t = np.arange(len(raw)) / fps
        fig, ax = plt.subplots(2, 2, figsize=(12, 7))
        for cname, ls in zip(["raw", "height_corrected (provided)"], ["-", "--"]):
            hL, hR = heights[cname]
            ax[0, 0].plot(t, hL, ls, lw=1, label=f"L, {cname}")
            ax[0, 0].plot(t, hR, ls, lw=1, label=f"R, {cname}")
        ax[0, 0].set_title("Foot height above ground (leg lengths), world")
        ax[0, 0].legend(fontsize=6)
        if img_hip is not None:
            ax[0, 1].plot(t, img_hip[:, 0] - img_hip[0, 0], lw=1, label="x (sideways)")
            ax[0, 1].plot(t, img_hip[:, 1] - img_hip[0, 1], lw=1, label="y (up)")
            ax[0, 1].legend(fontsize=7)
        ax[0, 1].set_title("Hip center in the image, from first frame (image leg lengths)")
        ax[1, 0].plot(t, foot_y, lw=1, color="gray")
        ax[1, 0].set_title("Per-frame ground used by the provided method (min toe y / leg)")
        for k in "LR":
            ax[1, 1].plot(t, yaws[k], lw=1, label=k)
        ax[1, 1].set_title("Toe direction vs clip mean (deg), raw")
        ax[1, 1].legend(fontsize=7)
        for a in ax.flat:
            a.set_xlabel("s")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{name}_analysis.png", dpi=150)
        plt.close(fig)
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, help="one raw keypoint .npz")
    ap.add_argument("--input-dir", type=Path, default=ROOT / "data" / "raw_keypoints")
    ap.add_argument("--fps", type=float, default=None,
                    help="video frame rate; default = fps stored in the npz, or 30 if none")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    paths = [args.file] if args.file else sorted(args.input_dir.rglob("*.npz"))
    if not paths:
        sys.exit(f"No .npz files found under {args.input_dir}")
    tables = [t for t in (analyze(p, args.fps, not args.no_plot) for p in paths) if t is not None]
    if tables:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        pd.concat(tables).to_csv(OUT_DIR / "all_metrics.csv", index=False)
        print(f"\nsaved csv / png -> {OUT_DIR}")


if __name__ == "__main__":
    main()
