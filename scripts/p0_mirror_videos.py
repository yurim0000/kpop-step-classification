"""
scripts/p0_mirror_videos.py

Data augmentation: save a left-right mirrored copy of every video in data/raw_videos/<step>/,
numbered after the last original (crab_1..8 -> crab_9..16).

The pairs are written to data/mirror_pairs.json  {"crab/crab_9": "crab/crab_1", ...}
p2_create_dataset.py reads it so a video and its mirror always land on the same side
of the train/validation split (otherwise validation would contain near-copies of training data).

Run from final_project/ (needs ffmpeg, included in the ai_ml_final env):
    python scripts/p0_mirror_videos.py
Then extract keypoints for the new files:
    python scripts/p1_create_raw_keypoints.py --local
"""
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "data" / "raw_videos"
PAIRS_PATH = ROOT / "data" / "mirror_pairs.json"
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


def number_of(path):
    m = re.search(r"_(\d+)$", path.stem)
    return int(m.group(1)) if m else None


def main():
    pairs = json.loads(PAIRS_PATH.read_text(encoding="utf-8")) if PAIRS_PATH.exists() else {}
    mirrors = set(pairs)                 # "step/name" of files that are mirrors
    sources = set(pairs.values())        # originals that already have a mirror

    for step_dir in sorted(p for p in VIDEO_DIR.iterdir() if p.is_dir()):
        videos = sorted((p for p in step_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS),
                        key=lambda p: (number_of(p) is None, number_of(p) or 0, p.name))
        next_n = max((number_of(p) or 0 for p in videos), default=0) + 1
        for src in videos:
            key = f"{step_dir.name}/{src.stem}"
            if key in mirrors or key in sources:
                continue
            dst = step_dir / f"{step_dir.name}_{next_n}{src.suffix.lower()}"
            next_n += 1
            # hflip; re-encode video (H.264, same fps), keep audio if any
            cmd = ["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-vf", "hflip",
                   "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "copy", str(dst)]
            subprocess.run(cmd, check=True)
            pairs[f"{step_dir.name}/{dst.stem}"] = key
            print(f"{key}  ->  {step_dir.name}/{dst.name}")

    PAIRS_PATH.write_text(json.dumps(pairs, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"pairs saved -> {PAIRS_PATH}  ({len(pairs)} mirrored videos)")


if __name__ == "__main__":
    main()
