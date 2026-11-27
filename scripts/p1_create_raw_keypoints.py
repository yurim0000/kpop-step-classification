# scripts/download_and_extract.py

import os
import sys
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_VIDEO_DIR = os.path.join(ROOT, "data", "raw_videos")
RAW_KEYPOINT_DIR = os.path.join(ROOT, "data", "raw_keypoints")
RAW_TEST_DIR = os.path.join(ROOT, "data", "test_keypoints")
TEST_VIDEO_DIR = os.path.join(ROOT, "data", "test_videos")  # held-out performer, never read by p2

os.makedirs(RAW_VIDEO_DIR, exist_ok=True)
os.makedirs(RAW_KEYPOINT_DIR, exist_ok=True)

mp_pose = mp.solutions.pose
POSE_ARGS = dict(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}

def download_youtube(url):
    from pytubefix import YouTube  # only needed for YouTube mode (--local works without it)
    yt = YouTube(url)
    name = yt.title.replace(" ", "_").replace("/", "_")
    filepath = f"{RAW_VIDEO_DIR}/{name}.mp4"
    yt.streams.filter(file_extension='mp4').first().download(
        output_path=RAW_VIDEO_DIR,
        filename=f"{name}.mp4"
    )
    print(f"Downloaded → {filepath}")
    return filepath, name


def extract_3d_keypoints(video_path, dir_path =RAW_KEYPOINT_DIR, name ="Data"):
    # new model per video: tracking/smoothing state must not carry over from the previous video
    pose_model = mp_pose.Pose(**POSE_ARGS)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    pose_rows = []
    image_rows = []  # 2D image landmarks: keep whole-body travel / jumps that hip-centered world coords drop

    pbar = tqdm(total=total_frames, desc="Extracting BlazePose 3D",
                ascii=True,          # unicode 막대 → ASCII 막대로 변경
                dynamic_ncols=False,  # 윈도우 콘솔 버그 방지,
                disable=True
                )
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose_model.process(rgb)

        if res.pose_world_landmarks:
            for i, lm in enumerate(res.pose_world_landmarks.landmark):
                pose_rows.append({
                    "frame": frame_idx,
                    "landmark": i,
                    "x": lm.x,
                    "y": -lm.y,      # flip for unity-like coords
                    "z": lm.z,
                    "visibility": lm.visibility,
                })
            # same frames: image coords (x, y normalized to [0,1], y downwards), saved separately
            for i, lm in enumerate(res.pose_landmarks.landmark):
                image_rows.append([frame_idx, i, lm.x, lm.y, lm.z, lm.visibility])

        frame_idx += 1
        pbar.update(1)
    cap.release()
    pbar.close()
    pose_model.close()

    df = pd.DataFrame(pose_rows)

    # extract local positions from pelvis


    out_path = f"{dir_path}/{name}.npz"
    # "data" = world landmarks (unchanged format, used by p2/p4)
    # "image", "fps", "width", "height" = extra keys, read by p1_3_analyze_postprocessing.py
    np.savez(out_path, data=df.to_numpy(), image=np.asarray(image_rows, dtype=float).reshape(-1, 6),
             fps=fps, width=width, height=height)
    n_detected = len(pose_rows) // 33
    print(f"Saved 3D keypoints -> {out_path}  (fps={fps:.2f}, "
          f"pose detected in {n_detected}/{frame_idx} frames)")
    if n_detected < frame_idx * 0.9:
        print("  WARNING: pose missing in >10% of frames - check framing / lighting")


def extract_local_videos(overwrite=False, src_dir=RAW_VIDEO_DIR, dst_dir=RAW_KEYPOINT_DIR):
    """<src_dir>/<label>/<name>.mp4  ->  <dst_dir>/<label>/<name>.npz"""
    os.makedirs(src_dir, exist_ok=True)
    class_dirs = sorted((d for d in os.scandir(src_dir) if d.is_dir()), key=lambda e: e.name)
    if not class_dirs:
        print(f"No label folders in {src_dir} (expected e.g. crab/, penguin/, ...)")
    for d in class_dirs:
        out_dir = os.path.join(dst_dir, d.name)
        os.makedirs(out_dir, exist_ok=True)
        for f in sorted(os.scandir(d.path), key=lambda e: e.name):
            name, ext = os.path.splitext(f.name)
            if ext.lower() not in VIDEO_EXTS:
                continue
            if not overwrite and os.path.exists(os.path.join(out_dir, f"{name}.npz")):
                print(f"skip (already extracted): {d.name}/{f.name}")
                continue
            print(f"{d.name}/{f.name}")
            extract_3d_keypoints(f.path, out_dir, name)

def main():
    # Own videos:  data/raw_videos/<label>/  -> data/raw_keypoints/  (training)   : --local
    #              data/test_videos/<label>/ -> data/test_keypoints/ (held out)   : --local --test
    if "--local" in sys.argv:
        overwrite = "--overwrite" in sys.argv
        if "--test" in sys.argv:
            extract_local_videos(overwrite, TEST_VIDEO_DIR, RAW_TEST_DIR)
        else:
            extract_local_videos(overwrite)
        return
    # url_list=["https://youtube.com/shorts/fNvA0dU5fZM?si=FFBbj_IprAQ4S_fs",
    # "https://youtube.com/shorts/rCECyHdKGnA?si=CQ_qbTmh2p0k3oJ3",
    # "https://youtube.com/shorts/GX6Tl9KtQgE?si=hNApdx7ZNOY_U0su",
    # 'https://youtube.com/shorts/Xuzazm0VAik?si=2ugZmNk6XtKHCkxe',
    # "https://youtube.com/shorts/Z5IKezdRTKU?si=n7hRZHvkP2Z809hy",
    # "https://youtube.com/shorts/-XgnHvuScP8?si=yxnOz7JfpsjHnv7q"]
    # url_list = ["https://youtube.com/shorts/vl8sBA0oIYU?si=4qZn3gt-5u2hs2YY",
    #             "https://youtube.com/shorts/xXNFSn_GU7Y?si=xh9T-XFXJxbsSA5I",
    #             "https://youtube.com/shorts/cakv_6_PTBA?si=gqxtP8QKsfpJGsoY",
    #             "https://youtube.com/shorts/N62KmR7zN0A?si=vstkCnOmTsC60D1x",
    #             "https://youtube.com/shorts/3tvVRaFbKKE?si=afZR4ymGQJWGhvRQ"
    #             ]
    #url_list=["https://youtube.com/shorts/ogufY7Ak7dU?si=QnjmZOwAR0nAhRn-"]
    # url_list=["https://youtube.com/shorts/-5LhNSMBrEs?si=2QPPzJ2rRr5jDduz",
    #           "https://youtube.com/shorts/bT2iY8IjEU0?si=oeAC9nITvjEihN4N",
    #           "https://youtube.com/shorts/pILmKQe5V94?si=NxizuiMq1_wAk0oW"]
    # url_list=["https://youtube.com/shorts/636IhyuvN-4?si=mbA9cwv1UcDXlG-_",
    #           "https://youtube.com/shorts/4Ntle88WdKU?si=CaS2ZVKJzPsHAEHr",
    #           "https://youtube.com/shorts/p8eMhRdeX4g?si=Rn84gSdaR0aLualY",]
    url_list=["https://youtube.com/shorts/ATYX9LAbpdY?si=ZYhnxaaltSJROsCA"]
    for url in url_list:
        video_path, name = download_youtube(url)
        extract_3d_keypoints(video_path, RAW_KEYPOINT_DIR, name)
        # Optional : Diverse Types of Data

if __name__ == "__main__":
    main()
