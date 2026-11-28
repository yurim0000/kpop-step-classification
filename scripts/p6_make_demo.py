"""
scripts/p6_make_demo.py

Demo video: the original dance video with the detected skeleton, and next to it the model's
prediction for the current second, the probabilities of all 5 steps and the final verdict.

Two modes
  default  cross-validation predictions: every clip is predicted by the 5-fold model that did NOT
           train on it (same setup as the final model of p5_run_experiments.py)
           clips: per step the best-predicted source videos, plus wrongly predicted ones
  --test   held-out test videos (data/test_videos/<step>/*.mp4, keypoints from
           `p1_create_raw_keypoints.py --local --test`), predicted by the model trained by p3

Run from final_project/ :
    python scripts/p6_make_demo.py              # -> data/output/demo.mp4
    python scripts/p6_make_demo.py --test       # -> data/output/demo_test.mp4
Options: --per-step N (correct clips per step, default 2), --wrong N (default 2), --seconds S (max per clip)
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

W, H, FPS = 1920, 1080, 30
KO = {"charleston": "찰스턴", "crab": "크랩", "hiphop_twostep": "힙합 투스텝",
      "penguin": "펭귄", "runningman_reverse": "리버스 런닝맨"}
BG, PANEL, DARK, MUTED = (247, 249, 252), (255, 255, 255), (34, 34, 34), (110, 110, 110)
BLUE, GREEN, RED, ORANGE = (0, 108, 184), (30, 150, 80), (200, 50, 45), (224, 123, 57)
BONES = [(11, 12), (11, 23), (12, 24), (23, 24), (11, 13), (13, 15), (12, 14), (14, 16),
         (23, 25), (25, 27), (27, 29), (29, 31), (27, 31), (24, 26), (26, 28), (28, 30), (30, 32), (28, 32)]
LOWER = set(range(23, 33))


def font(size, bold=False):
    for name in (["malgunbd.ttf"] if bold else []) + ["malgun.ttf", "arial.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


F = {k: font(*v) for k, v in {"title": (64, True), "h1": (46, True), "h2": (34, True), "body": (30, False),
                              "small": (24, False), "bar": (26, False), "big": (56, True)}.items()}


# ---------------------------------------------------------------- predictions
def cv_predictions():
    """{npz path: (probs per window (N,5), class names)} from the 5-fold models (each video once)."""
    import p5_run_experiments as p5
    cfg = next(c for c in p5.CONFIGS if c[0] == p5.FINAL)
    _, _, model_type, joints, norm, vel, L, bal, pel = cfg
    out = {}
    for fold in range(p5.N_FOLDS):
        data = p5.build(fold, joints, norm, vel, L, bal, pel)
        prob = p5.train(model_type, data, fold, bal)
        vids = data["val"][2]
        for d in p5.class_dirs:
            for p in p5.raw[d.name]:
                if p5.FOLD_OF[p] == fold:
                    out[p] = prob[vids == f"{d.name}/{p.stem}"]
        print(f"fold {fold + 1}/{p5.N_FOLDS} done", flush=True)
    return out, p5.names


def test_predictions():
    """{npz path: probs per window} for data/test_keypoints, with the model trained by p3."""
    import torch
    import p4_predict as p4
    from features import make_features, resample_fps, video_fps, load_root
    from p3_train_mlp import build_model
    meta = json.loads((ROOT / "data" / "dataset" / "feature_meta.json").read_text(encoding="utf-8"))
    labels = json.loads((ROOT / "data" / "dataset" / "labels.json").read_text(encoding="utf-8"))
    mm = json.loads((ROOT / "checkpoints" / "model_meta.json").read_text(encoding="utf-8"))
    model = build_model(mm["model_type"], (meta["seq_length"], meta["frame_dim"]), len(labels))
    model.load_state_dict(torch.load(ROOT / "checkpoints" / mm["checkpoint"], map_location="cpu", weights_only=True))
    model.eval()
    out = {}
    for p in sorted((ROOT / "data" / "test_keypoints").rglob("*.npz")):
        m = resample_fps(p4.load_raw_keypoints(p), video_fps(p))
        f = make_features(m, meta["feature_set"], meta.get("normalize", "none"), meta.get("velocity", False),
                          root=load_root(p, m, meta.get("root_motion", False)))
        X = torch.from_numpy(p4.create_sequences(f))
        with torch.no_grad():
            out[p] = torch.softmax(model(X), 1).numpy()
    return out, sorted(labels, key=labels.get)


# ---------------------------------------------------------------- drawing
def text(draw, xy, s, f, fill=DARK, anchor="la"):
    draw.text(xy, s, font=f, fill=fill, anchor=anchor)


def card_frame(title, lines, sub=None):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    text(d, (W // 2, 330), title, F["title"], BLUE, "mm")
    if sub:
        text(d, (W // 2, 420), sub, F["body"], MUTED, "mm")
    y = 520
    for ln in lines:
        text(d, (W // 2, y), ln, F["h2"] if ln.startswith("•") else F["body"], DARK, "mm")
        y += 64
    return np.array(img)


def draw_skeleton(frame, lm):
    """lm: (33, 2) normalized image coords (x, y down)"""
    h, w = frame.shape[:2]
    pts = (lm * [w, h]).astype(int)
    for i, j in BONES:
        col = (57, 123, 224) if (i in LOWER and j in LOWER) else (255, 255, 255)   # BGR: orange / white
        cv2.line(frame, tuple(pts[i]), tuple(pts[j]), col, max(3, w // 160), cv2.LINE_AA)
    for k in range(11, 33):
        cv2.circle(frame, tuple(pts[k]), max(4, w // 220), (255, 255, 255), -1, cv2.LINE_AA)
    return frame


def clip_frames(video, npz, probs, names, true_name, max_sec, index, total):
    z = np.load(npz, allow_pickle=True)
    img_rows = z["image"].reshape(-1, 6)
    img_rows = img_rows[np.lexsort((img_rows[:, 1], img_rows[:, 0]))].reshape(-1, 33, 6)
    frames_detected = img_rows[:, 0, 0].astype(int)
    lm = {int(fr): img_rows[i, :, 2:4] for i, fr in enumerate(frames_detected)}
    # window k covers detected frames [15k, 15k + 30)
    n_det = len(frames_detected)
    cover = np.zeros((n_det, len(names)))
    cnt = np.zeros(n_det)
    for k, pr in enumerate(probs):
        s, e = 15 * k, min(15 * k + 30, n_det)
        cover[s:e] += pr
        cnt[s:e] += 1
    last = probs[-1]
    det_prob = {int(fr): (cover[i] / cnt[i] if cnt[i] else last) for i, fr in enumerate(frames_detected)}
    final = probs.mean(0)
    final_name = names[int(final.argmax())]
    ok_final = final_name == true_name

    cap = cv2.VideoCapture(str(video))
    n = int(min(cap.get(cv2.CAP_PROP_FRAME_COUNT), max_sec * FPS))
    out, cur = [], np.full(len(names), 1 / len(names))
    for f in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        if f in lm:
            frame = draw_skeleton(frame, lm[f])
            cur = det_prob[f]
        vh = H - 120
        vw = int(frame.shape[1] * vh / frame.shape[0])
        frame = cv2.resize(frame, (vw, vh))
        img = Image.new("RGB", (W, H), BG)
        pred = names[int(cur.argmax())]
        good = pred == true_name
        d = ImageDraw.Draw(img)
        x0 = 90
        d.rectangle([x0 - 8, 52, x0 + vw + 8, 60 + vh + 8], fill=GREEN if good else RED)
        img.paste(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), (x0, 60))
        px = x0 + vw + 90
        text(d, (px, 70), f"{index} / {total}   {Path(video).stem}", F["small"], MUTED)
        text(d, (px, 120), "정답", F["body"], MUTED)
        text(d, (px, 160), KO.get(true_name, true_name), F["h1"], DARK)
        text(d, (px, 260), "지금 1초 예측", F["body"], MUTED)
        text(d, (px, 300), f"{KO.get(pred, pred)}  {cur.max():.0%}", F["big"], GREEN if good else RED)
        text(d, (px, 380), "맞음" if good else "틀림", F["h2"], GREEN if good else RED)
        by = 470
        for i, nm in enumerate(names):
            y = by + i * 70
            text(d, (px, y), KO.get(nm, nm), F["bar"], DARK)
            bx, bw = px + 230, 640
            d.rounded_rectangle([bx, y + 2, bx + bw, y + 36], 8, fill=(228, 232, 238))
            col = (GREEN if nm == true_name else RED) if nm == pred else (150, 170, 190)
            d.rounded_rectangle([bx, y + 2, bx + max(8, int(bw * cur[i])), y + 36], 8, fill=col)
            text(d, (bx + bw + 16, y + 4), f"{cur[i]:.0%}", F["bar"], DARK)
        prog = (f + 1) / n
        d.rounded_rectangle([px, 860, px + 870, 872], 6, fill=(228, 232, 238))
        d.rounded_rectangle([px, 860, px + max(12, int(870 * prog)), 872], 6, fill=BLUE)
        if f >= n - int(1.5 * FPS):
            text(d, (px, 905), "영상 전체 최종 판정", F["body"], MUTED)
            text(d, (px, 945), f"{KO.get(final_name, final_name)}  ({'정답' if ok_final else '오답'})",
                 F["h1"], GREEN if ok_final else RED)
        out.append(np.array(img))
    # hold the last frame 1.5 s so the verdict can be read
    out += [out[-1]] * int(1.5 * FPS)
    return out, ok_final


def write_video(chunks, path):
    tmp = path.with_suffix(".tmp.mp4")
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    n = 0
    for frames in chunks:
        for fr in frames:
            vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
            n += 1
    vw.release()
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "20", str(path)], check=True)
    tmp.unlink()
    return n / FPS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--per-step", type=int, default=2)
    ap.add_argument("--wrong", type=int, default=2)
    ap.add_argument("--seconds", type=float, default=7)
    args = ap.parse_args()

    if args.test:
        preds, names = test_predictions()
        video_dir = ROOT / "data" / "test_videos"
        if not preds:
            sys.exit("no test keypoints: put videos in data/test_videos/<step>/ and run "
                     "p1_create_raw_keypoints.py --local --test")
        chosen = sorted(preds)
        how = "테스트 영상 (학습에 쓰지 않은 다른 사람의 영상)"
        out_path = ROOT / "data" / "output" / "demo_test.mp4"
    else:
        preds, names = cv_predictions()
        video_dir = ROOT / "data" / "raw_videos"
        pairs = json.loads((ROOT / "data" / "mirror_pairs.json").read_text(encoding="utf-8"))
        info = []
        for p, pr in preds.items():
            if f"{p.parent.name}/{p.stem}" in pairs:              # originals only, no mirrored copies
                continue
            true = p.parent.name
            info.append({"p": p, "true": true, "ok": names[int(pr.mean(0).argmax())] == true,
                         "acc": float((pr.argmax(1) == names.index(true)).mean()), "n": len(pr)})
        chosen = []
        for st in names:
            good = sorted([i for i in info if i["true"] == st and i["ok"] and i["n"] >= 5],
                          key=lambda i: (-i["acc"], -i["n"]))
            chosen += [i["p"] for i in good[:args.per_step]]
        wrong = sorted([i for i in info if not i["ok"]], key=lambda i: i["acc"])
        chosen += [i["p"] for i in wrong[:args.wrong]]
        how = "5-fold 교차검증: 각 영상은 그 영상을 학습하지 않은 모델이 예측"
        out_path = ROOT / "data" / "output" / "demo.mp4"

    chunks = [[card_frame("K-pop 기본 스텝 분류 데모",
                          ["• 크랩   • 펭귄   • 힙합 투스텝   • 찰스턴   • 리버스 런닝맨",
                           "MediaPipe 3D 관절 좌표 (하체 10개) → 1D CNN",
                           "1초(30프레임) 조각마다 예측하고, 영상 전체는 조각 확률의 평균으로 판정"],
                          sub=how)] * int(4 * FPS)]
    results = []
    for i, p in enumerate(chosen, 1):
        video = next(video_dir.joinpath(p.parent.name).glob(f"{p.stem}.*"))
        frames, ok = clip_frames(video, p, preds[p], names, p.parent.name, args.seconds, i, len(chosen))
        chunks.append(frames)
        results.append(ok)
        print(f"{i}/{len(chosen)} {p.parent.name}/{p.stem}: {'correct' if ok else 'WRONG'}", flush=True)
    n_ok = sum(results)
    all_ok = sum(names[int(pr.mean(0).argmax())] == p.parent.name for p, pr in preds.items())
    scope = "교차검증 전체" if not args.test else "테스트 영상 전체"
    chunks.append([card_frame("정리",
                              [f"이 데모에서 {len(results)}개 중 {n_ok}개 정답, {len(results) - n_ok}개 오답",
                               f"{scope}: 영상 {len(preds)}개 중 {all_ok}개 정답 ({all_ok / len(preds):.1%})"])]
                  * int(5 * FPS))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sec = write_video(chunks, out_path)
    print(f"saved -> {out_path}  ({sec:.0f} s)")


if __name__ == "__main__":
    main()
