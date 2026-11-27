"""
scripts/p5_run_experiments.py

Reproduces the experiment table and figures of the report from data/raw_keypoints/.
Each configuration is trained on 3 different video-level train/val splits (SEEDS) and the
mean +- std validation accuracy is reported, both at the best epoch (what p3 saves) and at
the last epoch (no selection on validation data, the more honest number).

Run from final_project/ :
    python scripts/p5_run_experiments.py
Outputs:
    docs/figures/experiments.csv
    docs/figures/experiments.png          bar chart of all configurations
    docs/figures/confusion_matrix.png     final model (CNN + velocity), seed 42 split, last epoch
    docs/figures/per_video.csv            per validation video: true / predicted step
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2_create_dataset as p2
from features import make_features, resample_fps, video_fps
from p3_train_mlp import build_model, BATCH_SIZE, EPOCHS, LEARNING_RATE

ROOT = Path(__file__).resolve().parent.parent
IN_DIR = ROOT / "data" / "raw_keypoints"
OUT_DIR = ROOT / "docs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEEDS = [1, 7, 42]
STRIDE = 15

# (name, experiment group, model, joints, normalize, velocity, window frames)
CONFIGS = [
    ("MLP | none",               "3 preprocessing", "mlp", "lower", "none",  False, 30),
    ("MLP | video-align",        "3 preprocessing", "mlp", "lower", "video", False, 30),
    ("MLP | frame-align",        "3 preprocessing", "mlp", "lower", "frame", False, 30),
    ("MLP | video-align | 2s",   "window",          "mlp", "lower", "video", False, 60),
    ("MLP | video-align | full", "2 lower vs full", "mlp", "full",  "video", False, 30),
    ("MLP | +velocity",          "feature",         "mlp", "lower", "video", True,  30),
    ("CNN | position",           "1 model",         "cnn", "lower", "video", False, 30),
    ("CNN | +velocity (final)",  "1 model",         "cnn", "lower", "video", True,  30),
]
FINAL = "CNN | +velocity (final)"

class_dirs = sorted(p for p in IN_DIR.iterdir() if p.is_dir())
labels = {d.name: i for i, d in enumerate(class_dirs)}
names = list(labels)
raw = {d.name: {p: resample_fps(p2.load_raw_keypoints(p), video_fps(p))
                for p in sorted(d.glob("*.npz"))} for d in class_dirs}


def windows(f, L):
    return [f[s:s + L].reshape(L, -1) for s in range(0, len(f) - L + 1, STRIDE)]


def build(seed, joints, normalize, velocity, L):
    rng = np.random.default_rng(seed)
    data = {"train": ([], [], []), "val": ([], [], [])}
    for d in class_dirs:
        tr, va = p2.split_video_paths(list(raw[d.name]), rng)
        for split, paths in (("train", tr), ("val", va)):
            for p in paths:
                w = windows(make_features(raw[d.name][p], joints, normalize, velocity), L)
                X, y, v = data[split]
                X += w
                y += [labels[d.name]] * len(w)
                v += [f"{d.name}/{p.stem}"] * len(w)
    out = {}
    for split, (X, y, v) in data.items():
        out[split] = (torch.tensor(np.asarray(X), dtype=torch.float32),
                      torch.tensor(y, dtype=torch.long), np.asarray(v))
    return out


def train(model_type, data, seed):
    torch.manual_seed(seed)
    Xtr, ytr, _ = data["train"]
    Xva, yva, _ = data["val"]
    model = build_model(model_type, tuple(Xtr.shape[1:]), len(labels))
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    ce = nn.CrossEntropyLoss()
    best = 0.0
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(ytr))
        for i in range(0, len(perm), BATCH_SIZE):
            b = perm[i:i + BATCH_SIZE]
            if len(b) < 2:          # BatchNorm needs >1 sample
                continue
            opt.zero_grad()
            ce(model(Xtr[b]), ytr[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(Xva).argmax(1)
        acc = (pred == yva).float().mean().item()
        best = max(best, acc)
    with torch.no_grad():
        prob = torch.softmax(model(Xva), 1).numpy()
    return best, acc, prob


def main():
    rows, final_eval = [], None
    for name, group, model_type, joints, norm, vel, L in CONFIGS:
        bests, lasts = [], []
        for seed in SEEDS:
            data = build(seed, joints, norm, vel, L)
            b, a, prob = train(model_type, data, seed)
            bests.append(b)
            lasts.append(a)
            if name == FINAL and seed == 42:
                final_eval = (data["val"][1].numpy(), prob, data["val"][2])
        rows.append({"config": name, "experiment": group,
                     "best_mean": np.mean(bests), "best_std": np.std(bests),
                     "last_mean": np.mean(lasts), "last_std": np.std(lasts),
                     "best_per_seed": [round(x, 3) for x in bests],
                     "last_per_seed": [round(x, 3) for x in lasts]})
        r = rows[-1]
        print(f"{name:28s} best {r['best_mean']:.3f}+-{r['best_std']:.3f} | "
              f"last {r['last_mean']:.3f}+-{r['last_std']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "experiments.csv", index=False)

    # bar chart
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(df))
    ax.bar(x - 0.2, df.best_mean, 0.4, yerr=df.best_std, capsize=3, label="best epoch", color="#4C72B0")
    ax.bar(x + 0.2, df.last_mean, 0.4, yerr=df.last_std, capsize=3, label="last epoch", color="#DD8452")
    ax.axhline(1 / len(labels), color="gray", ls="--", lw=1, label="chance (20%)")
    ax.set_xticks(x, df.config, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("validation accuracy (windows)")
    ax.set_ylim(0, 1)
    ax.set_title("Validation accuracy, mean +- std over 3 video-level splits")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "experiments.png", dpi=150)

    # confusion matrix (windows) + per-video vote for the final model
    y, prob, vids = final_eval
    pred = prob.argmax(1)
    cm = np.zeros((len(names), len(names)), int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.imshow(cm, cmap="Blues")
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"{FINAL}, seed 42 split, last epoch\n"
                 f"window accuracy {(pred == y).mean():.1%}", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=150)

    per_video = []
    for v in dict.fromkeys(vids):
        m = vids == v
        p = prob[m].mean(0).argmax()
        per_video.append({"video": v, "true": names[y[m][0]], "pred": names[p],
                          "correct": names[p] == names[y[m][0]], "windows": int(m.sum())})
    pv = pd.DataFrame(per_video)
    pv.to_csv(OUT_DIR / "per_video.csv", index=False)
    print(f"\nper-video accuracy (final model, seed 42 split): {pv.correct.mean():.1%}")
    print(pv.to_string(index=False))
    print(f"\nsaved -> {OUT_DIR}")


if __name__ == "__main__":
    main()
