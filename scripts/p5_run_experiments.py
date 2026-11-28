"""
scripts/p5_run_experiments.py

Reproduces the experiment table and figures of the report from data/raw_keypoints/.

Evaluation = 5-fold cross-validation by video:
  - a source video and its mirrored copy form one group (never split between train and val)
  - inside each step, the groups are dealt into N_FOLDS folds; every fold is the validation set once,
    so every video is tested exactly once
  - scores are per VIDEO first, then averaged, so a 48 s video counts as much as a 3 s one:
      video_window_acc  share of a video's 1 s windows predicted correctly, averaged over videos
      video_vote_acc    share of videos whose averaged prediction (as in p4) is correct
      class_balanced    video_window_acc averaged per step first, then over the 5 steps
  - all numbers are at the last epoch (no choice of epoch on validation data)

Run from final_project/ :
    python scripts/p5_run_experiments.py
Outputs (docs/figures/):
    experiments.csv         one row per configuration (pooled scores + spread over folds)
    experiments.png         bar chart of all configurations
    confusion_matrix.png    final model, every video once (video-level vote)
    per_video.csv           final model: true / predicted step and window accuracy of every video
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
from features import make_features, resample_fps, video_fps, load_root
from p3_train_mlp import build_model, class_weights, BATCH_SIZE, EPOCHS, LEARNING_RATE

ROOT = Path(__file__).resolve().parent.parent
IN_DIR = ROOT / "data" / "raw_keypoints"
OUT_DIR = ROOT / "docs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
N_FOLDS = 5
FOLD_SEED = 0
STRIDE = 15

# (name, experiment group, model, joints, normalize, velocity, window frames, balance, pelvis motion)
# balance = class weights in the loss + at most p2.MAX_WINDOWS_PER_VIDEO training windows per video
# pelvis motion = heights from the floor + whole-body travel from the 2D track (features.ROOT_MOTION)
CONFIGS = [
    ("MLP | none",               "3 preprocessing", "mlp", "lower", "none",  False, 30, False, False),
    ("MLP | video-align",        "3 preprocessing", "mlp", "lower", "video", False, 30, False, False),
    ("MLP | frame-align",        "3 preprocessing", "mlp", "lower", "frame", False, 30, False, False),
    ("MLP | video-align | 2s",   "window",          "mlp", "lower", "video", False, 60, False, False),
    ("MLP | video-align | full", "2 lower vs full", "mlp", "full",  "video", False, 30, False, False),
    ("MLP | +velocity",          "feature",         "mlp", "lower", "video", True,  30, False, False),
    ("CNN | position",           "1 model",         "cnn", "lower", "video", False, 30, False, False),
    ("CNN | +velocity",          "1 model",         "cnn", "lower", "video", True,  30, False, False),
    ("CNN | +velocity +balance (final)", "imbalance", "cnn", "lower", "video", True, 30, True, False),
    ("CNN | +velocity +balance +pelvis motion", "pelvis", "cnn", "lower", "video", True, 30, True, True),
]
FINAL = "CNN | +velocity +balance (final)"

class_dirs = sorted(p for p in IN_DIR.iterdir() if p.is_dir())
labels = {d.name: i for i, d in enumerate(class_dirs)}
names = list(labels)
raw = {d.name: {p: resample_fps(p2.load_raw_keypoints(p), video_fps(p))
                for p in sorted(d.glob("*.npz"))} for d in class_dirs}
roots = {k: {p: load_root(p, m, True) for p, m in v.items()} for k, v in raw.items()}


def make_folds(n_folds=N_FOLDS, seed=FOLD_SEED):
    """{path: fold}. Groups = source video + its mirror; dealt round-robin per step after a shuffle."""
    rng = np.random.default_rng(seed)
    fold_of = {}
    for d in class_dirs:
        groups = {}
        for p in raw[d.name]:
            groups.setdefault(p2.source_of(p), []).append(p)
        keys = sorted(groups)
        rng.shuffle(keys)
        for i, k in enumerate(keys):
            for p in groups[k]:
                fold_of[p] = i % n_folds
    return fold_of


FOLD_OF = make_folds()


def windows(f, L):
    return [f[s:s + L].reshape(L, -1) for s in range(0, len(f) - L + 1, STRIDE)]


def build(fold, joints, normalize, velocity, L, balance=False, pelvis=False):
    cap_rng = np.random.default_rng(fold + 1)
    data = {"train": ([], [], []), "val": ([], [], [])}
    for d in class_dirs:
        for p in raw[d.name]:
            split = "val" if FOLD_OF[p] == fold else "train"
            root = roots[d.name][p] if pelvis else None
            w = windows(make_features(raw[d.name][p], joints, normalize, velocity, root=root), L)
            if balance and split == "train":
                w = p2.cap_windows(w, cap_rng)
            X, y, v = data[split]
            X += w
            y += [labels[d.name]] * len(w)
            v += [f"{d.name}/{p.stem}"] * len(w)
    out = {}
    for split, (X, y, v) in data.items():
        out[split] = (torch.tensor(np.asarray(X), dtype=torch.float32),
                      torch.tensor(y, dtype=torch.long), np.asarray(v))
    return out


def train(model_type, data, seed, balance=False):
    """train for EPOCHS, return softmax probabilities on the validation windows (last epoch)"""
    torch.manual_seed(seed)
    Xtr, ytr, _ = data["train"]
    Xva, _, _ = data["val"]
    model = build_model(model_type, tuple(Xtr.shape[1:]), len(labels))
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    ce = nn.CrossEntropyLoss(weight=class_weights(ytr, len(labels)) if balance else None)
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
        return torch.softmax(model(Xva), 1).numpy()


def per_video_rows(y, prob, vids, fold):
    rows = []
    for v in dict.fromkeys(vids):
        m = vids == v
        true = int(y[m][0])
        rows.append({"video": v, "fold": fold, "true": names[true],
                     "pred": names[int(prob[m].mean(0).argmax())],
                     "window_acc": float((prob[m].argmax(1) == true).mean()),
                     "windows": int(m.sum())})
    pv = pd.DataFrame(rows)
    pv["correct"] = pv.true == pv.pred
    return pv


def scores(pv):
    return {"video_window_acc": pv.window_acc.mean(),
            "video_vote_acc": pv.correct.mean(),
            "class_balanced": pv.groupby("true").window_acc.mean().mean()}


def main():
    sizes = pd.crosstab([p.parent.name for p in FOLD_OF], [f for f in FOLD_OF.values()])
    print(f"{N_FOLDS}-fold CV, validation files per step (rows) and fold (columns):\n{sizes}\n")
    rows, final_pv = [], None
    for name, group, model_type, joints, norm, vel, L, bal, pel in CONFIGS:
        pvs, fold_acc = [], []
        for fold in range(N_FOLDS):
            data = build(fold, joints, norm, vel, L, bal, pel)
            prob = train(model_type, data, fold, bal)
            pv = per_video_rows(data["val"][1].numpy(), prob, data["val"][2], fold)
            pvs.append(pv)
            fold_acc.append(pv.window_acc.mean())
        pv = pd.concat(pvs, ignore_index=True)           # every video exactly once
        s = scores(pv)
        rows.append({"config": name, "experiment": group, **s,
                     "fold_std": float(np.std(fold_acc)),
                     "per_fold": [round(x, 3) for x in fold_acc]})
        if name == FINAL:
            final_pv = pv
        print(f"{name:42s} video-window {s['video_window_acc']:.3f} (folds +-{np.std(fold_acc):.3f}) | "
              f"video-vote {s['video_vote_acc']:.3f} | class-balanced {s['class_balanced']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "experiments.csv", index=False)

    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    x = np.arange(len(df))
    ax.bar(x - 0.2, df.video_window_acc, 0.4, yerr=df.fold_std, capsize=3,
           label="per-video window accuracy (+- std over folds)", color="#4C72B0")
    ax.bar(x + 0.2, df.video_vote_acc, 0.4, label="per-video vote accuracy", color="#DD8452")
    ax.axhline(1 / len(labels), color="gray", ls="--", lw=1, label="chance (20%)")
    ax.set_xticks(x, df.config, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("validation accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(f"{N_FOLDS}-fold cross-validation by video, last epoch (every video tested once)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "experiments.png", dpi=150)

    cm = np.zeros((len(names), len(names)), int)
    for t, p in zip(final_pv.true, final_pv.pred):
        cm[labels[t], labels[p]] += 1
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
    ax.set_title(f"{FINAL}\n{N_FOLDS}-fold CV, videos (each once): vote accuracy {final_pv.correct.mean():.1%}",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=150)

    final_pv.to_csv(OUT_DIR / "per_video.csv", index=False)
    print(f"\nfinal model, wrong videos:\n{final_pv[~final_pv.correct].to_string(index=False)}")
    print(f"\nsaved -> {OUT_DIR}")


if __name__ == "__main__":
    main()
