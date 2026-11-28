"""Figures for the Assignment 02 slides (run with the ai_ml_final env, from final_project/)."""
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd, cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))
import p1_3_analyze_postprocessing as a   # load_raw, create_height_corrected_target, SG params

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
LOG = Path(sys.argv[2])                     # p1_3 console log (per-file diagnostics)
CLIP = "hiphop_twostep/hiphop_twostep_1"
FRAME = 60

BLUE, BLUE2, ORANGE, GREY, RED = "#006CB8", "#7FB3DE", "#E07B39", "#8A8A8A", "#C0392B"
plt.rcParams.update({"font.family": "Malgun Gothic", "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 12,
                     "axes.edgecolor": "#555555", "axes.labelcolor": "#333333",
                     "xtick.color": "#555555", "ytick.color": "#555555", "axes.unicode_minus": False})

d = a.load_raw(ROOT / "data" / "raw_keypoints" / f"{CLIP}.npz")
W, fps = d["world"], d["fps"]
pel = (W[:, 23] + W[:, 24]) / 2
leg = np.median(np.linalg.norm((W[:, 27] + W[:, 28]) / 2 - pel, axis=1))
t = np.arange(len(W)) / fps
BONES = [(11, 12), (11, 23), (12, 24), (23, 24), (11, 13), (13, 15), (12, 14), (14, 16),
         (23, 25), (25, 27), (27, 29), (29, 31), (27, 31), (24, 26), (26, 28), (28, 30), (30, 32), (28, 32)]
LOWER = set(range(23, 33))

# ---------------------------------------------------------------- 1. acquisition
FIG1_CLIP, FIG1_FRAME = "charleston/charleston_7", 60       # example frame shown on slide 1
d1 = a.load_raw(ROOT / "data" / "raw_keypoints" / f"{FIG1_CLIP}.npz")
cap = cv2.VideoCapture(str(ROOT / "data" / "raw_videos" / f"{FIG1_CLIP}.mp4"))
cap.set(cv2.CAP_PROP_POS_FRAMES, int(d1["frames"][FIG1_FRAME]))
ok, img = cap.read()
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
H = img.shape[0]
px = d1["image"][FIG1_FRAME].copy(); px[:, 1] = -px[:, 1]    # back to y down
fig = plt.figure(figsize=(7.6, 5.2))
ax = fig.add_subplot(1, 2, 1)
x0, x1 = max(0, px[:, 0].min() - 160), min(img.shape[1], px[:, 0].max() + 160)
y0, y1 = max(0, px[:, 1].min() - 160), min(H, px[:, 1].max() + 60)
ax.imshow(img[int(y0):int(y1), int(x0):int(x1)])
for i, j in BONES:
    c = ORANGE if (i in LOWER and j in LOWER) else "white"
    ax.plot([px[i, 0] - x0, px[j, 0] - x0], [px[i, 1] - y0, px[j, 1] - y0], c=c, lw=2.2)
ax.scatter(px[:, 0] - x0, px[:, 1] - y0, s=9, c="white", zorder=3)
ax.set_title("영상 + BlazePose 2D 관절"); ax.axis("off")
ax = fig.add_subplot(1, 2, 2, projection="3d")
P = d1["world"][FIG1_FRAME]
for i, j in BONES:
    c = ORANGE if (i in LOWER and j in LOWER) else BLUE
    ax.plot([P[i, 0], P[j, 0]], [P[i, 2], P[j, 2]], [P[i, 1], P[j, 1]], c=c, lw=2.2)
ax.scatter([0], [0], [0], c=RED, s=40, zorder=5)
ax.set_xlabel("x"); ax.set_ylabel("z (깊이)"); ax.set_zlabel("y (위)")
ax.set_box_aspect((1, 1, 1.75)); ax.view_init(elev=12, azim=-72)
ax.set_xlim(-0.45, 0.45); ax.set_ylim(-0.45, 0.45); ax.set_zlim(-0.95, 0.65)
ax.set_title("3D world 좌표 (m)\n빨간 점 = 원점 = 골반 중심")
fig.tight_layout(); fig.savefig(OUT / "fig1_acquisition.png", dpi=220, bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- 2. problems
pairs = json.loads((ROOT / "data" / "mirror_pairs.json").read_text(encoding="utf-8"))
txt = LOG.read_text(encoding="utf-8", errors="ignore")
blocks = re.findall(r"=== (\S+) .*?visibility < 0.5: ([\d.]+)%", txt, re.S)
vis = sorted((float(v) for k, v in blocks if k not in pairs), reverse=True)

fig, axs = plt.subplots(1, 3, figsize=(13.2, 3.7))
img_leg = np.median(np.linalg.norm((d["image"][:, 27] + d["image"][:, 28]) / 2
                                   - (d["image"][:, 23] + d["image"][:, 24]) / 2, axis=1))
ipel = (d["image"][:, 23] + d["image"][:, 24]) / 2
ax = axs[0]
ax.plot(t, pel[:, 1] / leg, c=BLUE, lw=2, label="3D world 골반 높이")
ax.plot(t, (ipel[:, 1] - ipel[0, 1]) / img_leg, c=ORANGE, lw=1.6, label="2D 이미지 골반 높이")
ax.set_title("A. 골반이 원점에 고정: 바닥, 이동 정보 없음")
ax.set_xlabel("시간 (초)"); ax.set_ylabel("다리 길이 단위"); ax.legend(fontsize=9, frameon=False, loc="upper right")
ax = axs[1]
m = (t >= 1.0) & (t <= 2.5)
raw_y = W[:, 31, 1] / leg
sm_y = savgol_filter(W[:, 31, 1], a.SG_WINDOW, a.SG_POLY) / leg
ax.plot(t[m], raw_y[m], c=GREY, lw=1.4, marker="o", ms=3, label="raw 왼발끝 y")
ax.plot(t[m], sm_y[m], c=BLUE, lw=2, label="추세 (S-G 근사)")
ax.set_title("B. 프레임 간 떨림 (1.5초 확대)")
ax.set_xlabel("시간 (초)"); ax.set_ylabel("다리 길이 단위"); ax.legend(fontsize=9, frameon=False)
ax = axs[2]
cols = [RED if v >= 50 else BLUE2 for v in vis]
ax.bar(range(len(vis)), vis, color=cols, width=0.85)
ax.axhline(50, c="#555555", lw=0.8, ls="--")
ax.set_title("C. 영상별 발 좌표 추출이 어려운 프레임 비율")
ax.set_xlabel("영상 (정렬)"); ax.set_ylabel("발 가시성 < 0.5 프레임 비율 (%)"); ax.set_xticks([])
fig.tight_layout(); fig.savefig(OUT / "fig2_problems.png", dpi=220, bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- 3. post-processing
corr = a.create_height_corrected_target(W)
imp, _ = a.improved_postprocess(d)
filled, _, _ = a.fill_gaps(d)


def lowest_foot(P):
    return np.minimum(np.minimum(P[:, 29, 1], P[:, 31, 1]), np.minimum(P[:, 30, 1], P[:, 32, 1]))


I2 = d["image"]
leg_px = np.median(np.linalg.norm((I2[:, 23] + I2[:, 24]) / 2 - (I2[:, 27] + I2[:, 28]) / 2, axis=1))
low2d = lowest_foot(np.concatenate([I2, np.zeros((len(I2), 33, 1))], axis=2)[:, :, [0, 1, 2]])
ref2d = (low2d - np.percentile(low2d, 5)) / leg_px

fig, axs = plt.subplots(1, 3, figsize=(13.2, 3.7))
ax = axs[0]
ax.plot(t, lowest_foot(corr) / leg, c=ORANGE, lw=1.8, label="① 높이 보정")
ax.plot(t, ref2d, c="#333333", lw=1.2, ls="--", label="2D 영상 기준")
ax.plot(t, lowest_foot(imp) / leg, c=BLUE, lw=2.2, label="②~④ 적용")
ax.set_title("A 해결 (③): 점프가 남는다 (더 낮은 발의 높이)")
ax.set_xlabel("시간 (초)"); ax.set_ylabel("다리 길이 단위"); ax.legend(fontsize=9, frameon=False, loc="upper right")
ax.set_ylim(top=ax.get_ylim()[1] * 1.25)
ax = axs[1]
m = (t >= 1.0) & (t <= 2.5)
sg_only = savgol_filter(filled, a.SG_WINDOW, a.SG_POLY, axis=0)
ax.plot(t[m], corr[m, 31, 1] / leg, c=ORANGE, lw=1.4, marker="o", ms=3, label="① 높이 보정")
ax.plot(t[m], (sg_only[m, 31, 1] - (W[m, 31, 1] - corr[m, 31, 1])) / leg, c=BLUE, lw=2.2, label="+ ④ 평활화")
ax.set_title("B 해결 (④): 떨림 제거 (1.5초 확대)")
ax.set_xlabel("시간 (초)"); ax.set_ylabel("다리 길이 단위"); ax.legend(fontsize=9, frameon=False)
ax = axs[2]
CLIP_C, J_C = "charleston/charleston_12", 31
dc = a.load_raw(ROOT / "data" / "raw_keypoints" / f"{CLIP_C}.npz")
Wc = dc["world"]
legc = np.median(np.linalg.norm((Wc[:, 23] + Wc[:, 24]) / 2 - (Wc[:, 27] + Wc[:, 28]) / 2, axis=1))
filled_c, _, _ = a.fill_gaps(dc)
tc = np.arange(len(Wc)) / dc["fps"]
low = dc["vis"][:, J_C] < a.VIS_MIN
fix = a.gap_runs(low, a.MAX_GAP)
t = tc
for s_, e_ in zip(*[np.flatnonzero(np.diff(np.r_[0, x.astype(int), 0]) == v) for x, v in ((low & ~fix, 1), (low & ~fix, -1))]):
    ax.axvspan(t[s_], t[min(e_, len(t) - 1)], color=RED, alpha=0.12, lw=0)
for s_, e_ in zip(*[np.flatnonzero(np.diff(np.r_[0, x.astype(int), 0]) == v) for x, v in ((fix, 1), (fix, -1))]):
    ax.axvspan(t[s_], t[min(e_, len(t) - 1)], color=BLUE, alpha=0.15, lw=0)
ax.plot(t, Wc[:, J_C, 1] / legc, c=GREY, lw=1.2, label="raw")
ax.plot(t, filled_c[:, J_C, 1] / legc, c=BLUE, lw=1.6, label="② 짧은 구간 보간")
ax.set_title("C 해결 (②): 파랑 = 보간, 빨강 = 긴 구간(그대로)")
ax.set_xlabel("시간 (초)"); ax.set_ylabel("왼발끝 y (다리 길이)")
y0, y1 = ax.get_ylim(); ax.set_ylim(y0 - 0.22 * (y1 - y0), y1)
ax.legend(fontsize=9, frameon=False, loc="lower center", ncol=2)
fig.tight_layout(); fig.savefig(OUT / "fig3_postprocessing.png", dpi=220, bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- 4. numbers for the results slide
dfm = pd.read_csv(ROOT / "data" / "output" / "all_metrics.csv")
dfm = dfm[~(dfm.step + "/" + dfm.file).isin(pairs)]
cols = ["both_feet_off_p95", "foot_lift_p95", "jitter_lower", "ankle_jerk_rms", "hip_y_range_world"]
med = dfm.groupby("condition")[cols].median()
print(med.round(4).to_string())
med.to_csv(OUT / "metrics_median.csv")
print("n videos:", dfm.file.nunique(), " foot vis >=50%:", sum(v >= 50 for v in vis), "/", len(vis))
