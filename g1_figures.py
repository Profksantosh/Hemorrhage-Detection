"""Figures for the revised G1 manuscript, drawn from runs/g1_final_analysis_*.json and saved predictions (2026-10-01)."""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("runs")
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "../Manuscript_G1_Revised/figures")
OUT.mkdir(parents=True, exist_ok=True)
MODELS = ["B2", "B4c", "WICL"]
LABEL = {"B2": "Fixed windows", "B4c": "Random window", "WICL": "WICL"}
TICK = {"B2": "Fixed\nwindows", "B4c": "Random\nwindow", "WICL": "WICL"}
COLOR = {"B2": "#4c72b0", "B4c": "#dd8452", "WICL": "#55a868"}
plt.rcParams.update({"font.family": "serif", "font.size": 9})
fa = {a: json.loads((R / f"g1_final_analysis_{a}.json").read_text()) for a in ("max", "top5mean")}


def study(npz):
    z = np.load(npz)
    ids, inv = np.unique(z["pid"], return_inverse=True)
    P = np.full(len(ids), -1.0)
    Y = np.zeros(len(ids))
    np.maximum.at(P, inv, z["prob"][:, 0])
    np.maximum.at(Y, inv, z["y"][:, 0])
    return z, P, Y


# ---------------------------------------------------------------- Figure 1: evaluation artifacts (seed 0)
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.1))
x = np.arange(len(MODELS))
slice_auc, study_auc = [], []
for m in MODELS:
    z, P, Y = study(R / f"p4_rescore/{m}__cq500.npz")
    slice_auc.append(roc_auc_score(z["y"][:, 0], z["prob"][:, 0]))
    study_auc.append(roc_auc_score(Y, P))
ax[0].bar(x - 0.18, slice_auc, 0.36, color="#bbbbbb", label="Slice level, study label copied to slices")
ax[0].bar(x + 0.18, study_auc, 0.36, color=[COLOR[m] for m in MODELS], label="Study level (correct)")
ax[0].set_xticks(x, [TICK[m] for m in MODELS])
ax[0].set_ylim(0.5, 1.12); ax[0].set_yticks([0.5,0.6,0.7,0.8,0.9,1.0])
ax[0].set_ylabel("Any-ICH AUROC")
ax[0].set_title("A. CQ500: label granularity", fontsize=9)
ax[0].legend(fontsize=6.5, loc="upper center", frameon=False, ncol=1)
for i, (a, b) in enumerate(zip(slice_auc, study_auc)):
    ax[0].text(i - 0.18, a + 0.005, f"{a:.2f}", ha="center", fontsize=7)
    ax[0].text(i + 0.18, b + 0.005, f"{b:.2f}", ha="center", fontsize=7)

rot, up = [], []
for m in MODELS:
    _, P, Y = study(R / f"p4_rescore/{m}__physionet.npz")
    rot.append((roc_auc_score(Y, P), (P[Y == 0] < .5).mean()))
    _, P, Y = study(R / f"g1_physionet_fixed/{m}__physionet.npz")
    up.append((roc_auc_score(Y, P), (P[Y == 0] < .5).mean()))
for i, m in enumerate(MODELS):
    ax[1].annotate("", xy=(up[i][1], up[i][0]), xytext=(rot[i][1], rot[i][0]),
                   arrowprops=dict(arrowstyle="->", color=COLOR[m], lw=1.4))
    ax[1].plot(*rot[i][::-1], "o", mfc="white", mec=COLOR[m])
    ax[1].plot(*up[i][::-1], "o", color=COLOR[m], label=LABEL[m])
ax[1].set_xlabel("Specificity at RSNA-locked threshold")
ax[1].set_ylabel("Any-ICH AUROC (per scan)")
ax[1].set_xlim(-0.03, 1.0)
ax[1].set_ylim(0.7, 1.0)
ax[1].set_title("B. PhysioNet-ICH: rotated (open) to upright (filled)", fontsize=9)
ax[1].legend(fontsize=7, frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "fig1_evaluation_artifacts.png", dpi=300)
plt.close(fig)

# ---------------------------------------------------------------- Figure 2: operating point by site
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
sites = ["rsna", "cq500", "physionet"]
names = ["RSNA\n(internal)", "CQ500", "PhysioNet-ICH"]
for k, agg in enumerate(("max", "top5mean")):
    for j, m in enumerate(MODELS):
        for i, site in enumerate(sites):
            sp = [fa[agg]["per_seed"][str(s)]["sites"][site][m]["spec_any@0.5"] for s in (0, 1, 2)]
            se = [fa[agg]["per_seed"][str(s)]["sites"][site][m]["sens_any@0.5"] for s in (0, 1, 2)]
            xs = i + (j - 1) * 0.22
            ax[k].scatter([xs] * 3, sp, color=COLOR[m], s=14, label=LABEL[m] if i == 0 else None)
            ax[k].scatter([xs] * 3, se, color=COLOR[m], s=14, marker="^", alpha=0.45)
            ax[k].plot([xs - 0.08, xs + 0.08], [np.mean(sp)] * 2, color=COLOR[m], lw=1.5)
    ax[k].set_xticks(range(3), names)
    ax[k].set_ylim(0, 1.03)
    ax[k].set_title("A. Scan score = max over slices" if agg == "max" else "B. Scan score = mean of top-5 slices",
                    fontsize=9)
ax[0].set_ylabel("Specificity (o) / sensitivity (^)")
ax[0].legend(fontsize=7, frameon=False, loc="lower left")
fig.tight_layout()
fig.savefig(OUT / "fig2_operating_point.png", dpi=300)
plt.close(fig)
print("saved", OUT)
