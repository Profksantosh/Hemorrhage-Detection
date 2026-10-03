"""Figures for the revised G1 manuscript, drawn from runs/g1_final_analysis_*.json and saved predictions.
Data and computations are unchanged from the first revision; only labels, legends, typography and layout were revised (2026-10-03)."""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R = Path("runs")
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "../Manuscript_G1_Revised/figures")
OUT.mkdir(parents=True, exist_ok=True)
MODELS = ["B2", "B4c", "WICL"]
LABEL = {"B2": "Fixed windows (B2)", "B4c": "Random window (RW)", "WICL": "WICL"}
TICK = {"B2": "Fixed\nwindows\n(B2)", "B4c": "Random\nwindow\n(RW)", "WICL": "WICL"}
COLOR = {"B2": "#2f5f9e", "B4c": "#d2691e", "WICL": "#2e8b57"}
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10.5,
                     "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "axes.spines.top": False,
                     "axes.spines.right": False})
fa = {a: json.loads((R / f"g1_final_analysis_{a}.json").read_text()) for a in ("max", "top5mean")}


def study(npz):
    z = np.load(npz)
    ids, inv = np.unique(z["pid"], return_inverse=True)
    P = np.full(len(ids), -1.0)
    Y = np.zeros(len(ids))
    np.maximum.at(P, inv, z["prob"][:, 0])
    np.maximum.at(Y, inv, z["y"][:, 0])
    return z, P, Y


# ------------------------------------------------ Figure 1: evaluation pitfalls (seed-0 models)
fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.9), gridspec_kw={"width_ratios": [1, 1.05]})
x = np.arange(len(MODELS))
slice_auc, study_auc = [], []
for m in MODELS:
    z, P, Y = study(R / f"p4_rescore/{m}__cq500.npz")
    slice_auc.append(roc_auc_score(z["y"][:, 0], z["prob"][:, 0]))
    study_auc.append(roc_auc_score(Y, P))
b1 = ax[0].bar(x - 0.19, slice_auc, 0.36, color="#b8b8b8", edgecolor="#555555", linewidth=0.6)
b2 = ax[0].bar(x + 0.19, study_auc, 0.36, color=[COLOR[m] for m in MODELS], edgecolor="#222222", linewidth=0.6)
ax[0].set_xticks(x, [TICK[m] for m in MODELS])
ax[0].set_ylim(0.5, 1.0)
ax[0].set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax[0].yaxis.grid(True, color="#dddddd", lw=0.6)
ax[0].set_axisbelow(True)
ax[0].set_ylabel("Any-ICH AUROC")
ax[0].set_title("A. CQ500: evaluation unit", loc="left")
for i, (a, b) in enumerate(zip(slice_auc, study_auc)):
    ax[0].text(i - 0.19, a + 0.008, f"{a:.2f}", ha="center", fontsize=9)
    ax[0].text(i + 0.19, b + 0.008, f"{b:.2f}", ha="center", fontsize=9)
ax[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, fc="#b8b8b8", ec="#555555"),
                      plt.Rectangle((0, 0), 1, 1, fc="#8aa4c8", ec="#222222")],
             labels=["Per slice (study label copied to every slice)", "Per study (correct unit)"],
             fontsize=8.2, loc="lower center", bbox_to_anchor=(0.5, -0.46), frameon=False, ncol=1)

rot, up = [], []
for m in MODELS:
    _, P, Y = study(R / f"p4_rescore/{m}__physionet.npz")
    rot.append((roc_auc_score(Y, P), (P[Y == 0] < .5).mean()))
    _, P, Y = study(R / f"g1_physionet_fixed/{m}__physionet.npz")
    up.append((roc_auc_score(Y, P), (P[Y == 0] < .5).mean()))
for i, m in enumerate(MODELS):
    ax[1].annotate("", xy=(up[i][1], up[i][0]), xytext=(rot[i][1], rot[i][0]),
                   arrowprops=dict(arrowstyle="-|>", color=COLOR[m], lw=1.4, shrinkA=4, shrinkB=4))
    ax[1].plot(*rot[i][::-1], "o", ms=7, mfc="white", mec=COLOR[m], mew=1.6)
    ax[1].plot(*up[i][::-1], "o", ms=7, color=COLOR[m])
ax[1].set_xlabel("Specificity at fixed threshold (0.5)")
ax[1].set_ylabel("Any-ICH AUROC (per scan)")
ax[1].set_xlim(-0.03, 1.0)
ax[1].set_ylim(0.7, 1.0)
ax[1].grid(True, color="#dddddd", lw=0.6)
ax[1].set_axisbelow(True)
ax[1].set_title("B. PhysioNet-ICH: image orientation", loc="left")
handles = [Line2D([], [], marker="o", ls="", ms=7, color=COLOR[m], label=LABEL[m]) for m in MODELS]
handles += [Line2D([], [], marker="o", ls="", ms=7, mfc="white", mec="#444444", mew=1.6, label="Read without rotation"),
            Line2D([], [], marker="o", ls="", ms=7, color="#444444", label="Rotated to radiological orientation")]
ax[1].legend(handles=handles, fontsize=8.2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.62), ncol=1)
fig.tight_layout()
fig.subplots_adjust(bottom=0.34, wspace=0.28)
fig.savefig(OUT / "fig1_evaluation_artifacts.png", dpi=300)
plt.close(fig)

# ------------------------------------------------ Figure 2: operating point by site (three seeds per model)
fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.7), sharey=True)
sites = ["rsna", "cq500", "physionet"]
names = ["RSNA\n(internal)", "CQ500", "PhysioNet-ICH"]
for k, agg in enumerate(("max", "top5mean")):
    for j, m in enumerate(MODELS):
        for i, site in enumerate(sites):
            sp = [fa[agg]["per_seed"][str(s)]["sites"][site][m]["spec_any@0.5"] for s in (0, 1, 2)]
            se = [fa[agg]["per_seed"][str(s)]["sites"][site][m]["sens_any@0.5"] for s in (0, 1, 2)]
            xs = i + (j - 1) * 0.24
            ax[k].scatter([xs] * 3, sp, color=COLOR[m], s=20, zorder=3)
            ax[k].scatter([xs] * 3, se, color=COLOR[m], s=22, marker="^", alpha=0.6, zorder=2)
            ax[k].plot([xs - 0.09, xs + 0.09], [np.mean(sp)] * 2, color=COLOR[m], lw=1.8, zorder=4)
    ax[k].set_xticks(range(3), names)
    ax[k].set_ylim(0, 1.03)
    ax[k].yaxis.grid(True, color="#dddddd", lw=0.6)
    ax[k].set_axisbelow(True)
    ax[k].set_title("A. Scan score: maximum slice probability" if agg == "max"
                    else "B. Scan score: mean of five highest slices", loc="left", fontsize=10)
ax[0].set_ylabel("Specificity / sensitivity at 0.5")
handles = [Line2D([], [], marker="s", ls="", ms=7, color=COLOR[m], label=LABEL[m]) for m in MODELS]
handles += [Line2D([], [], marker="o", ls="", ms=6, color="#444444", label="Specificity (one point per seed)"),
            Line2D([], [], marker="^", ls="", ms=6, color="#444444", alpha=0.6, label="Sensitivity (one point per seed)"),
            Line2D([], [], color="#444444", lw=1.8, label="Mean specificity")]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.5, frameon=False)
fig.tight_layout()
fig.subplots_adjust(bottom=0.30, wspace=0.08)
fig.savefig(OUT / "fig2_operating_point.png", dpi=300)
plt.close(fig)
print("saved", OUT)
