"""Manuscript numbers for the revised G1 paper, generated from the result files (2026-10-01).
Writes runs/g1_tables.json and prints a readable summary. Every number quoted in the manuscript comes from here."""
import json

import numpy as np

from sklearn.metrics import roc_auc_score

from g1_final_analysis import MODELS, R, SEEDS, path

COLS = ["any", "epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]


def ms(v, d=3):
    v = np.asarray(v, float)
    return f"{v.mean():.{d}f} ± {v.std(ddof=1):.{d}f}"


def main():
    fa = {agg: json.loads((R / f"g1_final_analysis_{agg}.json").read_text()) for agg in ("max", "top5mean")}
    rec = json.loads((R / "g1_recalibration.json").read_text())["results"]
    T = {}

    # Table: per-site performance, mean ± SD over seeds, both aggregations
    for agg, d in fa.items():
        for site in ("rsna", "cq500", "physionet"):
            for m in MODELS:
                rs = [d["per_seed"][str(s)]["sites"][site][m] for s in SEEDS]
                edh = [r["sub_sens@0.5"]["epidural"] for r in rs]
                T[f"perf/{agg}/{site}/{m}"] = {
                    "auroc": ms([r["auroc"]["any"] for r in rs]),
                    "sens": ms([r["sens_any@0.5"] for r in rs], 2), "spec": ms([r["spec_any@0.5"] for r in rs], 2),
                    "ece": ms([r["ece_any"] for r in rs]),
                    "edh_auroc": ms([r["auroc"]["epidural"] for r in rs]), "edh_detected": edh,
                    "subtype_auroc": {c: ms([r["auroc"][c] for r in rs]) for c in COLS[1:]},
                    "n_pos": rs[0]["positives"]}

    # Corrected H1 and operating-point deltas (max aggregation), per seed
    d = fa["max"]
    for site in ("cq500", "physionet"):
        for m in MODELS:
            h = [d["per_seed"][str(s)]["h1"][site][m] for s in SEEDS]
            T[f"h1/{site}/{m}"] = {"pooled": ms([x["pooled_auroc_rel_degradation"] for x in h]),
                                   "subtype": ms([x["subtype_mean_auroc_rel_degradation"] for x in h]),
                                   "diff_per_seed": [x["diff_subtype_minus_pooled"] for x in h],
                                   "p_per_seed": [x["p"] for x in h]}
            for agg in ("max", "top5mean"):
                o = [fa[agg]["per_seed"][str(s)]["operating_point"][site][m] for s in SEEDS]
                T[f"op/{agg}/{site}/{m}"] = {"d_sens": ms([x["delta_sens"] for x in o], 2),
                                             "d_spec": ms([x["delta_spec"] for x in o], 2),
                                             "p_spec_per_seed": [x["p_spec"] for x in o]}
    # H2 AUROC differences per seed
    for site in ("rsna", "cq500", "physionet"):
        for k in ("WICL-B4c", "WICL-B2", "B4c-B2"):
            h = [d["per_seed"][str(s)]["h2"][site][k] for s in SEEDS]
            T[f"h2/{site}/{k}"] = {"diff_per_seed": [x["diff"] for x in h], "p_per_seed": [x["p"] for x in h]}
    # H4 per seed
    T["h4"] = d["h4"]
    for s in SEEDS:
        f = R / ("baselines_bhsd_excluded/h4_4c_result.json" if s == 0 else f"g1_seeds/h4_seed{s}_result.json")
        T[f"h4_ci/seed{s}"] = {k: v["ci95"] for k, v in json.loads(f.read_text())["observed"].items()}

    # Recalibration: medians over seeds
    for site, ks in (("cq500", [10, 20, 50]), ("physionet", [10, 20])):
        for m in MODELS:
            rr = [rec[f"{site}/{m}/seed{s}"] for s in SEEDS]
            T[f"recal/{site}/{m}"] = {"fixed_spec": ms([r["fixed_0.5"][1] for r in rr], 2),
                                      "fixed_sens": ms([r["fixed_0.5"][0] for r in rr], 2),
                                      **{f"k{k}": {"sens": ms([r[f"k={k}"]["sens_median"] for r in rr], 2),
                                                   "spec": ms([r[f"k={k}"]["spec_median"] for r in rr], 2)}
                                         for k in ks}}

    # Evaluation artifacts
    slice_cq = np.load(R / "p4_rescore/B2__cq500.npz")

    T["artifact/cq500_slice_vs_study_B2_seed0"] = {
        "slice_auroc": round(roc_auc_score(slice_cq["y"][:, 0], slice_cq["prob"][:, 0]), 4),
        "study_auroc": d["per_seed"]["0"]["sites"]["cq500"]["B2"]["auroc"]["any"]}
    old = {}
    for m in ("B2", "B4c", "WICL"):
        z = np.load(R / f"p4_rescore/{m}__physionet.npz")
        ids, inv = np.unique(z["pid"], return_inverse=True)
        P = np.full(len(ids), -1.0)
        Y = np.zeros(len(ids))
        np.maximum.at(P, inv, z["prob"][:, 0])
        np.maximum.at(Y, inv, z["y"][:, 0])
        new = d["per_seed"]["0"]["sites"]["physionet"][m]
        old[m] = {"rotated_auroc": round(roc_auc_score(Y, P), 4), "rotated_spec": round(float((P[Y == 0] < .5).mean()), 4),
                  "rotated_slice_auroc": round(roc_auc_score(z["y"][:, 0], z["prob"][:, 0]), 4),
                  "fixed_auroc": new["auroc"]["any"], "fixed_spec": new["spec_any@0.5"]}
    T["artifact/physionet_orientation_seed0"] = old

    # Slice-level false-positive rate in negative studies by study length (B2, all seeds)
    fp = {}
    for site in ("rsna", "cq500", "physionet"):
        for s in SEEDS:

            z = np.load(path(site, "B2", s))
            ids, inv, cnt = np.unique(z["pid"], return_inverse=True, return_counts=True)
            ys = np.zeros(len(ids))
            np.maximum.at(ys, inv, z["y"][:, 0])
            neg, L, p = ys[inv] == 0, cnt[inv], z["prob"][:, 0]
            for lo, hi in ((0, 40), (40, 80), (80, 10000)):
                mm = neg & (L >= lo) & (L < hi)
                if mm.sum():
                    fp.setdefault(f"{site}/{lo}-{hi}", {"n_neg_studies": int(((cnt >= lo) & (cnt < hi) & (ys == 0)).sum()),
                                                        "fp_rate": []})["fp_rate"].append(float((p[mm] >= .5).mean()))
    T["fp_by_length_B2"] = {k: {"n_neg_studies": v["n_neg_studies"], "slice_fp_pct": ms(np.array(v["fp_rate"]) * 100, 2)}
                            for k, v in fp.items()}

    (R / "g1_tables.json").write_text(json.dumps(T, indent=1, default=str))
    for k, v in T.items():
        print(k, json.dumps(v, default=str))


if __name__ == "__main__":
    main()
