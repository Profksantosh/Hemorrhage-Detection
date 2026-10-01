"""G1 revision: final analysis at the correct unit (patient/study), all seeds available (2026-10-01).

Inputs (per seed s, model m in B2/B4c/WICL):
  RSNA internal per-patient : runs/g1_rsna_patient/{m}_seed{s}.npz
  CQ500                     : seed 0 runs/p4_rescore/{m}__cq500.npz, seeds 1-2 runs/g1_seeds_ext/{m}_seed{s}__cq500.npz
  PhysioNet (reoriented)    : runs/g1_physionet_fixed/ (see g1_physionet_reoriented_infer.py)
  H4 HU-corner              : seed 0 runs/baselines_bhsd_excluded/h4_4c_result.json, seeds 1-2 runs/g1_seeds/h4_seed{s}_result.json
Score of a patient/study = max over its slices (primary) or mean of its 5 highest slice scores (secondary);
label = max slice label.

Analyses
  A. Per-site discrimination and operating point (AUROC any + subtypes, sens/spec at the RSNA-locked 0.5).
  B. Corrected H1 (same unit, same threshold-free metric): relative degradation of pooled any-ICH AUROC versus the
     mean relative degradation of the five subtype AUROCs, RSNA -> CQ500 (and PhysioNet, exploratory).
  C. Operating-point transfer: change in sensitivity and specificity at 0.5, RSNA -> external.
  D. H2 across seeds: WICL - B4c and WICL - B2 study-level AUROC on each external site.
  E. H4 across seeds: mean corner degradation for B2, B4c, WICL.
Bootstrap: internal patients and external studies resampled independently (2,000 replicates, seed 0).
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

COLS = ["any", "epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]
SUB = list(range(1, 6))
MODELS = ["B2", "B4c", "WICL"]
SEEDS = [0, 1, 2]
SITES = ["cq500", "physionet"]
N_BOOT = 2000
R = Path("runs")


def path(site, m, s):
    if site == "rsna":
        return R / f"g1_rsna_patient/{m}_seed{s}.npz"
    if site == "physionet":
        return R / (f"g1_physionet_fixed/{m}__physionet.npz" if s == 0 else f"g1_physionet_fixed/{m}_seed{s}__physionet.npz")
    return R / (f"p4_rescore/{m}__{site}.npz" if s == 0 else f"g1_seeds_ext/{m}_seed{s}__{site}.npz")


def load(site, m, s, agg):
    p = path(site, m, s)
    if not p.exists():
        return None
    d = np.load(p)
    ids, inv = np.unique(d["pid"], return_inverse=True)
    Y = np.zeros((len(ids), 6))
    np.maximum.at(Y, inv, d["y"])
    if agg == "max":
        P = np.full((len(ids), 6), -np.inf)
        np.maximum.at(P, inv, d["prob"])
    else:
        order = np.argsort(inv, kind="stable")
        bounds = np.r_[0, np.cumsum(np.bincount(inv))]
        prob = d["prob"][order]
        P = np.stack([np.sort(prob[a:b], axis=0)[-5:].mean(axis=0) for a, b in zip(bounds[:-1], bounds[1:])])
    return P, Y


def auc(y, s):
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan


def aucs(P, Y, idx=None):
    if idx is not None:
        P, Y = P[idx], Y[idx]
    return np.array([auc(Y[:, j], P[:, j]) for j in range(6)])


def sens_spec(P, Y, j=0, thr=0.5, idx=None):
    if idx is not None:
        P, Y = P[idx], Y[idx]
    pos, neg = Y[:, j] == 1, Y[:, j] == 0
    return ((P[pos, j] >= thr).mean() if pos.any() else np.nan,
            (P[neg, j] < thr).mean() if neg.any() else np.nan)


def ece(y, p, bins=15):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def ci(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]


def p2(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    return round(float(min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean()))), 4)


def rel(a, b):
    return (a - b) / a


def main(agg):
    rng = np.random.default_rng(0)
    out = {"protocol": __doc__, "aggregation": agg, "per_seed": {}, "h4": {}}
    for s in SEEDS:
        data = {site: {m: load(site, m, s, agg) for m in MODELS} for site in ["rsna"] + SITES}
        if any(v is None for site in data.values() for v in site.values()):
            print(f"seed {s}: incomplete, skipped")
            continue
        n = {site: len(data[site]["B2"][1]) for site in data}
        boots = {site: [rng.integers(0, n[site], n[site]) for _ in range(N_BOOT)] for site in data}
        so = {"n": n, "sites": {}, "h1": {}, "operating_point": {}, "h2": {}}

        for site in data:
            so["sites"][site] = {}
            for m in MODELS:
                P, Y = data[site][m]
                a = aucs(P, Y)
                se, sp = sens_spec(P, Y)
                so["sites"][site][m] = {
                    "auroc": dict(zip(COLS, np.round(a, 4).tolist())),
                    "auroc_any_ci": ci([auc(Y[b, 0], P[b, 0]) for b in boots[site]]),
                    "sens_any@0.5": round(float(se), 4), "spec_any@0.5": round(float(sp), 4),
                    "ece_any": round(ece(Y[:, 0], P[:, 0]), 4),
                    "ece_any_ci": ci([ece(Y[b, 0], P[b, 0]) for b in boots[site]]),
                    "prevalence_any": round(float(Y[:, 0].mean()), 4),
                    "mean_score_any": round(float(P[:, 0].mean()), 4),
                    "sub_sens@0.5": {COLS[j]: f"{int(((P[:, j] >= .5) & (Y[:, j] == 1)).sum())}/{int(Y[:, j].sum())}"
                                     for j in SUB},
                    "positives": {COLS[j]: int(Y[:, j].sum()) for j in range(6)}}

        for site in SITES:
            so["h1"][site], so["operating_point"][site] = {}, {}
            for m in MODELS:
                (Pi, Yi), (Pe, Ye) = data["rsna"][m], data[site][m]

                def h1_stats(bi=None, be=None):
                    ai, ae = aucs(Pi, Yi, bi), aucs(Pe, Ye, be)
                    pooled = rel(ai[0], ae[0])
                    sub = np.nanmean([rel(ai[j], ae[j]) for j in SUB])
                    return pooled, sub, sub - pooled

                obs = h1_stats()
                bs = np.array([h1_stats(bi, be) for bi, be in zip(boots["rsna"], boots[site])])
                so["h1"][site][m] = {"pooled_auroc_rel_degradation": round(obs[0], 4), "pooled_ci": ci(bs[:, 0]),
                                     "subtype_mean_auroc_rel_degradation": round(obs[1], 4), "subtype_ci": ci(bs[:, 1]),
                                     "diff_subtype_minus_pooled": round(obs[2], 4), "diff_ci": ci(bs[:, 2]),
                                     "p": p2(bs[:, 2])}

                def op(bi=None, be=None):
                    si, pi = sens_spec(Pi, Yi, idx=bi)
                    se, pe = sens_spec(Pe, Ye, idx=be)
                    return se - si, pe - pi

                o = op()
                bo = np.array([op(bi, be) for bi, be in zip(boots["rsna"], boots[site])])
                so["operating_point"][site][m] = {"delta_sens": round(o[0], 4), "delta_sens_ci": ci(bo[:, 0]),
                                                  "delta_spec": round(o[1], 4), "delta_spec_ci": ci(bo[:, 1]),
                                                  "p_spec": p2(bo[:, 1])}

        for site in ["rsna"] + SITES:
            so["h2"][site] = {}
            Y = data[site]["B2"][1]
            for a_, b_ in [("WICL", "B4c"), ("WICL", "B2"), ("B4c", "B2")]:
                Pa, Pb = data[site][a_][0][:, 0], data[site][b_][0][:, 0]
                d = [auc(Y[b, 0], Pa[b]) - auc(Y[b, 0], Pb[b]) for b in boots[site]]
                so["h2"][site][f"{a_}-{b_}"] = {"diff": round(auc(Y[:, 0], Pa) - auc(Y[:, 0], Pb), 4),
                                                "ci": ci(d), "p": p2(d)}
        out["per_seed"][s] = so

    for s in SEEDS:
        f = R / ("baselines_bhsd_excluded/h4_4c_result.json" if s == 0 else f"g1_seeds/h4_seed{s}_result.json")
        if f.exists():
            h = json.loads(f.read_text())
            out["h4"][s] = {k: round(v["mean_corner_degradation"], 4) for k, v in h["observed"].items()}
            out["h4"][s]["pairwise_p"] = {k: v["p_value"] for k, v in h["pairwise"].items()}

    done = sorted(out["per_seed"])
    if len(done) > 1:
        summ = {}
        for site in ["rsna"] + SITES:
            for m in MODELS:
                v = [out["per_seed"][s]["sites"][site][m]["auroc"]["any"] for s in done]
                summ[f"{site}/{m}/auroc_any"] = [round(float(np.mean(v)), 4), round(float(np.std(v, ddof=1)), 4)]
                v = [out["per_seed"][s]["sites"][site][m]["spec_any@0.5"] for s in done]
                summ[f"{site}/{m}/spec@0.5"] = [round(float(np.mean(v)), 4), round(float(np.std(v, ddof=1)), 4)]
            for k in ["WICL-B4c", "WICL-B2", "B4c-B2"]:
                v = [out["per_seed"][s]["h2"][site][k]["diff"] for s in done]
                summ[f"{site}/{k}"] = {"per_seed": v, "mean": round(float(np.mean(v)), 4),
                                       "same_sign": bool(all(np.sign(v) == np.sign(v[0])))}
        out["across_seeds_mean_sd"] = summ

    (R / f"g1_final_analysis_{agg}.json").write_text(json.dumps(out, indent=1))
    for s, so in out["per_seed"].items():
        print(f"\n##### seed {s}  n={so['n']}")
        for site, d in so["sites"].items():
            for m, r in d.items():
                print(f"{site:9} {m:5} AUROC {r['auroc']['any']:.3f} {r['auroc_any_ci']}  sens {r['sens_any@0.5']:.3f} "
                      f"spec {r['spec_any@0.5']:.3f}  ECE {r['ece_any']:.3f} {r['ece_any_ci']} prev {r['prevalence_any']:.2f} "
                      f"mean score {r['mean_score_any']:.2f}  EDH {r['sub_sens@0.5']['epidural']} AUC {r['auroc']['epidural']:.3f}")
        for site, d in so["h1"].items():
            for m, r in d.items():
                print(f"H1 {site:9} {m:5} pooled {r['pooled_auroc_rel_degradation']:.3f} {r['pooled_ci']}  subtype "
                      f"{r['subtype_mean_auroc_rel_degradation']:.3f} {r['subtype_ci']}  diff {r['diff_subtype_minus_pooled']:+.3f} "
                      f"{r['diff_ci']} p={r['p']}")
        for site, d in so["operating_point"].items():
            for m, r in d.items():
                print(f"OP {site:9} {m:5} dSens {r['delta_sens']:+.3f} {r['delta_sens_ci']}  dSpec {r['delta_spec']:+.3f} "
                      f"{r['delta_spec_ci']} p={r['p_spec']}")
        for site, d in so["h2"].items():
            print(f"H2 {site:9} " + "  ".join(f"{k} {v['diff']:+.4f} {v['ci']} p={v['p']}" for k, v in d.items()))
    print("\nH4:", json.dumps(out["h4"]))
    if "across_seeds_mean_sd" in out:
        print("\nacross seeds:", json.dumps(out["across_seeds_mean_sd"], indent=1))


if __name__ == "__main__":
    import sys
    for agg in (sys.argv[1:] or ["max", "top5mean"]):
        print(f"\n==================== aggregation: {agg} ====================")
        main(agg)
