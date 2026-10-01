"""G1 revision round 2 analysis (2026-10-01): second architecture (ConvNeXt-Tiny, 3 seeds), larger training budget
(DenseNet-121, 100,000 slices, seed 0), and CQ500 slice-thickness harmonization. Max-over-slice scan scores,
RSNA-locked threshold 0.5, bootstrap 1,000 resamples of patients/studies. Writes runs/g1_extra_analysis.json."""
import json
from pathlib import Path

import numpy as np

from g1_final_analysis import COLS, SUB, auc, ci, ece, p2, rel, sens_spec
from g1_final_analysis import path as dn_path

R = Path("runs")
MODELS = ["B2", "B4c", "WICL"]
N_BOOT = 1000
FAMILIES = {"DN20k": [0, 1, 2], "CNX": [0, 1, 2], "BIG": [0]}


def fpath(fam, site, m, s):
    if fam == "DN20k":
        return dn_path(site, m, s)
    pre = f"{fam}_{m}_seed{s}"
    return R / (f"g1_rsna_patient/{pre}.npz" if site == "rsna" else f"g1_ext_more/{pre}__{site}.npz")


def to_scan(p):
    d = np.load(p)
    ids, inv = np.unique(d["pid"], return_inverse=True)
    P = np.full((len(ids), 6), -np.inf)
    Y = np.zeros((len(ids), 6))
    np.maximum.at(P, inv, d["prob"])
    np.maximum.at(Y, inv, d["y"])
    return ids, P, Y, d, inv


def family_block(fam, rng):
    res = {}
    for s in FAMILIES[fam]:
        files = {(site, m): fpath(fam, site, m, s) for site in ("rsna", "cq500", "physionet") for m in MODELS}
        if not all(f.exists() for f in files.values()):
            continue
        data = {k: to_scan(f)[1:3] for k, f in files.items()}
        n = {site: len(data[(site, "B2")][1]) for site in ("rsna", "cq500", "physionet")}
        boots = {site: [rng.integers(0, n[site], n[site]) for _ in range(N_BOOT)] for site in n}
        so = {}
        for site in ("rsna", "cq500", "physionet"):
            for m in MODELS:
                P, Y = data[(site, m)]
                se, sp = sens_spec(P, Y)
                so[f"{site}/{m}"] = {"auroc": round(auc(Y[:, 0], P[:, 0]), 4), "sens": round(float(se), 4),
                                     "spec": round(float(sp), 4), "ece": round(ece(Y[:, 0], P[:, 0]), 4),
                                     "sub_auroc": {COLS[j]: round(auc(Y[:, j], P[:, j]), 4) for j in SUB},
                                     "edh_detected": f"{int(((P[:, 1] >= .5) & (Y[:, 1] == 1)).sum())}/{int(Y[:, 1].sum())}"}
        for site in ("cq500", "physionet"):
            for m in MODELS:
                (Pi, Yi), (Pe, Ye) = data[("rsna", m)], data[(site, m)]

                def stats(bi=None, be=None):
                    ai = np.array([auc((Yi if bi is None else Yi[bi])[:, j], (Pi if bi is None else Pi[bi])[:, j]) for j in range(6)])
                    ae = np.array([auc((Ye if be is None else Ye[be])[:, j], (Pe if be is None else Pe[be])[:, j]) for j in range(6)])
                    si, pi_ = sens_spec(Pi, Yi, idx=bi)
                    se_, pe_ = sens_spec(Pe, Ye, idx=be)
                    pooled = rel(ai[0], ae[0])
                    return pooled, np.nanmean([rel(ai[j], ae[j]) for j in SUB]) - pooled, pe_ - pi_, se_ - si

                obs = stats()
                bs = np.array([stats(bi, be) for bi, be in zip(boots["rsna"], boots[site])])
                so[f"delta/{site}/{m}"] = {"h1_diff_subtype_minus_pooled": round(obs[1], 4), "h1_p": p2(bs[:, 1]),
                                           "d_spec": round(obs[2], 4), "d_spec_p": p2(bs[:, 2]),
                                           "d_sens": round(obs[3], 4), "d_sens_p": p2(bs[:, 3])}
        for site in ("rsna", "cq500", "physionet"):
            Y = data[(site, "B2")][1]
            for a_, b_ in (("WICL", "B4c"), ("B4c", "B2"), ("WICL", "B2")):
                Pa, Pb = data[(site, a_)][0][:, 0], data[(site, b_)][0][:, 0]
                d = [auc(Y[b, 0], Pa[b]) - auc(Y[b, 0], Pb[b]) for b in boots[site]]
                so[f"diff/{site}/{a_}-{b_}"] = {"diff": round(auc(Y[:, 0], Pa) - auc(Y[:, 0], Pb), 4), "p": p2(d)}
        h4 = R / ("g1_convnext" if fam == "CNX" else "") / f"h4_seed{s}_result.json"
        if fam == "CNX" and h4.exists():
            h = json.loads(h4.read_text())
            so["h4"] = {k: round(v["mean_corner_degradation"], 4) for k, v in h["observed"].items()}
            so["h4_p"] = {k: v["p_value"] for k, v in h["pairwise"].items()}
        res[s] = so
    return res


def harmonization():
    out = {}
    names = [(f"{m}_seed{s}", "DN20k", m, s) for s in (0, 1, 2) for m in MODELS] + \
            [(f"CNX_{m}_seed{s}", "CNX", m, s) for s in (0, 1, 2) for m in MODELS] + \
            [(f"BIG_{m}_seed0", "BIG", m, 0) for m in MODELS]
    for name, fam, m, s in names:
        h = R / f"g1_cq500_harmonized/{name}__cq500h.npz"
        o = fpath(fam, "cq500", m, s)
        if not (h.exists() and o.exists()):
            continue
        ids_h, Ph, Yh, dh, invh = to_scan(h)
        ids_o, Po, Yo, do, invo = to_scan(o)
        assert np.array_equal(ids_h, ids_o) and np.array_equal(Yh[:, 0], Yo[:, 0])
        thick = np.zeros(len(ids_h))
        thick[invh] = dh["thickness"]
        thin = thick < 4.5
        rec = {"n_thin_studies": int(thin.sum()), "n_thin_negative": int((thin & (Yh[:, 0] == 0)).sum())}
        for tag, P, Y, d, inv in (("original", Po, Yo, do, invo), ("harmonized", Ph, Yh, dh, invh)):
            se, sp = sens_spec(P, Y)
            se_t, sp_t = sens_spec(P, Y, idx=np.where(thin)[0])
            neg_thin_slice = (Y[inv, 0] == 0) & thin[inv]
            rec[tag] = {"auroc": round(auc(Y[:, 0], P[:, 0]), 4), "sens": round(float(se), 4), "spec": round(float(sp), 4),
                        "spec_thin": round(float(sp_t), 4), "sens_thin": round(float(se_t), 4),
                        "slice_fp_pct_thin_neg": round(float((d["prob"][neg_thin_slice, 0] >= .5).mean() * 100), 3)}
        rng = np.random.default_rng(0)
        Y = Yo[:, 0]
        bs = []
        for _ in range(N_BOOT):
            b = rng.integers(0, len(Y), len(Y))
            bs.append((auc(Y[b], Ph[b, 0]) - auc(Y[b], Po[b, 0]),
                       sens_spec(Ph, Yh, idx=b)[1] - sens_spec(Po, Yo, idx=b)[1],
                       sens_spec(Ph, Yh, idx=b)[0] - sens_spec(Po, Yo, idx=b)[0]))
        bs = np.array(bs)
        rec["paired"] = {"d_auroc": round(rec["harmonized"]["auroc"] - rec["original"]["auroc"], 4), "p_auroc": p2(bs[:, 0]),
                         "d_spec": round(rec["harmonized"]["spec"] - rec["original"]["spec"], 4), "p_spec": p2(bs[:, 1]),
                         "d_sens": round(rec["harmonized"]["sens"] - rec["original"]["sens"], 4), "p_sens": p2(bs[:, 2])}
        out[name] = rec
    return out


def main():
    rng = np.random.default_rng(0)
    out = {"protocol": __doc__, "families": {f: family_block(f, rng) for f in FAMILIES}, "harmonization": harmonization()}
    (R / "g1_extra_analysis.json").write_text(json.dumps(out, indent=1))
    for fam, blk in out["families"].items():
        for s, so in blk.items():
            print(f"\n## {fam} seed {s}")
            for k, v in so.items():
                print(" ", k, json.dumps(v))
    print("\n## harmonization")
    for k, v in out["harmonization"].items():
        print(" ", k, json.dumps(v))


if __name__ == "__main__":
    main()
