"""Local threshold recalibration after site transfer (G1 revision, 2026-10-01).

For each external site, studies are split deterministically into halves A and B (sorted IDs, alternating). A threshold is
chosen on k randomly drawn labelled studies from half A to reach >= 90% any-ICH sensitivity (200 draws; draws with fewer
than 2 positives skipped) and evaluated on half B, against the RSNA-locked 0.5 threshold. Max aggregation; B2/B4c/WICL,
seeds 0-2; PhysioNet in radiological orientation.
"""
import json

import numpy as np

from g1_final_analysis import MODELS, R, SEEDS, load, sens_spec

TARGET = 0.90


def main():
    rng = np.random.default_rng(0)
    out = {"protocol": __doc__, "results": {}}
    for site, ks in [("cq500", [10, 20, 50, 100]), ("physionet", [10, 20])]:
        for m in MODELS:
            for s in SEEDS:
                P, Y = load(site, m, s, "max")
                score, y = P[:, 0], Y[:, 0]
                order = np.arange(len(y))
                a, b = order[0::2], order[1::2]
                fixed = sens_spec(P, Y, idx=b)
                rec = {"n_half_B": int(len(b)), "fixed_0.5": [round(float(v), 4) for v in fixed]}
                for k in ks:
                    res = []
                    for _ in range(200):
                        cal = rng.choice(a, min(k, len(a)), replace=False)
                        pos = np.sort(score[cal][y[cal] == 1])
                        if len(pos) < 2:
                            continue
                        thr = pos[int(np.floor((1 - TARGET) * len(pos)))]
                        res.append(sens_spec(P, Y, thr=thr, idx=b))
                    res = np.array(res)
                    rec[f"k={k}"] = {"sens_median": round(float(np.median(res[:, 0])), 4),
                                     "spec_median": round(float(np.median(res[:, 1])), 4),
                                     "spec_iqr": [round(float(q), 4) for q in np.percentile(res[:, 1], [25, 75])],
                                     "draws": int(len(res))}
                out["results"][f"{site}/{m}/seed{s}"] = rec
                ks_txt = "  ".join(f"k={k}: sens {rec[f'k={k}']['sens_median']:.2f} spec {rec[f'k={k}']['spec_median']:.2f}"
                                   for k in ks)
                print(f"{site:9} {m:5} seed{s}  fixed sens {fixed[0]:.2f} spec {fixed[1]:.2f}  {ks_txt}")
    (R / "g1_recalibration.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
