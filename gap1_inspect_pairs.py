"""Gap 1: pixel-level inspection of sampled RSNA<->CQ500 near-duplicate
candidate pairs flagged by the phash check (see
runs/decisive_v2/leakage_check_full_hits.json, the full 302-pair
reproduction of leakage_check_rsna_cq500.py's subsampled check).

For each sampled pair, loads BOTH slices' real HU pixel data with the same
loaders the pipeline uses (rsna.read_hu / cq500's read_hu, which is the
same function), computes actual pixel-level diff metrics (mean absolute HU
difference, Pearson correlation, SSIM on the brain-windowed image), and
saves a side-by-side PNG (RSNA | CQ500 | abs-diff heatmap) for visual
inspection. Writes a structured per-pair JSON summary with a
classification (TRUE_DUPLICATE_OR_NEAR / ARTIFACT / UNCLEAR) and the
metrics backing it -- classification is NOT asserted without the numbers
attached.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.transform import resize
from skimage.metrics import structural_similarity as ssim

from ich_gen.datasets.rsna import read_hu
from ich_gen.windowing import CANONICAL_WINDOWS, window_hu

RUNS_DIR = Path("runs/decisive_v2")
OUT_DIR = RUNS_DIR / "leakage_pair_review_images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SHAPE = (256, 256)


def to_uint8_brain(hu: np.ndarray) -> np.ndarray:
    level, width = CANONICAL_WINDOWS["brain"]
    w = window_hu(hu, level, width)
    return np.clip(w, 0, 255).astype(np.uint8)


def load_hu_resized(path: str):
    hu = read_hu(Path(path))
    raw_shape = hu.shape
    hu_r = resize(hu.astype(np.float32), TARGET_SHAPE, order=1,
                  preserve_range=True, anti_aliasing=True)
    return hu, hu_r, raw_shape


def main():
    with open(RUNS_DIR / "leakage_check_full_hits.json") as f:
        full = json.load(f)

    by_h = full["by_hamming_distance"]
    rsna_paths = full["rsna_paths"]
    cq_paths = full["cq500_paths"]

    # Sample: ALL hamming=0 pairs, a spread of hamming=2, a spread of
    # hamming=4 (every Nth entry across the sorted list for spread, not
    # just the first few).
    sample = []
    sample += [("0", h) for h in by_h["0"]]
    h2 = by_h["2"]
    sample += [("2", h) for h in h2[::max(1, len(h2) // 8)][:8]]
    h4 = by_h["4"]
    sample += [("4", h) for h in h4[::max(1, len(h4) // 8)][:8]]

    print(f"inspecting {len(sample)} sampled pairs "
          f"({len(by_h['0'])} hamming=0, 8 of {len(h2)} hamming=2, "
          f"8 of {len(h4)} hamming=4)")

    results = []
    for idx, (hamming, h) in enumerate(sample):
        rsna_id, cq_id = h["rsna_id"], h["cq500_id"]
        rp, cp = rsna_paths[rsna_id], cq_paths[cq_id]
        try:
            hu_r_raw, hu_r_rs, shape_r = load_hu_resized(rp)
            hu_c_raw, hu_c_rs, shape_c = load_hu_resized(cp)
        except Exception as e:
            print(f"[{idx}] {rsna_id} vs {cq_id}: LOAD FAILED: {type(e).__name__}: {e}")
            results.append({"rsna_id": rsna_id, "cq500_id": cq_id,
                             "hamming": int(hamming), "error": str(e),
                             "classification": "UNCLEAR"})
            continue

        mae = float(np.mean(np.abs(hu_r_rs - hu_c_rs)))
        corr = float(np.corrcoef(hu_r_rs.ravel(), hu_c_rs.ravel())[0, 1])

        img_r = to_uint8_brain(hu_r_rs)
        img_c = to_uint8_brain(hu_c_rs)
        ssim_val = float(ssim(img_r, img_c, data_range=255))

        # near-empty / low-information heuristic: fraction of brain-window
        # pixels that are near-zero (black) -- a classic source of phash
        # false positives on CT (many slices near the vertex/skull-base
        # are mostly background/bone-only with almost no soft-tissue
        # signal, so very different anatomy can hash identically).
        frac_dark_r = float(np.mean(img_r < 5))
        frac_dark_c = float(np.mean(img_c < 5))
        std_r = float(hu_r_rs.std())
        std_c = float(hu_c_rs.std())

        # classification heuristic (numbers-first, not vibes):
        #  TRUE_DUPLICATE_OR_NEAR: very high structural agreement AND not
        #    just two near-blank/degenerate slices agreeing trivially.
        #  ARTIFACT: low information content (near-blank / near-uniform)
        #    on at least one side, which is a well-known phash blind spot,
        #    OR clearly-uncorrelated pixel content despite hash agreement.
        #  UNCLEAR: doesn't cleanly fall into either bucket.
        degenerate = (frac_dark_r > 0.85 or frac_dark_c > 0.85 or
                      std_r < 15 or std_c < 15)
        strong_match = (ssim_val > 0.85 and corr > 0.85 and mae < 25)
        weak_match = (ssim_val < 0.4 or corr < 0.3)

        if degenerate and not strong_match:
            classification = "ARTIFACT"
            reason = (f"near-blank/low-information slice on one or both "
                      f"sides (frac_dark_r={frac_dark_r:.2f}, "
                      f"frac_dark_c={frac_dark_c:.2f}, std_r={std_r:.1f}, "
                      f"std_c={std_c:.1f} HU) -- classic phash blind spot, "
                      f"not evidence of shared provenance")
        elif strong_match:
            classification = "TRUE_DUPLICATE_OR_NEAR"
            reason = (f"high structural/pixel agreement: SSIM={ssim_val:.3f}, "
                      f"corr={corr:.3f}, MAE={mae:.1f} HU, and not a "
                      f"degenerate/near-blank slice -- consistent with "
                      f"genuinely overlapping or near-identical imaging")
        elif weak_match:
            classification = "ARTIFACT"
            reason = (f"low pixel/structural agreement despite phash<= "
                      f"{hamming}: SSIM={ssim_val:.3f}, corr={corr:.3f} -- "
                      f"phash matched on coarse structure only "
                      f"(generic anatomy silhouette), not real content")
        else:
            classification = "UNCLEAR"
            reason = (f"middling agreement (SSIM={ssim_val:.3f}, "
                      f"corr={corr:.3f}, MAE={mae:.1f} HU) -- does not "
                      f"cleanly classify; needs a human look at the saved "
                      f"PNG")

        # save side-by-side PNG: RSNA | CQ500 | abs-diff heatmap
        diff_img = np.abs(hu_r_rs - hu_c_rs)
        diff_norm = np.clip(diff_img / max(diff_img.max(), 1e-6) * 255, 0, 255).astype(np.uint8)
        canvas = np.zeros((TARGET_SHAPE[0], TARGET_SHAPE[1] * 3 + 20), dtype=np.uint8)
        canvas[:, :TARGET_SHAPE[1]] = img_r
        canvas[:, TARGET_SHAPE[1] + 10:TARGET_SHAPE[1] * 2 + 10] = img_c
        canvas[:, TARGET_SHAPE[1] * 2 + 20:] = diff_norm
        png_name = f"h{hamming}_{idx:02d}_{rsna_id}_vs_{cq_id}.png"
        Image.fromarray(canvas).save(OUT_DIR / png_name)

        print(f"[{idx}] hamming={hamming} {rsna_id} vs {cq_id}: "
              f"MAE={mae:.1f} corr={corr:.3f} SSIM={ssim_val:.3f} "
              f"shapes=({shape_r},{shape_c}) -> {classification}")

        results.append({
            "rsna_id": rsna_id, "cq500_id": cq_id, "hamming": int(hamming),
            "rsna_path": rp, "cq500_path": cp,
            "rsna_raw_shape": list(shape_r), "cq500_raw_shape": list(shape_c),
            "mean_abs_hu_diff": mae, "pearson_corr": corr, "ssim": ssim_val,
            "frac_dark_rsna": frac_dark_r, "frac_dark_cq500": frac_dark_c,
            "std_hu_rsna": std_r, "std_hu_cq500": std_c,
            "classification": classification, "reasoning": reason,
            "image_path": str(OUT_DIR / png_name),
        })

    out_path = RUNS_DIR / "leakage_pair_review.json"
    with open(out_path, "w") as f:
        json.dump({"n_sampled": len(results),
                    "n_hamming0_total": len(by_h["0"]),
                    "n_hamming2_total": len(by_h["2"]),
                    "n_hamming4_total": len(by_h["4"]),
                    "results": results}, f, indent=2)
    print(f"\nwrote {out_path}")

    n_true = sum(1 for r in results if r.get("classification") == "TRUE_DUPLICATE_OR_NEAR")
    n_artifact = sum(1 for r in results if r.get("classification") == "ARTIFACT")
    n_unclear = sum(1 for r in results if r.get("classification") == "UNCLEAR")
    print(f"\nSUMMARY: {n_true} TRUE_DUPLICATE_OR_NEAR, {n_artifact} ARTIFACT, "
          f"{n_unclear} UNCLEAR (of {len(results)} sampled)")


if __name__ == "__main__":
    main()
