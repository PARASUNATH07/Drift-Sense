#!/usr/bin/env python3
"""Hybrid coarse-to-fine classical localizer for Drift-Sense images."""
import argparse, csv, time
from pathlib import Path
import cv2
import numpy as np

SCALES = np.linspace(.09, .11, 5)
ANGLES = np.linspace(-2, 2, 5)

def enhanced(im):
    im = cv2.resize(im, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
    return edge_image(im)

def edge_image(im):
    """Noise-tolerant edge representation at the image's current resolution."""
    im = cv2.GaussianBlur(im, (3, 3), 0)
    return cv2.Canny(im, 35, 100)

def transform_template(ref, scale, angle):
    # Apply the physical transform before edge extraction, matching image formation.
    t = cv2.resize(ref, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    m = cv2.getRotationMatrix2D((t.shape[1]/2, t.shape[0]/2), angle, 1)
    t = cv2.warpAffine(t, m, (t.shape[1], t.shape[0]), borderMode=cv2.BORDER_REFLECT)
    return enhanced(t)

def full_resolution_template(ref, scale, angle):
    """Full-resolution, lightly blurred template for final intensity refinement."""
    t = cv2.resize(ref, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    m = cv2.getRotationMatrix2D((t.shape[1] / 2, t.shape[0] / 2), angle, 1)
    t = cv2.warpAffine(t, m, (t.shape[1], t.shape[0]), borderMode=cv2.BORDER_REFLECT)
    return cv2.GaussianBlur(t, (3, 3), 0)

def parabolic_offset(corr, x, y):
    """Sub-pixel maximum of a correlation map using its 3x3 neighbourhood."""
    def offset(a, b, c):
        d = a - 2 * b + c
        return 0.5 * (a - c) / d if abs(d) > 1e-9 else 0.0
    dx = offset(corr[y, x - 1], corr[y, x], corr[y, x + 1]) if 0 < x < corr.shape[1] - 1 else 0.0
    dy = offset(corr[y - 1, x], corr[y, x], corr[y + 1, x]) if 0 < y < corr.shape[0] - 1 else 0.0
    return float(dx), float(dy)

def localize(ref_path, search_path):
    tic = time.perf_counter()
    ref = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE); search = cv2.imread(str(search_path), cv2.IMREAD_GRAYSCALE)
    if ref is None or search is None: raise FileNotFoundError("Could not read reference or search image")
    e_search = enhanced(search)
    candidates = []
    # Stage 1: cheap exhaustive scale/angle NCC in a downsampled edge domain.
    for scale in SCALES:
        for angle in ANGLES:
            t = transform_template(ref, scale, angle)
            corr = cv2.matchTemplate(e_search, t, cv2.TM_CCOEFF_NORMED)
            dil = cv2.dilate(corr, np.ones((9, 9), np.uint8))
            peak_mask = (corr == dil) & (corr >= max(.15, float(corr.max()) - .13))
            for y, x in np.argwhere(peak_mask):
                candidates.append((float(corr[y, x]), int(x), int(y), t.shape[1], t.shape[0], scale, angle, corr))
    if not candidates: raise RuntimeError("No correlation candidates found")
    # Stage 2: center prior actively disambiguates repeated layouts, instead of a tie break.
    cx0, cy0 = e_search.shape[1]/2, e_search.shape[0]/2
    diagonal = float(np.hypot(cx0, cy0))
    def rerank(c):
        score, x, y, tw, th, *_ = c
        center = np.array([x + tw/2, y + th/2])
        # Deliberately small: a prior guides near-equal periodic aliases without masking NCC evidence.
        return score - .025 * np.linalg.norm(center - [cx0, cy0]) / diagonal
    candidates.sort(key=rerank, reverse=True)
    best = candidates[0]
    score, x, y, tw, th, scale, angle, corr = best
    # Stage 3a: sub-pixel estimate in the 0.5x NCC map.
    dx, dy = parabolic_offset(corr, x, y)
    coarse_x, coarse_y = (x + dx + tw / 2) * 2, (y + dy + th / 2) * 2
    # Stage 3b: full-resolution local NCC removes coarse-map quantization while
    # keeping the expensive search restricted to this small candidate region.
    # Recheck five nearby scales and angles only within this local window.
    # Blurred intensity NCC is more stable than Canny at full resolution.
    radius = 24
    blurred_search = cv2.GaussianBlur(search, (3, 3), 0)
    refined = None
    for fine_scale in np.linspace(max(.09, scale - .005), min(.11, scale + .005), 5):
        for fine_angle in np.linspace(angle - 1, angle + 1, 5):
            full_t = full_resolution_template(ref, fine_scale, fine_angle)
            left = max(0, int(np.floor(coarse_x - full_t.shape[1] / 2 - radius)))
            top = max(0, int(np.floor(coarse_y - full_t.shape[0] / 2 - radius)))
            right = min(search.shape[1], int(np.ceil(coarse_x + full_t.shape[1] / 2 + radius)))
            bottom = min(search.shape[0], int(np.ceil(coarse_y + full_t.shape[0] / 2 + radius)))
            full_corr = cv2.matchTemplate(blurred_search[top:bottom, left:right], full_t, cv2.TM_CCOEFF_NORMED)
            _, refined_score, _, refined_loc = cv2.minMaxLoc(full_corr)
            fx, fy = refined_loc
            fdx, fdy = parabolic_offset(full_corr, fx, fy)
            candidate = (refined_score, fine_scale, fine_angle,
                         left + fx + fdx + full_t.shape[1] / 2,
                         top + fy + fdy + full_t.shape[0] / 2)
            if refined is None or candidate[0] > refined[0]:
                refined = candidate
    refined_score, refined_scale, refined_angle, pred_x, pred_y = refined
    ordered = sorted(candidates, key=rerank, reverse=True)
    margin = rerank(ordered[0]) - rerank(ordered[1]) if len(ordered) > 1 else score
    return dict(pred_x=pred_x, pred_y=pred_y, confidence=score, refined_score=refined_score, margin=margin,
                scale=refined_scale, angle_deg=refined_angle,
                runtime_ms=(time.perf_counter()-tic)*1000)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ref"); ap.add_argument("--search"); ap.add_argument("--batch"); ap.add_argument("--out")
    a = ap.parse_args()
    if a.batch:
        if not a.out: ap.error("--batch requires --out")
        with open(a.batch, newline="") as f: rows = list(csv.DictReader(f))
        results=[]
        for row in rows:
            result=localize(row["reference"], row["search"]); results.append({**row, **result})
        with open(a.out, "w", newline="") as f:
            w=csv.DictWriter(f, fieldnames=results[0].keys()); w.writeheader(); w.writerows(results)
        print(f"Wrote {len(results)} results to {a.out}")
    elif a.ref and a.search:
        print(localize(a.ref, a.search))
    else: ap.error("provide --ref/--search or --batch/--out")
if __name__ == "__main__": main()
