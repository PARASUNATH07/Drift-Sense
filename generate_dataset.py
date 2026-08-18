#!/usr/bin/env python3
"""Create procedural SEM-like reference/search pairs for Drift-Sense."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def layout(size, architecture, rng):
    """Procedural DRAM cells or FinFET lines (not fab imagery)."""
    y, x = np.mgrid[:size, :size]
    img = np.full((size, size), 75.0, np.float32)
    if architecture == "dram":
        # DRAM-style layout: word/bit-line grid, rectangular cell capacitors,
        # and periodic via/contact dots at each cell intersection.
        pitch = int(rng.integers(32, 45))
        ox, oy = rng.integers(0, pitch, 2)
        grid = ((x - ox) % pitch < 3) | ((y - oy) % pitch < 3)
        img[grid] = 175
        # Repeating rectangular memory cells, with slight intentional variation.
        cell = (((x - ox - 7) % pitch < pitch * .48) &
                ((y - oy - 8) % pitch < pitch * .35))
        img[cell] = 128
        for vx in range(int(ox + pitch // 2), size, pitch):
            for vy in range(int(oy + pitch // 2), size, pitch):
                cv2.circle(img, (vx, vy), 3, 215, -1)
    else:
        pitch = int(rng.integers(26, 38))
        phase = rng.integers(0, pitch)
        fins = ((x + phase) % pitch < 5)
        gates = ((y + phase) % (pitch * 3) < 4)
        img[fins] = 185
        img[gates] = np.maximum(img[gates], 150)
        img[((x + phase) % pitch > 7) & ((x + phase) % pitch < 10)] = 110
    # A few local process-like defects provide limited non-periodic identity.
    for _ in range(18):
        cx, cy = rng.integers(35, size - 35, 2)
        # Deliberate coarse process marks survive the 10:1 view and make the
        # otherwise periodic localization problem identifiable.
        cv2.circle(img, (int(cx), int(cy)), int(rng.integers(10, 30)),
                   float(rng.integers(95, 205)), -1)
    return img


def sem_degrade(image, rng, strength, params):
    """Approximate common SEM effects: PSF, shot/detector noise, contamination and charging."""
    img = cv2.GaussianBlur(image, (0, 0), params["psf_sigma"] * strength)
    # Gaussian PSF blur models finite probe / interaction-volume resolution.
    # Poisson shot noise models electron counting statistics (variance rises with signal).
    photons = params["photons"] / strength
    img = rng.poisson(np.clip(img, 0, 255) / 255 * photons) / photons * 255
    # Detector/readout noise is conventionally approximated as additive Gaussian noise.
    img += rng.normal(0, params["detector_sigma"] * strength, img.shape)
    # Sparse impulse noise approximates transient detector/contamination events.
    p = params["sp_probability"] * strength
    impulse = rng.random(img.shape)
    img[impulse < p / 2] = 0
    img[(impulse >= p / 2) & (impulse < p)] = 255
    # Edge brightening is a common topographic secondary-electron contrast effect.
    edge = cv2.Laplacian(img.astype(np.float32), cv2.CV_32F)
    img += params["edge_gain"] * np.maximum(edge, 0)
    h, w = img.shape
    yy, xx = np.mgrid[:h, :w]
    rr = ((xx - w/2)/(w/2))**2 + ((yy - h/2)/(h/2))**2
    # Mild vignetting / charging gradients are observed scan-field artifacts.
    img *= 1 - params["vignette"] * strength * rr
    for _ in range(params["streaks"]):
        row = int(rng.integers(0, h))
        img[max(0, row-1):min(h, row+2)] += rng.uniform(-10, 12) * strength
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20260815)
    args = ap.parse_args()
    out = args.out
    ref_dir, search_dir = out / "references", out / "searches"
    ref_dir.mkdir(parents=True, exist_ok=True); search_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    master = np.random.default_rng(args.seed)
    for i in range(args.n):
        seed = int(master.integers(0, 2**32 - 1)); rng = np.random.default_rng(seed)
        architecture = "dram" if rng.random() < .5 else "finfet"
        scale, angle = float(rng.uniform(.09, .11)), float(rng.uniform(-2, 2))
        # Bounded navigation drift: target remains near, but not necessarily at, field center.
        cx, cy = [float(rng.uniform(390, 610)) for _ in range(2)]
        params = dict(psf_sigma=float(rng.uniform(.6, 1.15)), photons=float(rng.uniform(80, 170)),
                      detector_sigma=float(rng.uniform(2, 5)), sp_probability=float(rng.uniform(.0003, .001)),
                      edge_gain=float(rng.uniform(.10, .22)), vignette=float(rng.uniform(.03, .09)),
                      streaks=int(rng.integers(1, 4)))
        reference_clean = layout(1000, architecture, rng)
        reference = sem_degrade(reference_clean, rng, .55, params)
        # Fast-scan wide-field background is independently noisier than the slow-scan reference.
        background = sem_degrade(layout(1000, architecture, rng), rng, 1.35, params)
        # Transform high-mag reference to its low-mag appearance and composite into wide field.
        small = cv2.resize(reference, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        rot = cv2.getRotationMatrix2D((small.shape[1]/2, small.shape[0]/2), angle, 1)
        small = cv2.warpAffine(small, rot, (small.shape[1], small.shape[0]), borderMode=cv2.BORDER_REFLECT)
        # Retain the same underlying slow-scan patch while the surrounding search field
        # carries the stronger fast-scan noise; this makes the paired correspondence valid.
        x0, y0 = int(round(cx-small.shape[1]/2)), int(round(cy-small.shape[0]/2))
        # Record the target centre actually rendered after integer placement.
        actual_cx, actual_cy = x0 + small.shape[1] / 2, y0 + small.shape[0] / 2
        mask = np.full(small.shape, 1.0, np.float32)
        background[y0:y0+small.shape[0], x0:x0+small.shape[1]] = small * mask
        search = background
        ref_path, search_path = ref_dir / f"ref_{i:04d}.png", search_dir / f"search_{i:04d}.png"
        cv2.imwrite(str(ref_path), reference); cv2.imwrite(str(search_path), search)
        rows.append(dict(id=i, reference=str(ref_path), search=str(search_path), true_x=f"{actual_cx:.4f}", true_y=f"{actual_cy:.4f}",
                         seed=seed, scale=f"{scale:.6f}", rotation_deg=f"{angle:.5f}", architecture=architecture,
                         parameters=json.dumps(params, separators=(",", ":"))))
    with (out / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {args.n} pairs and {out / 'manifest.csv'}")

if __name__ == "__main__": main()
