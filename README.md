# Drift-Sense

Classical, reproducible synthetic benchmark and localization baseline for SEM-style cross-magnification navigation recovery. It uses only NumPy and OpenCV. The generator implements two procedural (non-fab) layouts: DRAM-style rectangular memory-cell arrays with word/bit lines and via/contact dots, and FinFET-style parallel fin/gate lines.

## Install

```bash
python -m pip install -r requirements.txt
```

## Generate, localize, and evaluate

```bash
# Quick incremental check
python generate_dataset.py --n 5 --out data/quick/
python localize.py --batch data/quick/manifest.csv --out data/quick/results.csv
python evaluate.py --results data/quick/results.csv --out data/quick/evaluation/

# Full benchmark (40 pairs spanning architecture/noise/scale/rotation)
python generate_dataset.py --n 40 --out data/
python localize.py --batch data/manifest.csv --out data/results.csv
python evaluate.py --results data/results.csv --out data/evaluation/
```

Localize one pair:

```bash
python localize.py --ref data/references/ref_0000.png --search data/searches/search_0000.png
```

## File formats and conventions

All images are grayscale PNGs, 1000 x 1000 pixels. `manifest.csv` has `reference`, `search`, `true_x`, `true_y`, `seed`, `scale`, `rotation_deg`, `architecture`, and JSON `parameters` (the sampled degradation/noise settings).

Batch `results.csv` retains those columns and adds `pred_x`, `pred_y`, `confidence`, `refined_score`, `margin`, selected `scale`/`angle_deg`, and `runtime_ms`.

Coordinates name the target center in search-image pixels: origin is top-left; x increases rightward and y increases downward. The target is sampled in the central search region as a bounded-drift assumption. Reference-to-search scale is 0.09 to 0.11 and rotation is -2 to +2 degrees.

## Pipeline

1. **Coarse search:** edge-enhanced images are downsampled 2x and searched with
   normalized cross-correlation across five scale and five angle hypotheses.
2. **Center-prior re-ranking:** all strong, spatially separated peaks are retained
   and re-ranked by `NCC - 0.025 x normalized distance from center`, applying a
   bounded-drift prior to disambiguate periodic-layout aliases.
3. **Coarse sub-pixel fit:** a parabolic fit around the winning coarse peak gives
   an initial sub-pixel center estimate.
4. **Full-resolution local refinement:** within a small window (±24 px) around the
   coarse estimate, the reference is re-rendered at full resolution (lightly
   blurred, not edge-extracted) and matched against the *unblurred* search image
   intensities across five nearby scale/angle values. This removes the
   quantization error introduced by the 2x-downsampled coarse stage and typically
   reduces final error by roughly 1 px. `refined_score` is the NCC confidence of
   this final, full-resolution match; `confidence` (from step 1) remains the
   original coarse-stage score.

## Confidence and periodic patterns

NCC confidence is a normalized local similarity score, not a probability. It can be low (for example, around 0.32) even for a correct prediction because the 10:1 resampling, blur/noise, and edge extraction reduce direct similarity, while repeating DRAM/FinFET elements create several comparable peaks. The `0.025` center-distance term is intentionally small and only ranks near-equal peaks; the `max - 0.13` peak band deliberately preserves plausible aliases for that ranking. A low NCC margin is therefore an ambiguity indicator, not automatically an incorrect result. The evaluator records the worst case with this root-cause note.

The synthetic effects are deliberately approximate, procedural SEM analogues rather than calibrated instrument physics. Source comments identify the conventional motivation: Gaussian PSF (probe/interaction resolution), Poisson noise (electron counting), Gaussian detector read noise, impulsive contamination events, secondary-electron edge contrast, and scan-field vignette/charging streaks.