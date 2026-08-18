# Drift-Sense

**A classical, reproducible localizer for SEM navigation-error recovery** — built for the
Applied Materials "Drift-Sense" challenge at SEMICON India Hackathon 2026.

Given a high-magnification reference image and a wider low-magnification search image,
Drift-Sense finds the center `(x, y)` of the reference pattern inside the search image —
so a tool that has drifted off-target can recover its true position automatically.

**No trained model. No GPU. No internet access. Pure NumPy + OpenCV.** It installs and runs
anywhere in seconds, which matters when reproducibility and deployability count as much as
raw accuracy.

## Quick start

```bash
python -m pip install -r requirements.txt
python run.py <input-dir> <output-dir>
```

That's it — `<output-dir>` is created automatically and filled with per-pair JSON results
plus an aggregated `results.csv`.

## Results (40-sample synthetic benchmark)

Benchmarked on 40 synthetic pairs (24 FinFET, 16 DRAM), spanning scale 0.09–0.11, rotation
±2°, and varying noise/blur levels:

| Metric | Value |
|---|---|
| **Pass @ 1px** | **95%** |
| Pass @ 2px | 95% |
| Pass @ 5px | 95% |
| Median error | 0.24 px |
| Mean error | 8.06 px |
| Worst-case error | 245.7 px |
| Mean runtime | ~320 ms/pair (CPU) |

The worst-case error is a single occasional coarse-stage misfire on a highly periodic
region — see [Confidence and periodic patterns](#confidence-and-periodic-patterns) below,
and the visualized failure case at `data/evaluation/worst_case.png` after running the
benchmark commands further down.

## Why classical, not ML?

No training data, no labeling, no GPU dependency, no model drift between runs — the same
input always gives the same output. For a navigation-recovery tool that has to run
reliably on a fab floor, that reproducibility is the point, not a compromise.

## Folder structure

```
drift-sense/
├── run.py                  # standardized batch entry point
├── localizer_core.py       # core matching algorithm (coarse-to-fine NCC + sub-pixel refinement)
├── generate_dataset.py     # synthetic DRAM/FinFET dataset generator
├── evaluate.py              # scoring script (error metrics, pass rates, failure visualization)
├── requirements.txt
├── README.md
└── models/                 # intentionally empty -- no trained weights are used
```

## Running on your own data

```bash
python run.py <input-dir> <output-dir>
```

- Creates `<output-dir>` if it doesn't already exist.
- Reads every reference/search image pair from `<input-dir>`.
- Writes one `<id>.json` per pair to `<output-dir>`, plus an aggregated `results.csv`.
- Runs entirely on CPU; no GPU, internet access, API keys, model downloads, or manual
  configuration required.

### Input-dir contract

Either layout works:

1. **Subfolders:** `<input-dir>/references/` and `<input-dir>/searches/`, matched pairwise
   by identical filename, or by a shared id after stripping a leading `ref_`/`reference_`/
   `search_` token (e.g. `references/ref_0000.png` matches `searches/search_0000.png`).
2. **Flat folder:** paired files sharing a common id in one directory, e.g.
   `0000_reference.png` / `0000_search.png`.

Accepted formats: `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp` (grayscale or color; color
is converted to grayscale automatically). Any pair the algorithm fails on is still recorded
in the output (with `pred_x`/`pred_y` as `null` and an `error` field set) rather than
crashing the whole batch.

### Output format

Each `<output-dir>/<id>.json`:

```json
{
  "id": "0000",
  "reference": "path/to/reference.png",
  "search": "path/to/search.png",
  "pred_x": 501.72,
  "pred_y": 605.77,
  "confidence": 0.436,
  "refined_score": 0.859,
  "margin": 0.032,
  "scale": 0.11,
  "angle_deg": 1.5,
  "runtime_ms": 329.4
}
```

`<output-dir>/results.csv` aggregates the same fields for every pair in one file.

## Generate synthetic data and evaluate (development / validation workflow)

```bash
# Full benchmark (40 pairs spanning architecture/noise/scale/rotation)
python generate_dataset.py --n 40 --out data/
python run.py data/ data/predictions/
python evaluate.py --results data/predictions/results.csv --out data/evaluation/
```

`generate_dataset.py` produces the `references/` + `searches/` subfolder layout `run.py`
expects, plus `manifest.csv` recording ground truth (id, true_x, true_y, seed, scale,
rotation_deg, architecture, noise parameters) for every pair.

<details>
<summary><strong>Technical details: how the pipeline works</strong></summary>

1. **Coarse search:** edge-enhanced images are downsampled 2x and searched with normalized
   cross-correlation (NCC) across five scale hypotheses (0.09–0.11) and five rotation
   hypotheses (-2° to +2°).
2. **Center-prior re-ranking:** all strong, spatially separated peaks are retained and
   re-ranked by `NCC - 0.025 x normalized distance from center`, applying a bounded-drift
   prior to disambiguate periodic-layout aliases (DRAM/FinFET repeats).
3. **Coarse sub-pixel fit:** a parabolic fit around the winning coarse peak gives an initial
   sub-pixel center estimate.
4. **Full-resolution local refinement:** within a small window (±24 px) around the coarse
   estimate, the reference is re-rendered at full resolution (lightly blurred, not
   edge-extracted) and matched against the unblurred search image intensities across five
   nearby scale/angle values. This removes the quantization error introduced by the
   2x-downsampled coarse stage and typically reduces final error by roughly 1 px.
   `refined_score` is the NCC confidence of this final match; `confidence` is the original
   coarse-stage score.

### Confidence and periodic patterns

NCC confidence is a normalized local similarity score, not a probability. It can be low
(for example, around 0.3–0.4) even for a correct prediction, because the 10:1 resampling,
blur/noise, and edge extraction reduce direct similarity, while repeating DRAM/FinFET
elements create several comparable peaks. The center-distance term is intentionally small
and only ranks near-equal peaks; it never overrides clearly stronger NCC evidence. A low
NCC margin is therefore an ambiguity indicator, not automatically an incorrect result.
`evaluate.py` records the worst observed case together with this root-cause note.

The synthetic degradations in `generate_dataset.py` are deliberately approximate,
procedural SEM analogues rather than calibrated instrument physics: Gaussian PSF
(probe/interaction resolution), Poisson noise (electron counting), Gaussian detector read
noise, impulsive contamination events, secondary-electron edge contrast, and scan-field
vignette/charging streaks.

</details>

## Notes

- **No trained model is used.** `models/` is present only to satisfy the standard submission
  folder convention; the algorithm is entirely classical (OpenCV template matching + NumPy),
  so it needs no weights, no training step, and no GPU.
- **Legacy CLI:** `localizer_core.py` retains its own `--ref/--search` and `--batch/--out`
  arguments if you need to localize a single named pair directly during development;
  `run.py` is the standardized entry point for batch evaluation.
