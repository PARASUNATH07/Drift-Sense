#!/usr/bin/env python3
"""
Drift-Sense: standardized batch entry point.

Usage:
    python run.py <input-dir> <output-dir>

Input-dir contract (any ONE of these layouts is accepted):
  1. Two subfolders: <input-dir>/references/*  and <input-dir>/searches/*
     matched pairwise by identical filename (e.g. ref_0000.png / search_0000.png
     under their respective folders, or references/0000.png / searches/0000.png).
  2. Flat folder with paired filenames sharing a common id, e.g.
     0000_reference.png / 0000_search.png, or ref_0000.png / search_0000.png.

Accepted image formats: .png, .jpg, .jpeg, .tif, .tiff, .bmp (grayscale or
color; color is converted to grayscale automatically).

For every matched (reference, search) pair with shared id <id>, this script
writes <output-dir>/<id>.json containing the predicted match center and
supporting fields, and also writes a single aggregated <output-dir>/results.csv
covering every pair for convenient batch scoring.

No internet access, API keys, GPU, model downloads, or manual configuration
are required -- the localizer is classical (OpenCV/NumPy) with no trained
weights. The models/ folder is intentionally present (per submission
convention) but empty: this solution does not use a learned model.
"""
import csv
import json
import re
import sys
import time
from pathlib import Path

from localizer_core import localize

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _id_from_paired_name(stem: str):
    """Strip a leading/trailing reference|search token, return the shared id."""
    m = re.match(r"^(ref(?:erence)?)[_-]?(.+)$", stem, re.IGNORECASE)
    if m:
        return m.group(2)
    m = re.match(r"^(search)[_-]?(.+)$", stem, re.IGNORECASE)
    if m:
        return m.group(2)
    m = re.match(r"^(.+?)[_-](ref(?:erence)?)$", stem, re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.match(r"^(.+?)[_-](search)$", stem, re.IGNORECASE)
    if m:
        return m.group(1)
    return None


def discover_pairs(input_dir: Path):
    """Return a list of (id, reference_path, search_path), sorted by id."""
    pairs = {}

    ref_dir = None
    search_dir = None
    for cand in ("references", "reference", "refs", "ref"):
        if (input_dir / cand).is_dir():
            ref_dir = input_dir / cand
            break
    for cand in ("searches", "search"):
        if (input_dir / cand).is_dir():
            search_dir = input_dir / cand
            break

    if ref_dir and search_dir:
        def _strip_prefix(stem: str) -> str:
            # Accept identical filenames, or a leading ref_/reference_/search_ token.
            m = re.match(r"^(?:ref(?:erence)?|search)[_-]?(.+)$", stem, re.IGNORECASE)
            return m.group(1) if m else stem

        refs = {_strip_prefix(p.stem): p for p in ref_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
        searches = {_strip_prefix(p.stem): p for p in search_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
        for key in sorted(set(refs) & set(searches)):
            pairs[key] = (refs[key], searches[key])
        missing = set(refs) ^ set(searches)
        if missing:
            print(f"Warning: {len(missing)} unmatched file(s) between references/ and searches/: "
                  f"{sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}", file=sys.stderr)
        return [(k, *pairs[k]) for k in sorted(pairs)]

    # Flat-folder convention: match by shared id after stripping a ref/search token.
    ref_files, search_files = {}, {}
    for p in sorted(input_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
            continue
        stem = p.stem
        pid = _id_from_paired_name(stem)
        if pid is None:
            continue
        lower = stem.lower()
        if "search" in lower:
            search_files[pid] = p
        elif "ref" in lower:
            ref_files[pid] = p

    for key in sorted(set(ref_files) & set(search_files)):
        pairs[key] = (ref_files[key], search_files[key])
    missing = set(ref_files) ^ set(search_files)
    if missing:
        print(f"Warning: {len(missing)} unmatched id(s) in flat input folder: "
              f"{sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}", file=sys.stderr)
    return [(k, *pairs[k]) for k in sorted(pairs)]


def main():
    if len(sys.argv) != 3:
        print("Usage: python run.py <input-dir> <output-dir>", file=sys.stderr)
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    if not input_dir.is_dir():
        print(f"Error: input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = discover_pairs(input_dir)
    if not pairs:
        print(f"Error: no reference/search pairs found under {input_dir}. "
              "Expected references/+searches/ subfolders or paired filenames "
              "(e.g. 0000_reference.png / 0000_search.png).", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(pairs)} pair(s). Writing results to {output_dir}")
    rows = []
    for pid, ref_path, search_path in pairs:
        try:
            result = localize(ref_path, search_path)
        except Exception as exc:  # keep batch running; record the failure explicitly
            print(f"  [{pid}] FAILED: {exc}", file=sys.stderr)
            result = dict(pred_x=None, pred_y=None, confidence=None, refined_score=None,
                          margin=None, scale=None, angle_deg=None, runtime_ms=None,
                          error=str(exc))
        record = dict(id=pid, reference=str(ref_path), search=str(search_path), **result)
        rows.append(record)
        with open(output_dir / f"{pid}.json", "w") as f:
            json.dump(record, f, indent=2, default=float)
        status = "ok" if result.get("pred_x") is not None else "FAILED"
        print(f"  [{pid}] {status}  pred=({record.get('pred_x')}, {record.get('pred_y')})")

    fieldnames = list(rows[0].keys())
    with open(output_dir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Done. {len(rows)} pair(s) processed. Aggregated results: {output_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
