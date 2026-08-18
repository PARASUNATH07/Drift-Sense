#!/usr/bin/env python3
"""Score Drift-Sense batch predictions and save an annotated worst failure."""
import argparse, csv
from pathlib import Path
import cv2
import numpy as np

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--results", required=True); ap.add_argument("--out", required=True)
    a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with open(a.results, newline="") as f: rows=list(csv.DictReader(f))
    errors=np.array([np.hypot(float(r['pred_x'])-float(r['true_x']), float(r['pred_y'])-float(r['true_y'])) for r in rows])
    runtimes=np.array([float(r['runtime_ms']) for r in rows])
    stats={"samples":len(rows), "mean_error_px":float(errors.mean()), "median_error_px":float(np.median(errors)),
           "worst_error_px":float(errors.max()), "runtime_mean_ms":float(runtimes.mean()), "runtime_median_ms":float(np.median(runtimes)),
           **{f"pass_at_{t}px":float((errors<=t).mean()) for t in (5,4,2,1)}}
    with open(out/'summary.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=stats.keys()); w.writeheader();w.writerow(stats)
    worst=int(errors.argmax()); r=rows[worst]; im=cv2.imread(r['search'])
    truth=(round(float(r['true_x'])),round(float(r['true_y']))); pred=(round(float(r['pred_x'])),round(float(r['pred_y'])))
    cv2.drawMarker(im,truth,(0,220,0),cv2.MARKER_CROSS,28,2); cv2.drawMarker(im,pred,(0,0,255),cv2.MARKER_TILTED_CROSS,28,2)
    note=("Likely repeated-pattern ambiguity at low NCC margin." if float(r['margin']) < .03 else "Mismatch likely caused by noise/scale-rotation residual.")
    cv2.putText(im,f"error={errors[worst]:.2f}px; {note}",(15,35),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,255),2,cv2.LINE_AA)
    cv2.imwrite(str(out/'worst_case.png'),im)
    (out/'worst_case_note.txt').write_text(f"Sample {r['id']}: {note}\nerror_px={errors[worst]:.3f}, NCC={r['confidence']}, margin={r['margin']}\nGreen=candidate truth; red=prediction.\n")
    print(stats); print(f"Saved {out/'worst_case.png'}")
if __name__=='__main__': main()
