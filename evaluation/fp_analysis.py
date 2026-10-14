"""
fp_analysis.py
==============
False Positive analysis — prints detailed context for each FP detection
from Basic Pitch compared against GuitarSet ground truth.

Paths are resolved relative to this file's location (portable across machines).
Override via environment variables:
    GUITARSET_DIR  — root of the GuitarSet dataset  (default: <repo>/GuitarSet)
    BP_OUTPUT_DIR  — dir holding Basic Pitch CSV output (default: <repo>/basicpitch_output)
"""

import os
import pathlib
import csv
import numpy as np
import jams
from scipy.optimize import linear_sum_assignment

# ── Portable path resolution ──────────────────────────────────────────────────
_HERE = pathlib.Path(__file__).resolve().parent          # .../guitarify/evaluation/
ROOT  = _HERE.parent                                     # .../guitarify/

GUITARSET_DIR = pathlib.Path(os.getenv("GUITARSET_DIR", str(ROOT / "GuitarSet")))
BP_OUTPUT_DIR = pathlib.Path(os.getenv("BP_OUTPUT_DIR",  str(ROOT / "basicpitch_output")))

JAMS_PATH = GUITARSET_DIR / "annotations" / "00_BN1-129-Eb_solo.jams"
BP_CSV    = BP_OUTPUT_DIR / "00_BN1-129-Eb_solo_mic_basic_pitch.csv"
# ─────────────────────────────────────────────────────────────────────────────

j = jams.load(str(JAMS_PATH))
anns = j.search(namespace="note_midi")
gt = []
for src_idx, ann in enumerate(anns):
    for obs in ann.data:
        s = float(obs.time); d = float(obs.duration); p = float(obs.value)
        gt.append({"start": s, "end": s+d, "pitch": p, "source": src_idx})
gt.sort(key=lambda n: n["start"])

pred = []
with open(str(BP_CSV), newline="") as f:
    reader = csv.reader(f); header = None
    for row in reader:
        if not row or all(c.strip()==""  for c in row): continue
        if header is None: header=row; continue
        try: pred.append({"start":float(row[0]),"end":float(row[1]),"pitch":float(row[2]),"velocity":int(row[3])})
        except: pass
pred.sort(key=lambda n: n["start"])

ng, np_ = len(gt), len(pred)
INF = 1e9
cost = np.full((ng, np_), INF)
for gi, gn in enumerate(gt):
    for pi, pn in enumerate(pred):
        if abs(gn["start"]-pn["start"])<=0.1 and abs(gn["pitch"]-pn["pitch"])<=0.5:
            cost[gi,pi] = abs(gn["start"]-pn["start"])
padded = np.full((max(ng,np_),max(ng,np_)), INF)
padded[:ng,:np_] = cost
ri, ci = linear_sum_assignment(padded)
matched_pred = set()
for gi, pi in zip(ri, ci):
    if gi<ng and pi<np_ and padded[gi,pi]<INF:
        matched_pred.add(pi)
fp_indices = [i for i in range(np_) if i not in matched_pred]

NAMES = ["C","C#","D","Eb","E","F","F#","G","Ab","A","Bb","B"]
def mname(m):
    m=int(round(m)); return NAMES[m%12]+str(m//12-1)

print("="*80)
print("FALSE POSITIVE ANALYSIS  (33 FPs vs 71 GT notes)")
print("="*80)
print()

for idx, pi in enumerate(fp_indices):
    pn = pred[pi]
    dur = pn["end"]-pn["start"]
    print("FP #%02d | %.4fs-%.4fs dur=%.3fs  MIDI=%d (%s)  vel=%d" % (
        idx+1, pn["start"], pn["end"], dur, pn["pitch"], mname(pn["pitch"]), pn["velocity"]))

    nearby = sorted(gt, key=lambda g: abs(g["start"]-pn["start"]))[:4]
    print("  Nearest GT notes by onset:")
    for g in nearby:
        od = (pn["start"]-g["start"])*1000
        pd = pn["pitch"]-g["pitch"]
        print("    str%d %.3f-%.3fs p=%.2f (%s)  onset_diff=%+.1fms  pitch_diff=%+.2f" % (
            g["source"], g["start"], g["end"], g["pitch"], mname(g["pitch"]), od, pd))

    overlapping = [g for g in gt if g["start"] < pn["end"] and pn["start"] < g["end"]]
    if overlapping:
        print("  GT notes that OVERLAP IN TIME with this FP:")
        for g in overlapping:
            pd = pn["pitch"]-g["pitch"]
            ov = min(pn["end"],g["end"])-max(pn["start"],g["start"])
            print("    str%d %.3f-%.3fs p=%.2f (%s)  pitch_diff=%+.2f  overlap=%.3fs" % (
                g["source"], g["start"], g["end"], g["pitch"], mname(g["pitch"]), pd, ov))
    else:
        print("  *** NO GT notes overlap in time -- completely phantom note ***")
    print()
