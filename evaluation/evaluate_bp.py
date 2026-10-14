"""
evaluate_bp.py
==============
Full evaluation of Basic Pitch predictions against GuitarSet note_midi ground truth.

Ground truth : GuitarSet/annotations/00_BN1-129-Eb_solo.jams  (namespace=note_midi, all 6 sources)
Predictions  : basicpitch_output/00_BN1-129-Eb_solo_mic_basic_pitch.csv

Matching     : one-to-one, minimum-cost bipartite matching (Hungarian algorithm).
               Cost = onset_distance if within tolerance else infinity.
               Pitch must also be within tolerance for a valid match.

Run from any directory — paths are resolved relative to this file's location:
    python evaluation/evaluate_bp.py
"""

import csv
import os
import pathlib
import jams
import numpy as np
from scipy.optimize import linear_sum_assignment

# ── Portable path resolution ──────────────────────────────────────────────────
_HERE = pathlib.Path(__file__).resolve().parent          # .../guitarify/evaluation/
ROOT  = _HERE.parent                                     # .../guitarify/

GUITARSET_DIR = pathlib.Path(os.getenv("GUITARSET_DIR", str(ROOT / "GuitarSet")))
BP_OUTPUT_DIR = pathlib.Path(os.getenv("BP_OUTPUT_DIR",  str(ROOT / "basicpitch_output")))

JAMS_PATH = GUITARSET_DIR / "annotations" / "00_BN1-129-Eb_solo.jams"
BP_CSV    = BP_OUTPUT_DIR / "00_BN1-129-Eb_solo_mic_basic_pitch.csv"
OUT_DIR   = str(_HERE)
OUT_CSV   = _HERE / "results.csv"
NOTES_CSV = _HERE / "note_matches.csv"
# ─────────────────────────────────────────────────────────────────────────────


def load_ground_truth(jams_path):
    j = jams.load(str(jams_path))
    anns = j.search(namespace="note_midi")
    notes = []
    for src_idx, ann in enumerate(anns):
        for obs in ann.data:
            start = float(obs.time)
            dur   = float(obs.duration)
            pitch = float(obs.value)
            notes.append({"start": start, "end": start + dur, "pitch": pitch, "source": src_idx})
    notes.sort(key=lambda n: n["start"])
    return notes, anns


def load_predictions(csv_path):
    notes = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = None
        for row in reader:
            if not row or all(c.strip() == "" for c in row):
                continue
            if header is None:
                header = row
                continue
            try:
                notes.append({"start": float(row[0]), "end": float(row[1]),
                               "pitch": float(row[2]), "velocity": int(row[3])})
            except (ValueError, IndexError):
                continue
    notes.sort(key=lambda n: n["start"])
    return notes


def match_notes(gt_notes, pred_notes, onset_tol_s, pitch_tol):
    ng  = len(gt_notes)
    np_ = len(pred_notes)
    INF = 1e9
    cost = np.full((ng, np_), INF)
    for gi, gn in enumerate(gt_notes):
        for pi, pn in enumerate(pred_notes):
            if abs(gn["start"] - pn["start"]) <= onset_tol_s and abs(gn["pitch"] - pn["pitch"]) <= pitch_tol:
                cost[gi, pi] = abs(gn["start"] - pn["start"])
    max_n = max(ng, np_)
    padded = np.full((max_n, max_n), INF)
    padded[:ng, :np_] = cost
    row_ind, col_ind = linear_sum_assignment(padded)
    matched, matched_gt, matched_pred = [], set(), set()
    for gi, pi in zip(row_ind, col_ind):
        if gi < ng and pi < np_ and padded[gi, pi] < INF:
            matched.append((gi, pi))
            matched_gt.add(gi)
            matched_pred.add(pi)
    fn_idx = [i for i in range(ng)  if i not in matched_gt]
    fp_idx = [i for i in range(np_) if i not in matched_pred]
    return matched, fn_idx, fp_idx


def compute_metrics(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def evaluate(gt, pred, onset_tol_s, pitch_tol):
    matched, fn_idx, fp_idx = match_notes(gt, pred, onset_tol_s, pitch_tol)
    tp = len(matched)
    fp_n = len(fp_idx)
    fn_n = len(fn_idx)
    prec, rec, f1 = compute_metrics(tp, fp_n, fn_n)
    return tp, fp_n, fn_n, prec, rec, f1, matched, fn_idx, fp_idx


def duration_analysis(gt, pred, matched):
    errors = [abs((gt[gi]["end"]-gt[gi]["start"]) - (pred[pi]["end"]-pred[pi]["start"])) for gi, pi in matched]
    if not errors:
        return float("nan"), float("nan")
    return float(np.mean(errors)), float(np.median(errors))


def simultaneity_report(gt):
    pairs = []
    for i in range(len(gt)):
        for j in range(i+1, len(gt)):
            a, b = gt[i], gt[j]
            if a["start"] < b["end"] and b["start"] < a["end"]:
                pairs.append((i, j))
    return pairs


def save_note_matches(gt, pred, matched, fn_idx, fp_idx, path):
    rows = []
    for gi, pi in matched:
        gn, pn = gt[gi], pred[pi]
        rows.append({"type":"MATCH","gt_start":round(gn["start"],4),"gt_end":round(gn["end"],4),
            "gt_pitch":round(gn["pitch"],3),"gt_source":gn["source"],
            "pred_start":round(pn["start"],4),"pred_end":round(pn["end"],4),
            "pred_pitch":round(pn["pitch"],3),"pred_vel":pn["velocity"],
            "onset_diff_ms":round(abs(gn["start"]-pn["start"])*1000,2),
            "pitch_diff":round(abs(gn["pitch"]-pn["pitch"]),3),
            "dur_err_s":round(abs((gn["end"]-gn["start"])-(pn["end"]-pn["start"])),4)})
    for gi in fn_idx:
        gn = gt[gi]
        rows.append({"type":"FALSE_NEGATIVE","gt_start":round(gn["start"],4),"gt_end":round(gn["end"],4),
            "gt_pitch":round(gn["pitch"],3),"gt_source":gn["source"],
            "pred_start":"","pred_end":"","pred_pitch":"","pred_vel":"",
            "onset_diff_ms":"","pitch_diff":"","dur_err_s":""})
    for pi in fp_idx:
        pn = pred[pi]
        rows.append({"type":"FALSE_POSITIVE","gt_start":"","gt_end":"","gt_pitch":"","gt_source":"",
            "pred_start":round(pn["start"],4),"pred_end":round(pn["end"],4),
            "pred_pitch":round(pn["pitch"],3),"pred_vel":pn["velocity"],
            "onset_diff_ms":"","pitch_diff":"","dur_err_s":""})
    rows.sort(key=lambda r: (r["gt_start"] if r["gt_start"] != "" else 999999,
                              r["pred_start"] if r["pred_start"] != "" else 999999))
    fields = ["type","gt_start","gt_end","gt_pitch","gt_source",
              "pred_start","pred_end","pred_pitch","pred_vel",
              "onset_diff_ms","pitch_diff","dur_err_s"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def save_results(rows, path):
    fields = ["onset_tol_ms","pitch_tol","TP","FP","FN","Precision","Recall","F1"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading ground truth...")
    gt, anns = load_ground_truth(JAMS_PATH)
    print(f"  Total GT notes (all 6 strings): {len(gt)}")
    for i, ann in enumerate(anns):
        print(f"    String {i}: {len(ann.data)} notes")

    print("\nLoading Basic Pitch predictions...")
    pred = load_predictions(BP_CSV)
    print(f"  Total BP predictions: {len(pred)}")

    ov = simultaneity_report(gt)
    print(f"\n=== Simultaneity: {len(ov)} overlapping GT note pairs ===")
    for i, j in ov[:5]:
        a, b = gt[i], gt[j]
        print(f"  str{a['source']} {a['start']:.3f}-{a['end']:.3f} p={a['pitch']:.1f}  X  "
              f"str{b['source']} {b['start']:.3f}-{b['end']:.3f} p={b['pitch']:.1f}")
    if len(ov) > 5:
        print(f"  ... and {len(ov)-5} more.")

    PRIMARY_ONSET, PRIMARY_PITCH = 0.100, 0.5
    tp, fp_n, fn_n, prec, rec, f1, matched, fn_idx, fp_idx = evaluate(gt, pred, PRIMARY_ONSET, PRIMARY_PITCH)
    dur_mean, dur_median = duration_analysis(gt, pred, matched)

    save_note_matches(gt, pred, matched, fn_idx, fp_idx, NOTES_CSV)

    results_rows = []
    onset_tols = [0.050, 0.100, 0.200]
    print(f"\n=== Onset tolerance sweep (pitch tol=+-{PRIMARY_PITCH} MIDI) ===")
    print(f"{'Onset tol':>12} {'TP':>5} {'FP':>5} {'FN':>5} {'Prec':>8} {'Rec':>8} {'F1':>8}")
    for ot in onset_tols:
        tp_,fp_,fn_,pr_,rc_,f_,*_ = evaluate(gt, pred, ot, PRIMARY_PITCH)
        print(f"  +-{int(ot*1000):3d} ms    {tp_:>5} {fp_:>5} {fn_:>5}  {pr_*100:>7.2f}%  {rc_*100:>7.2f}%  {f_*100:>7.2f}%")
        results_rows.append({"onset_tol_ms":int(ot*1000),"pitch_tol":PRIMARY_PITCH,
            "TP":tp_,"FP":fp_,"FN":fn_,"Precision":round(pr_,6),"Recall":round(rc_,6),"F1":round(f_,6)})

    pitch_tols = [0.25, 0.5, 1.0]
    print(f"\n=== Pitch tolerance sweep (onset tol=+-{int(PRIMARY_ONSET*1000)}ms) ===")
    print(f"{'Pitch tol':>12} {'TP':>5} {'FP':>5} {'FN':>5} {'Prec':>8} {'Rec':>8} {'F1':>8}")
    for pt in pitch_tols:
        tp_,fp_,fn_,pr_,rc_,f_,*_ = evaluate(gt, pred, PRIMARY_ONSET, pt)
        print(f"  +-{pt:.2f} MIDI   {tp_:>5} {fp_:>5} {fn_:>5}  {pr_*100:>7.2f}%  {rc_*100:>7.2f}%  {f_*100:>7.2f}%")
        if pt != PRIMARY_PITCH:
            results_rows.append({"onset_tol_ms":int(PRIMARY_ONSET*1000),"pitch_tol":pt,
                "TP":tp_,"FP":fp_,"FN":fn_,"Precision":round(pr_,6),"Recall":round(rc_,6),"F1":round(f_,6)})

    save_results(results_rows, OUT_CSV)

    tp50,fp50,fn50,pr50,rc50,f150,*_ = evaluate(gt, pred, 0.050, 0.5)
    tp200,fp200,fn200,pr200,rc200,f1200,*_ = evaluate(gt, pred, 0.200, 0.5)

    print(f"""
=============================================================
Basic Pitch vs GuitarSet -- Final Report
=============================================================
Ground truth notes  : {len(gt)}
Basic Pitch preds   : {len(pred)}

At +-50ms / +-0.5 MIDI:
  TP={tp50}  FP={fp50}  FN={fn50}
  Precision={pr50*100:.2f}%  Recall={rc50*100:.2f}%  F1={f150*100:.2f}%

At +-100ms / +-0.5 MIDI:
  TP={tp}  FP={fp_n}  FN={fn_n}
  Precision={prec*100:.2f}%  Recall={rec*100:.2f}%  F1={f1*100:.2f}%

At +-200ms / +-0.5 MIDI:
  TP={tp200}  FP={fp200}  FN={fn200}
  Precision={pr200*100:.2f}%  Recall={rc200*100:.2f}%  F1={f1200*100:.2f}%

Pitch tolerance comparison (onset +-100ms):""")
    for pt in [0.25, 0.5, 1.0]:
        tp_,fp_,fn_,pr_,rc_,f_,*_ = evaluate(gt, pred, PRIMARY_ONSET, pt)
        print(f"  +-{pt} MIDI:  Precision={pr_*100:.2f}%  Recall={rc_*100:.2f}%  F1={f_*100:.2f}%")

    print(f"""
Duration error (matched, +-100ms / +-0.5 MIDI):
  Mean   : {dur_mean:.4f} s
  Median : {dur_median:.4f} s

Consistency check:
  TP + FN = {tp} + {fn_n} = {tp+fn_n}  (GT={len(gt)})  {'OK' if tp+fn_n==len(gt) else 'MISMATCH'}
  TP + FP = {tp} + {fp_n} = {tp+fp_n}  (Pred={len(pred)})  {'OK' if tp+fp_n==len(pred) else 'MISMATCH'}

FINAL F1        : {f1*100:.2f}%
FINAL PRECISION : {prec*100:.2f}%
FINAL RECALL    : {rec*100:.2f}%
=============================================================
""")
    print(f"Saved: {OUT_CSV}")
    print(f"Saved: {NOTES_CSV}")


if __name__ == "__main__":
    main()
