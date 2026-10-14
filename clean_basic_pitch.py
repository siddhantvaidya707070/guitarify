import csv
import os
import pathlib

# ── Portable path resolution ──────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent           # .../guitarify/
# ─────────────────────────────────────────────────────────────────────────────

input_file  = ROOT / "basicpitch_output" / "00_BN1-129-Eb_solo_mic_basic_pitch.csv"
output_file = ROOT / "basicpitch_clean.csv"

with open(input_file, "r") as f:
    reader = csv.reader(f)
    next(reader)

    rows = []

    for row in reader:
        if len(row) >= 4:
            rows.append(row[:4])

with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "start_time_s",
        "end_time_s",
        "pitch_midi",
        "velocity"
    ])

    writer.writerows(rows)

print("Saved", len(rows), "predictions to", output_file)