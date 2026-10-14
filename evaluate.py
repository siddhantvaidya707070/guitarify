import csv
import json
import os
import pathlib

# ── Portable path resolution ──────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent           # .../guitarify/
GUITARSET_DIR = pathlib.Path(os.getenv("GUITARSET_DIR", str(ROOT / "GuitarSet")))
# ─────────────────────────────────────────────────────────────────────────────

# Load Basic Pitch predictions
with open(ROOT / "basicpitch_clean.csv", "r") as f:
    reader = csv.DictReader(f)
    predictions = list(reader)

# Load GuitarSet ground truth
with open(GUITARSET_DIR / "annotations" / "00_BN1-129-Eb_solo.jams", "r") as f:
    data = json.load(f)

ground_truth = []

for annotation in data["annotations"]:
    if annotation["namespace"] == "note_midi":

        for note in annotation["data"]:
            ground_truth.append({
                "start": note["time"],
                "pitch": note["value"]
            })

print("Basic Pitch predictions:", len(predictions))
print("GuitarSet notes:", len(ground_truth))

# Matching settings
TIME_TOLERANCE = 0.1
PITCH_TOLERANCE = 0.5

matches = 0

for truth in ground_truth:

    truth_start = truth["start"]
    truth_pitch = truth["pitch"]

    for prediction in predictions:

        pred_start = float(prediction["start_time_s"])
        pred_pitch = float(prediction["pitch_midi"])

        time_difference = abs(truth_start - pred_start)
        pitch_difference = abs(truth_pitch - pred_pitch)

        if (
            time_difference <= TIME_TOLERANCE
            and pitch_difference <= PITCH_TOLERANCE
        ):
            matches += 1
            break

print("Matches:", matches)

recall = matches / len(ground_truth)

print("Recall:", round(recall * 100, 2), "%")