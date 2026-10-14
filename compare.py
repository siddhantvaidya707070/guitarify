import json
import os
import pathlib

# ── Portable path resolution ──────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parent           # .../guitarify/
GUITARSET_DIR = pathlib.Path(os.getenv("GUITARSET_DIR", str(ROOT / "GuitarSet")))
# ─────────────────────────────────────────────────────────────────────────────

with open(GUITARSET_DIR / "annotations" / "00_BN1-129-Eb_solo.jams", "r") as f:
    data = json.load(f)

print("Source | Start | End | MIDI")
print("-" * 40)

for annotation in data["annotations"]:
    if annotation["namespace"] == "note_midi":
        source = annotation["annotation_metadata"]["data_source"]

        for note in annotation["data"]:
            start = note["time"]
            end = start + note["duration"]
            midi = note["value"]

            print(f"{source} | {start:.3f} | {end:.3f} | {midi:.2f}")