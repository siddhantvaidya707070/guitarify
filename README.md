# Guitarify 🎸

> **Full plan:** [`docs/PLAN.md`](docs/PLAN.md)

A local, modular Python system (CLI + FastAPI Studio UI) that transforms a clean guitar recording **or** a full-band music video/song into accurate, bar-aligned, expressive guitar tablature (`.gp5`, ASCII, MIDI, JSON) with full scientific benchmarking on GuitarSet.

---

## Pre-M0 Baseline (Current State)

- Basic Pitch 0.4.0 evaluation harness in place (`evaluation/evaluate_bp.py`)
- N=1 result on `00_BN1-129-Eb_solo`: **Precision 0.65 / Recall 0.87 / F1 0.75** (33 FPs, 71 GT notes)
- Full GuitarSet dataset (360 WAV + 360 JAMS) available locally — **not committed to git**

---

## Quick Start (New Machine)

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd guitarify
```

### 2. Run the automated setup script

The script creates a fresh virtual environment, installs all dependencies, and guides you through placing the GuitarSet dataset:

```bash
# Windows (PowerShell)
python setup_env.py

# macOS / Linux
python3 setup_env.py
```

Or set it up manually:

```bash
# Create venv
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate
# Activate (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
# OR for a fully pinned, reproducible environment:
pip install -r requirements-lock.txt
```

### 3. Place the GuitarSet dataset

GuitarSet (360 WAV + 360 JAMS) is **not** included in this repo (audio datasets don't belong in git).

Options to get it on a new machine:
- **Flash drive / network share:** Copy your existing `GuitarSet/` folder into the repo root as `guitarify/GuitarSet/`
- **Download from Zenodo:** Run the included download helper:

```bash
python scripts/download_guitarset.py
```

> The scripts support a `GUITARSET_DIR` environment variable if you want to store GuitarSet outside the repo:
> ```bash
> export GUITARSET_DIR=/data/GuitarSet   # macOS/Linux
> $env:GUITARSET_DIR = "D:\GuitarSet"   # Windows PowerShell
> ```

### 4. Verify the setup

```bash
python evaluation/evaluate_bp.py
```

You should see the N=1 baseline report: `FINAL F1: 74.xx%`

---

## Repository Structure

```
guitarify/
├── evaluate.py              # Quick recall check (pre-M0 script)
├── compare.py               # JAMS ground truth dumper
├── clean_basic_pitch.py     # Basic Pitch CSV cleaner
├── basicpitch_clean.csv     # Cleaned BP output (N=1 baseline)
├── basicpitch_output/       # Raw Basic Pitch CSV outputs
├── evaluation/
│   ├── evaluate_bp.py       # Full Hungarian-matching evaluator
│   ├── fp_analysis.py       # False-positive deep-dive analysis
│   ├── results.csv          # N=1 baseline metrics (committed)
│   └── note_matches.csv     # Per-note match log
├── audio/                   # Small test audio files
├── docs/
│   └── PLAN.md              # Full v1.0 master execution plan
├── requirements.txt         # Top-level dependencies
├── requirements-lock.txt    # Fully pinned lock file
└── setup_env.py             # Cross-device setup script
```

---

## Roadmap

See [`docs/PLAN.md`](docs/PLAN.md) for the full milestone breakdown (M0–M7).

| Milestone | Goal |
| :--- | :--- |
| **M0** | Repo hygiene, GPU setup, `pyproject.toml`, DirectML |
| **M1** | Full 360-file dataset loader + multi-metric harness |
| **M2** | Harmonic Ghost-Note Sieve (Precision: 0.65 → >0.80) |
| **M3** | Biomechanical Chord-Viterbi + Tuning/Capo inference |
| **M4** | Tone-robust TabNet CRNN (electric guitar augmentation) |
| **M5** | Chord symbols, expressive techniques, rhythm grid |
| **M6** | Stem isolation (BS-RoFormer) + `.gp5` export |
| **M7** | Interactive Practice Studio web app + v1.0 release |
