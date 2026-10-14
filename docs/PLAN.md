# Guitarify v1.0 — Production-Grade Scope & Execution Plan

## Context & Core Upgrade Philosophy
The original PRD mixed a shippable application with multi-year PhD research (3D video hand-tracking, JEPA foundation models, AI conversational teacher), while the initial stripped-down v1 plan over-corrected by cutting features that real guitarists actually need (full-band stem isolation, chord symbols, expressive techniques, rhythm/bar lines, harmonic ghost-note suppression, and Guitar Pro export).

This updated master plan bridges that gap: it defines a **local, modular Python system (CLI + FastAPI Studio UI)** powered by specialized, lightweight audio neural networks and dynamic programming that transforms **either a clean guitar recording OR a full-band music video/song** into an accurate, bar-aligned, expressive guitar tablature (`.gp5`, ASCII, MIDI, JSON) with full scientific benchmarking.

## 1. Reality Check — What Is Actually in `C:\guitarify` Today

| Finding | Evidence | Implication |
| :--- | :--- | :--- |
| **Further along than PRD** | Basic Pitch 0.4.0 already run; `evaluation/evaluate_bp.py` (Hungarian matching), `fp_analysis.py`, `results.csv` exist | PRD status is stale; basic single-file ingestion and pitch evaluation exist |
| **Full GuitarSet is local** | 360 WAV (`GuitarSet/audio-play`, mono-mic) + 360 JAMS | Full 6-fold cross-validation across all 360 files is immediately possible |
| **Current result is N=1** | `results.csv`: Precision 0.65 / Recall 0.87 / F1 0.75 on 71 notes (1 file); 33 False Positives | Precision (overtone/harmonic ghost notes and string squeaks) is the #1 accuracy bottleneck to solve |
| **Rich JAMS Ground Truth** | JAMS files contain: (1) 6 per-string `note_midi` (`data_source` "0".."5"), (2) continuous `pitch_contour` per string, (3) `chord` labels, and (4) `beat_position` grids | Gives us ground truth for **strings/frets, bends/vibrato contours, chord symbols, and bar/beat tracking** with zero manual labeling |
| **No project hygiene** | No git, no `pyproject.toml`, 45k-file venv inside root, hardcoded paths, no unit tests | Fix immediately in Milestone 0 |
| **Hardware & Toolchain** | Python 3.11.9, Ryzen 5 7600X, 15 GB RAM, AMD RX 7700 XT, Node 24 | Use PyTorch + `onnxruntime-directml` so the RX 7700 XT accelerates stem separation and TabNet on Windows natively |

## 2. What Separates a "High-School Toy Tabber" from a Reliable Production Tool

| Toy Transcriber Failure Mode | Guitarify Production Fix | How It Works & How We Validate It |
| :--- | :--- | :--- |
| **1. The "33 False Positives" Ghost-Note Plague** (Basic Pitch triggers on 12th-fret octave overtones, sympathetic open strings, and fret squeaks) | **Harmonic Sieve & Transient Verification Filter** | Suppresses simultaneous notes at +12/+19 semitones that share the fundamental's decay envelope and lack an independent pluck transient or TabNet string posterior; filters <45 ms non-harmonic fret squeaks. Evaluated via FP reduction on GuitarSet. |
| **2. "Solo-Only / Clean-Mic Only" Limitation** (Fails on music videos, backing tracks, or electric guitar with amp tone) | **Stage 0: 6-Stem Isolation (`BS-RoFormer SW`) + Amp-Tone Augmentation** | Strips vocals, drums, bass, and keys from `.mp4`/`.mp3`/YouTube audio (9.05 dB Guitar SDR); trains TabNet with simulated amp overdrive, cab IRs, and room reverb so electric guitars work reliably. |
| **3. Sterile Fret Numbers (`5 - 7 - 9`)** (Misses how a guitarist actually plays the notes and ignores chords) | **Polyphonic Chord Naming + Expressive Techniques (`h`, `p`, `/`, `b`, `~`, `PM`)** | Extracts continuous pitch contours and onset ratios after Viterbi string assignment to label hammer-ons, pull-offs, slides, bends, vibrato, and palm mutes, plus chord symbols (`Am7`, `Gmaj7`) above the staff. |
| **4. Unreadable Millisecond Spacing** (Dumps notes by raw seconds with no rhythm, tempo, or bar lines) | **Beat/Downbeat Grid Quantizer + Dual Timestamps** | Detects BPM and beat grid (`librosa`/`BeatNet`), snaps display notes to 16th/triplet subdivisions with bar lines (`\|`), and validates against GuitarSet `beat_position` annotations while keeping exact millisecond sync for playback. |
| **5. Crashes on Drop-D, Eb, or Capo Songs** (Assumes every guitar is tuned to `E2 A2 D3 G3 B3 E4` at 440 Hz) | **Sub-Semitone Pitch Normalization + Decoder-Likelihood Tuning/Capo Finder** | Corrects detuned guitars (`estimate_tuning`) and evaluates top tunings (Standard, Drop D, Eb, D Standard, DADGAD) + Capo 0–7 by picking the hypothesis that eliminates out-of-range lows and minimizes Viterbi hand-stretch cost. |
| **6. Inconsistent Fingering on Repeated Riffs** (Tabs the same chorus riff three different ways across the song) | **Self-Similarity Riff Consensus + `.gp5` Guitar Pro Export** | Detects repeated 2–4 bar riff blocks via chroma recurrence and pools TabNet probabilities so repeated riffs get identical canonical fingerings. Exports native `.gp5` (via `pyguitarpro`) alongside ASCII, MIDI, and JSON. |

## 3. Final Feasibility & Scope Matrix (v1.0)

| Feature / Component | Technical Verdict | v1.0 Status |
| :--- | :--- | :--- |
| **Stage 0: Video/Song Guitar Isolation** (`ffmpeg` + `BS-RoFormer SW` / `htdemucs_6s`) | Achievable via `audio-separator`; 9.05 dB SDR on guitar stem; DirectML GPU acceleration on RX 7700 XT | **v1 Core** (`--isolate-guitar`) |
| **Stage 1: Pitch + Continuous Contour Extraction** (`Basic Pitch` ONNX + Harmonic Sieve) | Pretrained (<1 MB); adding Overtone/Squeak Post-Filter fixes the 0.65 Precision bottleneck | **v1 Core** |
| **Stage 2: String/Timbre Neural Model** (`TabNet` CRNN in PyTorch) | Trains on 360 GuitarSet CQT caches + electric/amp augmentations in <45 mins on CPU/DirectML | **v1 Core** |
| **Stage 3: Biomechanical Chord-Level Viterbi Decoder** | Evaluates 1-to-6 note polyphonic states with hand-span, shift, and finger-assignment costs | **v1 Headline** |
| **Stage 4: Polyphonic Chord Symbol Naming** (`Am7`, `Dsus4`, `G/B`) | Template + bass-note voicing matcher over active Viterbi states; validated on 360 GuitarSet JAMS `chord` labels | **v1 Core** |
| **Stage 5: Expressive Technique Detection** (`h`, `p`, `/`, `b`, `~`, `PM`, `<12>`) | Post-Viterbi DSP pitch-contour/onset physics + supervised classifier benchmarked on **IDMT-SMT-Guitar** & **Guitar-TECHS** | **v1 Headline** |
| **Stage 6: Rhythm Grid, Bar Lines & Riff Consensus** | Beat/downbeat tracking + self-similarity pooling for repeated riffs; validated on GuitarSet `beat_position` | **v1 Core** |
| **Stage 7: Auto-Tuning / Capo Inference & Confidence Calibration** | Cents-offset correction + multi-tuning Viterbi likelihood ranker; temperature-scaled ECE confidence flags | **v1 Core** |
| **Stage 8: Practice Web Studio & `.gp5` / ASCII / MIDI Export** | FastAPI + Canvas UI: 3-stem mixer (Full / Guitar / Backing), live 2D fretboard with finger numbers (1–4), Click-to-Lock re-decoding, and `pyguitarpro` `.gp5` export | **v1 Core** |
| **3D Video Hand-Tracking / JEPA Foundation Model / AI Chatbot Teacher** | Unvalidated, lacks paired datasets, distracts from core transcription reliability | **Cut Permanently** |

## 4. Upgraded System Architecture & Target Repository Layout

```
guitarify/
├── pyproject.toml
├── README.md
├── LICENSE
├── .gitignore
├── docs/
├── src/guitarify/
│   ├── data/
│   │   ├── guitarset.py          # JAMS loader: notes, strings, pitch_contour, chords, beats
│   │   ├── idmt_techniques.py    # Loader for IDMT-SMT-Guitar / Guitar-TECHS technique clips
│   │   └── augment.py            # Amp overdrive, cab IR, EQ, and noise augmentation for TabNet
│   ├── preprocess/
│   │   ├── separate.py           # Stage 0: ffmpeg video/audio extraction + BS-RoFormer / Demucs
│   │   └── tuning.py             # Cents offset correction + Drop-D / Eb / Capo likelihood scanner
│   ├── sieve.py                  # Harmonic overtone (+12/+19 st) and fret-squeak suppression filter
│   ├── transcribe/
│   │   ├── basic_pitch.py        # Wrapper extracting notes, onset spikes, and continuous contours
│   │   └── tabnet.py             # PyTorch CRNN model (6 strings x 21 classes) with DirectML support
│   ├── tab/
│   │   ├── candidates.py         # Polyphonic chord-state enumeration & span pruning
│   │   ├── costs.py              # Biomechanical transition costs + finger (1-4) assignment
│   │   ├── decode.py             # Constrained Chord-Viterbi + Riff self-similarity consensus
│   │   └── tunings.py            # Standard, Drop D, Eb, D Standard, Drop C, DADGAD + Capo 0-7
│   ├── intelligence/
│   │   ├── chords.py             # Polyphonic chord symbol recognizer (mir_eval compatible)
│   │   ├── techniques.py         # Contour DSP + IDMT classifier for h, p, /, b, ~, PM
│   │   └── rhythm.py             # Beat/downbeat tracking & 16th/triplet bar-line quantizer
│   ├── eval/
│   │   ├── metrics.py            # mir_eval multipitch, Tab P/R/F1, TDR, Chord acc, Technique F1
│   │   ├── calibration.py        # Temperature scaling, ECE, and uncertainty flag-precision curves
│   │   └── benchmark.py          # Single-command reproducible benchmark runner with bootstrap CIs
│   ├── export/
│   │   ├── guitarpro.py          # Native .gp5 exporter via pyguitarpro (notes, techniques, bars)
│   │   ├── ascii_tab.py          # Bar-aligned ASCII tab renderer with chord headers & techniques
│   │   └── midi.py               # Multi-channel MIDI exporter (1 channel per string + pitch bends)
│   ├── cli.py                    # guitarify transcribe in.mp4 --isolate-guitar --format gp5
│   └── web/
│       ├── api.py                # FastAPI endpoints (/transcribe, /redecode, /stems)
│       └── static/               # Canvas TAB + Live 2D Fretboard + 3-Stem Mixer
├── tests/
├── experiments/
└── notebooks/
```

## 5. Upgraded Milestones & Exit Gates

| # | Milestone | Key Deliverables | Strict Exit Gate |
| :--- | :--- | :--- | :--- |
| **M0** | **Repo Hygiene & GPU Setup** (1–2 d) | `git init`, `.gitignore`, `pyproject.toml`, clean `.venv`, `onnxruntime-directml` + PyTorch setup for RX 7700 XT, `pytest` skeleton | `pip install -e .` and `pytest` pass on clean checkout; DirectML device verified |
| **M1** | **Full Dataset Loader & Multi-Metric Harness** (3 d) | `guitarset.py` loading all 360 JAMS (notes, strings, `pitch_contour`, `chord`, `beat_position`); player folds `00`–`05`; `metrics.py` + `docs/METRICS.md` | All 360 files load with `0 <= fret <= 21` assertion; unit tests pass for pitch, tab, chord, and beat metrics |
| **M2** | **Baselines + Harmonic Ghost-Note Sieve** (3 d) | Run Basic Pitch across all 360 files; build error taxonomy; implement `sieve.py` (overtone & fret-squeak filter) | Full 360-file baseline table with CIs; `sieve.py` demonstrably improves Precision over raw Basic Pitch (0.65 -> >0.80) without tanking Recall |
| **M3** | **Biomechanical Chord-Viterbi + Tuning/Riff Solver** (5 d) | Polyphonic chord-state Viterbi (`decode.py`), finger `1..4` assignment, learned vs. biomechanical transition costs, `tuning.py` (Drop-D/Capo inference), riff consensus | **Oracle pitch test:** GT pitches into Viterbi clearly beats lowest-fret baseline on string/fret accuracy; accurately recovers simulated Drop-D/Capo test cases |
| **M4** | **Tone-Robust TabNet CRNN** (5–6 d) | PyTorch CRNN on cached CQT with `augment.py` (electric overdrive/cab IR/reverb); 6-fold player CV | Matches published TabCNN benchmark (~0.75 Tab F1 on clean GuitarSet) and maintains >0.60 F1 on augmented electric tone validation |
| **M5** | **Fusion, Chords, Techniques & Rhythm** (6–7 d) | Combine TabNet emissions + Chord-Viterbi; add `chords.py` (chord symbols), `techniques.py` (`h`, `p`, `/`, `b`, `~`, `PM` via contours + IDMT-SMT-Guitar), `rhythm.py` (bar lines), and ECE calibration | Complete ablation table across all 360 GuitarSet files + Chord Accuracy row + Technique P/R/F1 benchmark table on IDMT-SMT-Guitar / GuitarSet contours |
| **M6** | **Stage 0 Stem Isolation & Pro Exporters (`.gp5`)** (3–4 d) | `separate.py` (`ffmpeg` + `BS-RoFormer SW` / `htdemucs_6s` with bleed gating); `guitarpro.py` (`.gp5` export), multi-channel MIDI, and bar-aligned ASCII | End-to-end CLI turns a full-band `.mp4` music video into a valid `.gp5` file that opens cleanly in Guitar Pro / TuxGuitar with bars, chords, and techniques |
| **M7** | **Interactive Practice Studio Web App & v1.0 Release** (6–8 d) | FastAPI + Canvas UI: 3-stem audio mixer (Full / Isolated Guitar / Backing Track), live 2D fretboard diagram showing fingers `1–4`, confidence highlighting, Click-to-Lock re-decode, A–B loop by riff section, README + Docker | 3 unseen test cases (1 acoustic fingerstyle, 1 electric riff with bends/slides, 1 full-band music video) transcribe, play back in sync, re-decode on click, and export to `.gp5` |

## 6. Verification & Benchmark Suite (`python -m guitarify.eval.benchmark --all`)
1. **Table 1 — Multipitch & Tablature Ablation (360 GuitarSet files, 6-fold Player CV):** Raw Basic Pitch vs. Basic Pitch + Harmonic Sieve vs. Lowest-Fret vs. Biomechanical Chord-Viterbi vs. TabNet Alone vs. **Guitarify Fusion + Riff Consensus**.
2. **Table 2 — Musical Intelligence Benchmark:** Chord Symbol Accuracy (`mir_eval.chord` on 180 GuitarSet `_comp` tracks) and Beat/Bar Alignment F-measure (`mir_eval.beat`).
3. **Table 3 — Expressive Technique P/R/F1:** Evaluated on IDMT-SMT-Guitar + GuitarSet contour-derived bends/vibratos across `Hammer-on`, `Pull-off`, `Slide`, `Bend`, `Vibrato`, and `Palm Mute`.
4. **Table 4 — Calibration & Human-in-the-Loop Gain:** Expected Calibration Error (ECE) + how many neighbor errors are automatically fixed when a user locks 1 low-confidence note in Constrained Viterbi.
