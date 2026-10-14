#!/usr/bin/env python3
"""
setup_env.py — Guitarify cross-device bootstrap script
=======================================================
Run this once after cloning the repo on a new machine.

  python setup_env.py            # creates .venv + installs requirements
  python setup_env.py --locked   # installs from the fully-pinned lock file
  python setup_env.py --verify   # only checks whether setup looks correct

What it does:
  1. Checks Python version (requires 3.10+, tested on 3.11.9)
  2. Creates a fresh virtual environment at .venv/
  3. Upgrades pip + installs wheel/setuptools
  4. Installs dependencies from requirements.txt (or requirements-lock.txt)
  5. Verifies all critical imports work
  6. Prints instructions for placing the GuitarSet dataset
"""

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).resolve().parent
VENV_DIR     = REPO_ROOT / ".venv"
REQ_FILE     = REPO_ROOT / "requirements.txt"
REQ_LOCK     = REPO_ROOT / "requirements-lock.txt"

REQUIRED_PACKAGES = [
    "basic_pitch",
    "tensorflow",
    "librosa",
    "jams",
    "mir_eval",
    "pretty_midi",
    "scipy",
    "pandas",
    "matplotlib",
    "numpy",
]

GUITARSET_EXPECTED = REPO_ROOT / "GuitarSet"
# ─────────────────────────────────────────────────────────────────────────────


def banner(msg: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run(cmd, check=True, **kwargs)


def get_venv_python() -> Path:
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def get_venv_pip() -> Path:
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "pip.exe"
    return VENV_DIR / "bin" / "pip"


def check_python_version() -> None:
    banner("Checking Python version")
    major, minor = sys.version_info[:2]
    print(f"  Python {major}.{minor}.{sys.version_info[2]} — {sys.executable}")
    if (major, minor) < (3, 10):
        print(f"\n  ❌  Python 3.10+ required (you have {major}.{minor}). Aborting.")
        sys.exit(1)
    print("  ✅  Version OK")


def create_venv() -> None:
    banner("Creating virtual environment at .venv/")
    if VENV_DIR.exists():
        print(f"  ⚠️   .venv/ already exists — skipping creation.")
        print("  If you want a fresh venv, delete .venv/ and re-run this script.")
        return
    run([sys.executable, "-m", "venv", str(VENV_DIR)])
    print("  ✅  Virtual environment created")


def upgrade_pip() -> None:
    banner("Upgrading pip / setuptools / wheel")
    pip = get_venv_pip()
    run([str(pip), "install", "--upgrade", "pip", "setuptools", "wheel"])


def install_requirements(use_lock: bool) -> None:
    req_path = REQ_LOCK if use_lock else REQ_FILE
    if not req_path.exists():
        print(f"\n  ❌  {req_path} not found. Aborting.")
        sys.exit(1)
    label = "pinned lock file" if use_lock else "requirements.txt"
    banner(f"Installing dependencies from {label}")
    pip = get_venv_pip()
    run([str(pip), "install", "-r", str(req_path)])
    print("  ✅  Dependencies installed")


def verify_imports() -> bool:
    banner("Verifying critical imports")
    python = get_venv_python()
    all_ok = True
    for pkg in REQUIRED_PACKAGES:
        result = subprocess.run(
            [str(python), "-c", f"import {pkg}; print({pkg}.__version__ if hasattr({pkg}, '__version__') else 'ok')"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"  ✅  {pkg:<20} {version}")
        else:
            print(f"  ❌  {pkg:<20} FAILED — {result.stderr.strip()[:80]}")
            all_ok = False
    return all_ok


def check_guitarset() -> None:
    banner("Checking GuitarSet dataset")
    guitarset_dir_env = os.getenv("GUITARSET_DIR")
    if guitarset_dir_env:
        gs_path = Path(guitarset_dir_env)
        print(f"  GUITARSET_DIR env override: {gs_path}")
    else:
        gs_path = GUITARSET_EXPECTED

    if not gs_path.exists():
        print(f"\n  ⚠️   GuitarSet not found at: {gs_path}")
        print("""
  The GuitarSet dataset (360 WAV + 360 JAMS) is not included in the repo
  because audio datasets don't belong in git.

  To get it on this machine, choose one option:

  Option A — Copy from your original device:
    Copy your GuitarSet/ folder into the repo root:
      guitarify/GuitarSet/annotations/   ← 360 .jams files
      guitarify/GuitarSet/audio-play/    ← 360 .wav files

  Option B — Use a custom path (e.g. on another drive):
    Set GUITARSET_DIR before running scripts:
      Windows:  $env:GUITARSET_DIR = "D:\\GuitarSet"
      Mac/Linux: export GUITARSET_DIR=/data/GuitarSet

  Option C — Download from Zenodo (run after venv is active):
    python scripts/download_guitarset.py
""")
    else:
        anns = list((gs_path / "annotations").glob("*.jams")) if (gs_path / "annotations").exists() else []
        wavs = list((gs_path / "audio-play").glob("*.wav"))  if (gs_path / "audio-play").exists()  else []
        print(f"  ✅  GuitarSet found: {len(anns)} JAMS, {len(wavs)} WAVs")
        if len(anns) < 360 or len(wavs) < 360:
            print("  ⚠️   Dataset appears incomplete (expected 360 of each)")


def print_activation_instructions() -> None:
    banner("Setup complete! Activate your environment")
    if platform.system() == "Windows":
        print("""
  PowerShell:  .venv\\Scripts\\Activate.ps1
  CMD:         .venv\\Scripts\\activate.bat
""")
    else:
        print("""
  bash/zsh:    source .venv/bin/activate
""")
    print("  Then verify with:  python evaluation/evaluate_bp.py")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Guitarify cross-device setup")
    parser.add_argument("--locked",  action="store_true",
                        help="Install from requirements-lock.txt (fully pinned)")
    parser.add_argument("--verify",  action="store_true",
                        help="Only run import verification and GuitarSet check")
    args = parser.parse_args()

    print("\n🎸  Guitarify Setup Script")
    print(f"    Repo root : {REPO_ROOT}")
    print(f"    Platform  : {platform.system()} {platform.machine()}")

    check_python_version()

    if args.verify:
        if not VENV_DIR.exists():
            print("\n  ❌  .venv/ not found. Run setup_env.py without --verify first.")
            sys.exit(1)
        ok = verify_imports()
        check_guitarset()
        sys.exit(0 if ok else 1)

    create_venv()
    upgrade_pip()
    install_requirements(use_lock=args.locked)
    ok = verify_imports()
    check_guitarset()

    if ok:
        print_activation_instructions()
    else:
        print("\n  ⚠️   Some imports failed. Check the errors above and re-run.")
        sys.exit(1)


if __name__ == "__main__":
    main()
