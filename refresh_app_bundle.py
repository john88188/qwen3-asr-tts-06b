from __future__ import annotations

import shutil
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
BUNDLE_DIR = Path(__file__).resolve().parent / "app_bundle"

FILES_TO_COPY = [
    "api_server.py",
    "asr_engine.py",
    "tts_engine.py",
    "run_asr.py",
    "run_tts.py",
    "mnt_data/input.wav",
]


def main() -> int:
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    for rel_path in FILES_TO_COPY:
        src = ROOT_DIR / rel_path
        if not src.exists():
            raise FileNotFoundError(f"source file not found: {src}")

        dest = BUNDLE_DIR / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        print(f"copied: {src} -> {dest}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
