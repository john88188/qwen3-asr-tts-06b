#!/usr/bin/env python3
import argparse
import os
import sys

from tts_engine import TTSConfig, TTSEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline TTS inference for Qwen3-TTS-12Hz-0.6B-CustomVoice")
    parser.add_argument("--model-dir", default="/models/Qwen3-TTS-12Hz-0.6B-CustomVoice", help="Local model directory")
    parser.add_argument("--text", required=True, help="Input text to synthesize")
    parser.add_argument("--speaker", default="Vivian", help="Speaker name, e.g. Vivian / Ryan / Uncle_Fu")
    parser.add_argument("--language", default="Auto", help="Language hint, e.g. Auto / Chinese / English")
    parser.add_argument("--instruct", default=None, help="Optional style instruction (0.6B model ignores this)")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Max generated codec tokens")
    parser.add_argument("--output", default="/mnt/data/tts_output.wav", help="Output wav file path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = TTSConfig(
        model_dir=args.model_dir,
        default_speaker=args.speaker,
        default_language=args.language,
        max_new_tokens=args.max_new_tokens,
    )

    engine = TTSEngine(cfg)
    wav_bytes, sample_rate, speaker, language = engine.synthesize_custom_voice(
        args.text,
        speaker=args.speaker,
        language=args.language,
        instruct=args.instruct,
        max_new_tokens=args.max_new_tokens,
    )

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "wb") as f:
        f.write(wav_bytes)

    print(f"written={args.output}")
    print(f"sample_rate={sample_rate}")
    print(f"speaker={speaker}")
    print(f"language={language}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"tts failed: {e}", file=sys.stderr)
        raise
