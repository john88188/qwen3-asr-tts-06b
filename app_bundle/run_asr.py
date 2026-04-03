#!/usr/bin/env python3
import argparse
import os
import sys

from asr_engine import ASRConfig, ASREngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline ASR inference for Qwen3-ASR-0.6B")
    parser.add_argument("--model-dir", default="/models/Qwen3-ASR-0.6B", help="Local model directory")
    parser.add_argument("--audio", default="/mnt/data/input.wav", help="Input audio path")
    parser.add_argument("--output", default="/mnt/data/asr_output.txt", help="Output text file path")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max generated tokens")
    parser.add_argument("--language", default=None, help="Optional language hint")
    parser.add_argument("--task", default=None, help="Optional task hint, e.g. transcribe")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not os.path.exists(args.audio):
        print(f"Audio file not found: {args.audio}", file=sys.stderr)
        return 1

    engine = ASREngine(ASRConfig(model_dir=args.model_dir))
    text = engine.transcribe(
        args.audio,
        max_new_tokens=args.max_new_tokens,
        language=args.language,
        task=args.task,
    )
    print(text)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(text + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
