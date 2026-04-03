from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Optional

import requests


def _auth_headers(api_key: str = "") -> dict[str, str]:
    return {"X-API-Key": api_key} if api_key else {}


def _print_json(data: Any) -> Any:
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return data


def gpu_summary() -> dict[str, Any]:
    summary: dict[str, Any] = {}

    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        summary["nvidia_smi_returncode"] = proc.returncode
        summary["nvidia_smi_stdout"] = proc.stdout.strip()
        if proc.stderr.strip():
            summary["nvidia_smi_stderr"] = proc.stderr.strip()
    except FileNotFoundError:
        summary["nvidia_smi_error"] = "nvidia-smi not found"

    try:
        import torch  # type: ignore

        summary["torch_version"] = getattr(torch, "__version__", "unknown")
        summary["cuda_available"] = bool(torch.cuda.is_available())
        summary["cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        summary["cuda_device_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as exc:  # noqa: BLE001
        summary["torch_error"] = str(exc)

    return _print_json(summary)


def healthz(base_url: str, *, api_key: str = "", timeout: int = 120) -> dict[str, Any]:
    resp = requests.get(f"{base_url.rstrip('/')}/healthz", headers=_auth_headers(api_key), timeout=timeout)
    resp.raise_for_status()
    return _print_json(resp.json())


def tts_speakers(base_url: str, *, api_key: str = "", timeout: int = 120) -> dict[str, Any]:
    resp = requests.get(
        f"{base_url.rstrip('/')}/v1/tts/speakers",
        headers=_auth_headers(api_key),
        timeout=timeout,
    )
    resp.raise_for_status()
    return _print_json(resp.json())


def tts_synthesize_to_file(
    base_url: str,
    *,
    output_path: str | Path,
    text: str = "你好，欢迎使用 Qwen3-TTS。",
    speaker: str = "Vivian",
    language: str = "zh",
    max_new_tokens: int = 128,
    api_key: str = "",
    timeout: int = 600,
) -> dict[str, Any]:
    payload = {
        "text": text,
        "speaker": speaker,
        "language": language,
        "max_new_tokens": max_new_tokens,
    }
    resp = requests.post(
        f"{base_url.rstrip('/')}/v1/tts/synthesize_wav",
        json=payload,
        headers=_auth_headers(api_key),
        timeout=timeout,
    )
    resp.raise_for_status()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(resp.content)

    result = {
        "output_path": str(output_path),
        "bytes": len(resp.content),
        "sample_rate": resp.headers.get("X-TTS-Sample-Rate"),
        "speaker": resp.headers.get("X-TTS-Speaker"),
        "language": resp.headers.get("X-TTS-Language"),
    }
    return _print_json(result)


def asr_transcribe_file(
    base_url: str,
    *,
    audio_path: str | Path,
    language: Optional[str] = "zh",
    max_new_tokens: int = 256,
    api_key: str = "",
    timeout: int = 600,
) -> dict[str, Any]:
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"audio file not found: {audio_path}")

    data: dict[str, Any] = {"max_new_tokens": str(int(max_new_tokens))}
    if language:
        data["language"] = language

    with audio_path.open("rb") as file_obj:
        resp = requests.post(
            f"{base_url.rstrip('/')}/v1/asr/transcribe",
            data=data,
            files={"file": (audio_path.name, file_obj, "application/octet-stream")},
            headers=_auth_headers(api_key),
            timeout=timeout,
        )
    resp.raise_for_status()
    return _print_json(resp.json())
