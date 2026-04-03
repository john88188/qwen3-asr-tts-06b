from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


ASR_MODEL_REPO = "Qwen/Qwen3-ASR-0.6B"
TTS_MODEL_REPO = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
DEFAULT_TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu121"
TRYCLOUDFLARE_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


def _run(cmd: list[str], *, cwd: Optional[Path] = None, env: Optional[dict[str, str]] = None) -> None:
    print("$", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None, env=env)


def _python_cmd() -> list[str]:
    return [sys.executable, "-m", "pip"]


def _command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def install_system_packages() -> None:
    missing = [name for name in ("ffmpeg", "sox", "git") if not _command_exists(name)]
    if not missing and Path("/usr/lib/x86_64-linux-gnu/libsndfile.so.1").exists():
        print("System packages already available.")
        return

    _run(["apt-get", "update"])
    _run(["apt-get", "install", "-y", "ffmpeg", "git", "libsndfile1", "sox"])


def install_python_packages(repo_dir: str | Path, *, torch_install_mode: str = "auto") -> None:
    repo_dir = Path(repo_dir).resolve()
    requirements_file = repo_dir / "colab" / "requirements-colab.txt"
    if not requirements_file.is_file():
        raise FileNotFoundError(f"requirements file not found: {requirements_file}")

    mode = (torch_install_mode or "auto").strip().lower()
    if mode not in {"auto", "force", "skip"}:
        raise ValueError("torch_install_mode must be one of: auto, force, skip")

    should_install_torch = mode == "force"
    if mode == "auto":
        try:
            import torch  # type: ignore

            should_install_torch = not bool(torch.cuda.is_available())
            print(
                json.dumps(
                    {
                        "torch_version": getattr(torch, "__version__", "unknown"),
                        "cuda_available": bool(torch.cuda.is_available()),
                        "torch_install_mode": mode,
                        "will_reinstall_torch": should_install_torch,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        except Exception:
            should_install_torch = True

    _run(_python_cmd() + ["install", "--upgrade", "pip", "setuptools", "wheel"])

    if should_install_torch:
        _run(
            _python_cmd()
            + [
                "install",
                "--index-url",
                DEFAULT_TORCH_INDEX_URL,
                "torch==2.5.1",
                "torchaudio==2.5.1",
            ]
        )
    else:
        print("Keeping existing torch installation.")

    _run(_python_cmd() + ["install", "-r", str(requirements_file)])
    _run(_python_cmd() + ["install", "--no-deps", "qwen-asr==0.0.1"])


def show_runtime_summary() -> dict[str, object]:
    import torch  # type: ignore

    summary = {
        "python": sys.version.split()[0],
        "torch": getattr(torch, "__version__", "unknown"),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cuda_device_name": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def download_models(
    models_root: str | Path,
    *,
    include_tts: bool = True,
    hf_token: Optional[str] = None,
) -> dict[str, Optional[str]]:
    from huggingface_hub import snapshot_download

    models_root = Path(models_root).resolve()
    asr_dir = models_root / "Qwen3-ASR-0.6B"
    tts_dir = models_root / "Qwen3-TTS-12Hz-0.6B-CustomVoice"
    asr_dir.mkdir(parents=True, exist_ok=True)
    if include_tts:
        tts_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=ASR_MODEL_REPO,
        local_dir=str(asr_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        token=hf_token,
    )
    if include_tts:
        snapshot_download(
            repo_id=TTS_MODEL_REPO,
            local_dir=str(tts_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
            token=hf_token,
        )

    result = {
        "models_root": str(models_root),
        "asr_model_dir": str(asr_dir),
        "tts_model_dir": str(tts_dir) if include_tts else None,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def build_runtime_env(
    *,
    repo_dir: str | Path,
    runtime_root: str | Path,
    models_root: str | Path,
    host: str = "0.0.0.0",
    port: int = 8000,
    enable_tts: bool = True,
    api_key: str = "",
    asr_log_level: str = "INFO",
    tts_default_speaker: str = "Vivian",
    tts_default_language: str = "Auto",
    tts_max_new_tokens: int = 128,
    tts_max_new_tokens_limit: int = 512,
    tts_attn_implementation: Optional[str] = None,
) -> dict[str, str]:
    repo_dir = Path(repo_dir).resolve()
    runtime_root = Path(runtime_root).resolve()
    models_root = Path(models_root).resolve()
    home_dir = runtime_root / "home"
    tmp_dir = runtime_root / "tmp"
    hf_home = runtime_root / "hf_cache"
    xdg_cache = runtime_root / ".cache"
    torch_home = runtime_root / "torch_home"
    pycache_dir = runtime_root / "pycache"
    logs_dir = runtime_root / "logs"

    for path in (runtime_root, home_dir, tmp_dir, hf_home, xdg_cache, torch_home, pycache_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "HOME": str(home_dir),
            "TMPDIR": str(tmp_dir),
            "TMP": str(tmp_dir),
            "TEMP": str(tmp_dir),
            "ASR_TMP_DIR": str(tmp_dir),
            "PYTHONPYCACHEPREFIX": str(pycache_dir),
            "HF_HOME": str(hf_home),
            "HUGGINGFACE_HUB_CACHE": str(hf_home / "hub"),
            "TRANSFORMERS_CACHE": str(hf_home / "transformers"),
            "XDG_CACHE_HOME": str(xdg_cache),
            "NUMBA_CACHE_DIR": str(xdg_cache / "numba"),
            "TORCH_HOME": str(torch_home),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
            "ASR_API_HOST": str(host),
            "ASR_API_PORT": str(int(port)),
            "ASR_MODEL_DIR": str(models_root / "Qwen3-ASR-0.6B"),
            "ENABLE_TTS": "1" if enable_tts else "0",
            "ASR_API_KEY": api_key or "",
            "ASR_LOG_LEVEL": asr_log_level,
            "TTS_DEFAULT_SPEAKER": tts_default_speaker,
            "TTS_DEFAULT_LANGUAGE": tts_default_language,
            "TTS_MAX_NEW_TOKENS": str(int(tts_max_new_tokens)),
            "TTS_MAX_NEW_TOKENS_LIMIT": str(int(tts_max_new_tokens_limit)),
        }
    )

    if enable_tts:
        env["TTS_MODEL_DIR"] = str(models_root / "Qwen3-TTS-12Hz-0.6B-CustomVoice")
    else:
        env.pop("TTS_MODEL_DIR", None)

    if tts_attn_implementation:
        env["TTS_ATTN_IMPLEMENTATION"] = str(tts_attn_implementation)
    else:
        env.pop("TTS_ATTN_IMPLEMENTATION", None)

    if not (repo_dir / "api_server.py").is_file():
        raise FileNotFoundError(f"api_server.py not found under repo dir: {repo_dir}")
    if not Path(env["ASR_MODEL_DIR"]).is_dir():
        raise FileNotFoundError(f"ASR model dir not found: {env['ASR_MODEL_DIR']}")
    if enable_tts and not Path(env["TTS_MODEL_DIR"]).is_dir():
        raise FileNotFoundError(f"TTS model dir not found: {env['TTS_MODEL_DIR']}")

    return env


def wait_for_healthz(url: str, *, timeout_s: int = 900, poll_interval_s: float = 3.0) -> dict[str, object]:
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
                print(json.dumps(payload, ensure_ascii=False, indent=2))
                return payload
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(poll_interval_s)
    raise TimeoutError(f"service did not become healthy before timeout: {last_error}")


def read_log_tail(log_path: str | Path, *, lines: int = 80) -> str:
    path = Path(log_path)
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    parts = text.splitlines()
    return "\n".join(parts[-lines:])


def ensure_cloudflared(runtime_root: str | Path) -> Path:
    runtime_root = Path(runtime_root).resolve()
    bin_dir = runtime_root / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    binary = bin_dir / "cloudflared"
    if binary.is_file():
        binary.chmod(0o755)
        return binary

    url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    _run(["wget", "-O", str(binary), url])
    binary.chmod(0o755)
    return binary


def start_cloudflared_tunnel(
    *,
    runtime_root: str | Path,
    port: int,
    wait_timeout_s: int = 45,
) -> tuple[subprocess.Popen[str], str, str]:
    cloudflared = ensure_cloudflared(runtime_root)
    runtime_root = Path(runtime_root).resolve()
    logs_dir = runtime_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "cloudflared.log"
    log_file = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [
            str(cloudflared),
            "tunnel",
            "--url",
            f"http://127.0.0.1:{int(port)}",
            "--no-autoupdate",
        ],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )

    deadline = time.time() + wait_timeout_s
    public_url = None
    while time.time() < deadline:
        if proc.poll() is not None:
            tail = read_log_tail(log_path)
            raise RuntimeError(f"cloudflared exited early:\n{tail}")
        text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        match = TRYCLOUDFLARE_RE.search(text)
        if match:
            public_url = match.group(0)
            break
        time.sleep(1.0)

    if not public_url:
        proc.terminate()
        tail = read_log_tail(log_path)
        raise TimeoutError(f"cloudflared URL not found in time:\n{tail}")

    return proc, public_url, str(log_path)


@dataclass
class ServiceSession:
    repo_dir: str
    runtime_root: str
    models_root: str
    local_url: str
    health_url: str
    log_path: str
    api_process: subprocess.Popen[str] = field(repr=False)
    public_url: Optional[str] = None
    cloudflared_log_path: Optional[str] = None
    cloudflared_process: Optional[subprocess.Popen[str]] = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Optional[str]]:
        return {
            "repo_dir": self.repo_dir,
            "runtime_root": self.runtime_root,
            "models_root": self.models_root,
            "local_url": self.local_url,
            "health_url": self.health_url,
            "log_path": self.log_path,
            "public_url": self.public_url,
            "cloudflared_log_path": self.cloudflared_log_path,
            "api_pid": str(self.api_process.pid) if self.api_process else None,
            "cloudflared_pid": str(self.cloudflared_process.pid) if self.cloudflared_process else None,
        }


def launch_api(
    *,
    repo_dir: str | Path,
    runtime_root: str | Path,
    models_root: str | Path,
    host: str = "0.0.0.0",
    port: int = 8000,
    enable_tts: bool = True,
    api_key: str = "",
    asr_log_level: str = "INFO",
    tts_default_speaker: str = "Vivian",
    tts_default_language: str = "Auto",
    tts_max_new_tokens: int = 128,
    tts_max_new_tokens_limit: int = 512,
    tts_attn_implementation: Optional[str] = None,
    start_cloudflared_tunnel_flag: bool = False,
    startup_timeout_s: int = 900,
) -> ServiceSession:
    repo_dir = Path(repo_dir).resolve()
    runtime_root = Path(runtime_root).resolve()
    env = build_runtime_env(
        repo_dir=repo_dir,
        runtime_root=runtime_root,
        models_root=models_root,
        host=host,
        port=port,
        enable_tts=enable_tts,
        api_key=api_key,
        asr_log_level=asr_log_level,
        tts_default_speaker=tts_default_speaker,
        tts_default_language=tts_default_language,
        tts_max_new_tokens=tts_max_new_tokens,
        tts_max_new_tokens_limit=tts_max_new_tokens_limit,
        tts_attn_implementation=tts_attn_implementation,
    )

    logs_dir = runtime_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "api_server.log"
    log_file = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "api_server:app",
            "--host",
            str(host),
            "--port",
            str(int(port)),
            "--workers",
            "1",
        ],
        cwd=str(repo_dir),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )

    local_url = f"http://127.0.0.1:{int(port)}"
    health_url = f"{local_url}/healthz"
    try:
        wait_for_healthz(health_url, timeout_s=startup_timeout_s)
    except Exception:
        if proc.poll() is None:
            proc.terminate()
        tail = read_log_tail(log_path)
        raise RuntimeError(f"API startup failed. Log tail:\n{tail}") from None

    cloudflared_proc: Optional[subprocess.Popen[str]] = None
    cloudflared_log_path: Optional[str] = None
    public_url: Optional[str] = None
    if start_cloudflared_tunnel_flag:
        cloudflared_proc, public_url, cloudflared_log_path = start_cloudflared_tunnel(
            runtime_root=runtime_root,
            port=port,
        )

    session = ServiceSession(
        repo_dir=str(repo_dir),
        runtime_root=str(runtime_root),
        models_root=str(Path(models_root).resolve()),
        local_url=local_url,
        health_url=health_url,
        log_path=str(log_path),
        api_process=proc,
        public_url=public_url,
        cloudflared_log_path=cloudflared_log_path,
        cloudflared_process=cloudflared_proc,
    )
    print(json.dumps(session.to_dict(), ensure_ascii=False, indent=2))
    return session


def stop_service(session: ServiceSession) -> None:
    for proc in (session.cloudflared_process, session.api_process):
        if not proc or proc.poll() is not None:
            continue
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def fetch_json(url: str, *, headers: Optional[dict[str, str]] = None, timeout: int = 60) -> dict[str, object]:
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post_json(
    url: str,
    payload: dict[str, object],
    *,
    headers: Optional[dict[str, str]] = None,
    timeout: int = 300,
) -> tuple[int, dict[str, object]]:
    merged_headers = {"Content-Type": "application/json"}
    if headers:
        merged_headers.update(headers)

    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=merged_headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return int(resp.status), json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
