#!/usr/bin/env python3
import base64
import hashlib
import hmac
import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional
from urllib import error as urlerror
from urllib import request as urlrequest

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi import Request
from fastapi.responses import JSONResponse, Response

from asr_engine import ASRConfig, ASREngine
from tts_engine import TTSConfig, TTSEngine

app = FastAPI(title="Qwen3 ASR+TTS Offline API")

_engine: Optional[ASREngine] = None
_engine_lock = threading.Lock()
_tts_engine: Optional[TTSEngine] = None
_tts_engine_lock = threading.Lock()
_tts_init_error: Optional[str] = None
_tts_max_new_tokens_limit: int = 4096

_jobs_lock = threading.Lock()
_jobs: dict[str, "_Job"] = {}

_API_KEY_ENV = "ASR_API_KEY"
_NO_AUTH_PATHS = {
    "/healthz",
    "/openapi.json",
    "/docs",
    "/docs/oauth2-redirect",
    "/redoc",
}

_LOG_LEVEL = (os.environ.get("ASR_LOG_LEVEL") or "INFO").strip().upper()
_logger = logging.getLogger("asr_tts_api")
_logger.setLevel(getattr(logging, _LOG_LEVEL, logging.INFO))


@app.middleware("http")
async def _auth_middleware(request: Request, call_next):
    """
    Optional auth for public exposure (e.g. Cloudflare Tunnel).
    If ASR_API_KEY is set, require it via:
      - X-API-Key: <key>
      - Authorization: Bearer <key>
    """
    request_id = (request.headers.get("x-request-id") or "").strip() or uuid.uuid4().hex
    request.state.request_id = request_id
    client_ip = request.client.host if request.client else "-"
    method = request.method
    path = request.url.path or "/"
    started = time.perf_counter()

    _logger.info(
        "REQ_START id=%s method=%s path=%s client=%s content_length=%s",
        request_id,
        method,
        path,
        client_ip,
        request.headers.get("content-length", "-"),
    )

    def _finish(response: Response) -> Response:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        response.headers.setdefault("X-Request-Id", request_id)
        response.headers.setdefault("X-Process-Time-Ms", f"{elapsed_ms:.1f}")
        _logger.info(
            "REQ_END id=%s method=%s path=%s status=%s elapsed_ms=%.1f",
            request_id,
            method,
            path,
            getattr(response, "status_code", "-"),
            elapsed_ms,
        )
        return response

    api_key = os.environ.get(_API_KEY_ENV, "").strip()
    if not api_key:
        try:
            return _finish(await call_next(request))
        except Exception:
            _logger.exception(
                "REQ_FAIL id=%s method=%s path=%s elapsed_ms=%.1f",
                request_id,
                method,
                path,
                (time.perf_counter() - started) * 1000.0,
            )
            raise

    if path in _NO_AUTH_PATHS:
        try:
            return _finish(await call_next(request))
        except Exception:
            _logger.exception(
                "REQ_FAIL id=%s method=%s path=%s elapsed_ms=%.1f",
                request_id,
                method,
                path,
                (time.perf_counter() - started) * 1000.0,
            )
            raise

    got = (request.headers.get("x-api-key") or "").strip()
    if not got:
        auth = (request.headers.get("authorization") or "").strip()
        if auth.lower().startswith("bearer "):
            got = auth[7:].strip()

    if got != api_key:
        return _finish(JSONResponse(status_code=401, content={"detail": "unauthorized"}))

    try:
        return _finish(await call_next(request))
    except Exception:
        _logger.exception(
            "REQ_FAIL id=%s method=%s path=%s elapsed_ms=%.1f",
            request_id,
            method,
            path,
            (time.perf_counter() - started) * 1000.0,
        )
        raise


def _get_tmp_dir() -> str:
    # WeCom runtime may be read-only except /mnt/data.
    return os.environ.get("ASR_TMP_DIR", "/mnt/data/tmp")


def _ensure_tmp_dir() -> str:
    tmp_dir = _get_tmp_dir()
    os.makedirs(tmp_dir, exist_ok=True)
    return tmp_dir


def _write_bytes_to_tmp(data: bytes, *, suffix: str) -> str:
    tmp_dir = _ensure_tmp_dir()
    with tempfile.NamedTemporaryFile(prefix="asr_", suffix=suffix, dir=tmp_dir, delete=False) as tmp:
        tmp.write(data)
        return tmp.name


def _normalize_suffix(suffix: Optional[str], *, default: str = ".wav") -> str:
    s = (suffix or "").strip()
    if not s:
        return default
    if not s.startswith("."):
        s = "." + s
    return s.lower()


def _guess_suffix_from_data_uri(s: str, fallback: str) -> str:
    if not s.startswith("data:") or "base64," not in s:
        return fallback
    mime = s.split(";", 1)[0][5:].lower()
    mapping = {
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/opus": ".opus",
        "audio/ogg": ".ogg",
        "audio/webm": ".webm",
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "audio/aac": ".aac",
        "audio/mp4": ".m4a",
        "audio/x-m4a": ".m4a",
        "audio/flac": ".flac",
    }
    return mapping.get(mime, fallback)


def _maybe_strip_data_uri(s: str) -> str:
    # Accept "data:audio/wav;base64,...." too.
    if s.startswith("data:") and "base64," in s:
        return s.split("base64,", 1)[1]
    return s


def _request_id_from(request: Optional[Request]) -> str:
    if request is None:
        return "-"
    rid = getattr(request.state, "request_id", None)
    if rid is None:
        return "-"
    return str(rid)


def _parse_bool(value: Any, default: bool, *, field_name: str) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "on"}:
            return True
        if v in {"0", "false", "no", "off"}:
            return False
    raise HTTPException(status_code=400, detail=f"invalid {field_name}")


def _parse_int(value: Any, default: int, *, field_name: str) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid {field_name}: {e}") from e


def _parse_optional_float(value: Any, *, field_name: str) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid {field_name}: {e}") from e


def _parse_optional_int(value: Any, *, field_name: str) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid {field_name}: {e}") from e


def _extract_tts_generation_kwargs(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "do_sample": _parse_bool(payload.get("do_sample"), True, field_name="do_sample"),
        "top_k": _parse_optional_int(payload.get("top_k"), field_name="top_k"),
        "top_p": _parse_optional_float(payload.get("top_p"), field_name="top_p"),
        "temperature": _parse_optional_float(payload.get("temperature"), field_name="temperature"),
        "repetition_penalty": _parse_optional_float(payload.get("repetition_penalty"), field_name="repetition_penalty"),
        "non_streaming_mode": _parse_bool(payload.get("non_streaming_mode"), True, field_name="non_streaming_mode"),
    }


def _run_ffmpeg(cmd: list[str], *, field_name: str) -> None:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail="ffmpeg not found in runtime") from e
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise HTTPException(status_code=400, detail=f"{field_name} ffmpeg failed: {err[:800]}")


def _decode_audio_to_wav_path(input_path: str) -> str:
    """
    Normalize any supported audio format (wav/opus/ogg/mp3/...) to wav for ASR.
    """
    suffix = os.path.splitext(input_path)[1].lower()
    if suffix == ".wav":
        return input_path

    tmp_dir = _ensure_tmp_dir()
    with tempfile.NamedTemporaryFile(prefix="asr_norm_", suffix=".wav", dir=tmp_dir, delete=False) as tmp:
        output_path = tmp.name

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-loglevel",
        "error",
        "-i",
        input_path,
        "-ar",
        "16000",
        "-ac",
        "1",
        output_path,
    ]
    _run_ffmpeg(cmd, field_name="asr input decode")
    return output_path


def _encode_wav_to_opus_bytes(wav_bytes: bytes, *, bitrate: str = "24k") -> bytes:
    tmp_dir = _ensure_tmp_dir()
    in_path = None
    out_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix="tts_in_", suffix=".wav", dir=tmp_dir, delete=False) as f_in:
            in_path = f_in.name
            f_in.write(wav_bytes)
        with tempfile.NamedTemporaryFile(prefix="tts_out_", suffix=".ogg", dir=tmp_dir, delete=False) as f_out:
            out_path = f_out.name

        cmd = [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-loglevel",
            "error",
            "-i",
            in_path,
            "-c:a",
            "libopus",
            "-b:a",
            bitrate,
            "-vbr",
            "on",
            "-application",
            "audio",
            out_path,
        ]
        _run_ffmpeg(cmd, field_name="tts opus encode")
        with open(out_path, "rb") as f:
            return f.read()
    finally:
        for p in (in_path, out_path):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


def _run_tts_infer(
    payload: dict[str, Any],
    *,
    request: Optional[Request] = None,
    route: str = "",
) -> tuple[bytes, int, str, str]:
    if _tts_engine is None:
        raise HTTPException(status_code=503, detail=f"tts engine not ready: {_tts_init_error or 'not initialized'}")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="invalid json body")

    text = str(payload.get("text") or payload.get("chatcontent") or payload.get("chat") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="missing text/chatcontent/chat")

    speaker = payload.get("speaker")
    language = payload.get("language")
    instruct = payload.get("instruct")
    default_max_tokens = int(getattr(_tts_engine.config, "max_new_tokens", 2048))
    max_new_tokens = _parse_int(
        payload.get("max_new_tokens") or payload.get("max_output_length"),
        default_max_tokens,
        field_name="max_new_tokens",
    )
    if max_new_tokens > _tts_max_new_tokens_limit:
        raise HTTPException(
            status_code=400,
            detail=f"max_new_tokens must be <= {_tts_max_new_tokens_limit}",
        )
    generation_kwargs = _extract_tts_generation_kwargs(payload)

    rid = _request_id_from(request)
    text_chars = len(text)
    started = time.perf_counter()
    _logger.info(
        "TTS_START id=%s route=%s text_chars=%s speaker=%s language=%s max_new_tokens=%s",
        rid,
        route or "-",
        text_chars,
        speaker or "-",
        language or "-",
        max_new_tokens,
    )

    try:
        lock_wait_started = time.perf_counter()
        with _tts_engine_lock:
            lock_wait_ms = (time.perf_counter() - lock_wait_started) * 1000.0
            infer_started = time.perf_counter()
            wav_bytes, sample_rate, final_speaker, final_language = _tts_engine.synthesize_custom_voice(
                text,
                speaker=(str(speaker) if speaker is not None else None),
                language=(str(language) if language is not None else None),
                instruct=(str(instruct) if instruct is not None else None),
                max_new_tokens=max_new_tokens,
                generation_kwargs=generation_kwargs,
            )
            infer_ms = (time.perf_counter() - infer_started) * 1000.0
        total_ms = (time.perf_counter() - started) * 1000.0
        _logger.info(
            "TTS_INFER_DONE id=%s route=%s text_chars=%s lock_wait_ms=%.1f infer_ms=%.1f total_ms=%.1f sample_rate=%s wav_bytes=%s speaker=%s language=%s",
            rid,
            route or "-",
            text_chars,
            lock_wait_ms,
            infer_ms,
            total_ms,
            sample_rate,
            len(wav_bytes),
            final_speaker,
            final_language,
        )
        return wav_bytes, sample_rate, final_speaker, final_language
    except HTTPException:
        raise
    except Exception as e:
        _logger.exception(
            "TTS_FAIL id=%s route=%s text_chars=%s elapsed_ms=%.1f err=%s",
            rid,
            route or "-",
            text_chars,
            (time.perf_counter() - started) * 1000.0,
            str(e),
        )
        raise HTTPException(status_code=400, detail=f"tts failed: {e}") from e


@dataclass
class _TranscribeB64Request:
    audio_b64: str
    suffix: str = ".wav"
    max_new_tokens: int = 512
    language: Optional[str] = None
    task: Optional[str] = None


@dataclass
class _TranscribeCallbackRequest:
    audio_b64: str
    callback_url: str
    request_id: str
    suffix: str = ".wav"
    max_new_tokens: int = 512
    language: Optional[str] = None
    task: Optional[str] = None
    meta: Optional[dict[str, Any]] = None
    callback_headers: Optional[dict[str, str]] = None
    callback_timeout_s: float = 10.0
    callback_retries: int = 0
    callback_secret: Optional[str] = None
    callback_use_proxy: bool = False


@dataclass
class _Job:
    request_id: str
    status: str  # pending|running|succeeded|failed
    created_at: float
    updated_at: float
    text: str = ""
    error: Optional[str] = None
    meta: Optional[dict[str, Any]] = None
    callback_url: Optional[str] = None
    callback_ok: Optional[bool] = None
    callback_status_code: Optional[int] = None
    callback_error: Optional[str] = None


def _job_to_dict(job: _Job) -> dict[str, Any]:
    return {
        "request_id": job.request_id,
        "status": job.status,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "text": job.text,
        "error": job.error,
        "meta": job.meta,
        "callback_url": job.callback_url,
        "callback_ok": job.callback_ok,
        "callback_status_code": job.callback_status_code,
        "callback_error": job.callback_error,
    }


def _purge_jobs_unlocked(*, ttl_s: int = 86400, max_jobs: int = 1000) -> None:
    # Best-effort in-memory retention control.
    now = time.time()
    expired = [k for k, v in _jobs.items() if (now - v.updated_at) > ttl_s]
    for k in expired:
        _jobs.pop(k, None)

    if len(_jobs) <= max_jobs:
        return

    # Drop oldest first.
    items = sorted(_jobs.items(), key=lambda kv: kv[1].updated_at)
    for k, _ in items[: max(0, len(_jobs) - max_jobs)]:
        _jobs.pop(k, None)


def _hmac_sha256(secret: str, body: bytes) -> str:
    return hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def _post_json_callback(
    *,
    url: str,
    body: dict[str, Any],
    headers: Optional[dict[str, str]] = None,
    timeout_s: float = 10.0,
    retries: int = 0,
    secret: Optional[str] = None,
    use_proxy: bool = False,
) -> tuple[bool, Optional[int], Optional[str]]:
    payload = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    req_headers: dict[str, str] = {"Content-Type": "application/json; charset=utf-8"}
    if headers:
        for k, v in headers.items():
            if v is None:
                continue
            req_headers[str(k)] = str(v)

    if secret:
        req_headers["X-ASR-Signature"] = "sha256=" + _hmac_sha256(secret, payload)

    request = urlrequest.Request(url=url, data=payload, headers=req_headers, method="POST")

    # Default behavior: ignore env proxies because build-time proxies may be baked into the image.
    opener = (
        urlrequest.build_opener()
        if use_proxy
        else urlrequest.build_opener(urlrequest.ProxyHandler({}))
    )

    last_err: Optional[str] = None
    last_code: Optional[int] = None
    for attempt in range(max(0, int(retries)) + 1):
        try:
            with opener.open(request, timeout=float(timeout_s)) as resp:
                code = int(getattr(resp, "status", resp.getcode()))
                # Drain response for better compatibility; keep a small limit.
                _ = resp.read(4096)
                if 200 <= code < 300:
                    return True, code, None
                last_code = code
                last_err = f"callback http {code}"
        except urlerror.HTTPError as e:
            last_code = int(getattr(e, "code", 0) or 0) or None
            last_err = f"callback http error: {e}"
        except Exception as e:
            last_err = f"callback error: {e}"

        if attempt < max(0, int(retries)):
            time.sleep(min(2 ** attempt, 10))

    return False, last_code, last_err


def _transcribe_and_callback(req: _TranscribeCallbackRequest) -> None:
    # Runs in a background thread. Keep it robust and never raise.
    job: Optional[_Job] = None
    with _jobs_lock:
        job = _jobs.get(req.request_id)
        if job:
            job.status = "running"
            job.updated_at = time.time()

    raw_path = None
    asr_path = None
    text = ""
    err: Optional[str] = None
    try:
        req_suffix = _normalize_suffix(req.suffix, default=".wav")
        req_suffix = _guess_suffix_from_data_uri(req.audio_b64, req_suffix)
        b64 = _maybe_strip_data_uri(req.audio_b64)
        data = base64.b64decode(b64, validate=False)
        raw_path = _write_bytes_to_tmp(data, suffix=req_suffix)
        asr_path = _decode_audio_to_wav_path(raw_path)

        if _engine is None:
            raise RuntimeError("engine not ready")

        with _engine_lock:
            text = _engine.transcribe(
                asr_path,
                max_new_tokens=req.max_new_tokens,
                language=req.language,
                task=req.task,
            )
    except Exception as e:
        err = str(e)
    finally:
        for p in (asr_path, raw_path):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

    now = time.time()
    with _jobs_lock:
        job = _jobs.get(req.request_id)
        if job:
            if err:
                job.status = "failed"
                job.error = err
                job.text = ""
            else:
                job.status = "succeeded"
                job.error = None
                job.text = text
            job.updated_at = now

    callback_body: dict[str, Any] = {
        "request_id": req.request_id,
        "status": "failed" if err else "succeeded",
        "text": "" if err else text,
        "error": err,
        "meta": req.meta,
    }
    ok, code, cb_err = _post_json_callback(
        url=req.callback_url,
        body=callback_body,
        headers=req.callback_headers,
        timeout_s=req.callback_timeout_s,
        retries=req.callback_retries,
        secret=req.callback_secret,
        use_proxy=req.callback_use_proxy,
    )
    with _jobs_lock:
        job = _jobs.get(req.request_id)
        if job:
            job.callback_ok = ok
            job.callback_status_code = code
            job.callback_error = cb_err
            job.updated_at = time.time()


@app.on_event("startup")
def _startup() -> None:
    global _engine, _tts_engine, _tts_init_error, _tts_max_new_tokens_limit
    _tts_max_new_tokens_limit = _parse_int(
        os.environ.get("TTS_MAX_NEW_TOKENS_LIMIT"),
        4096,
        field_name="TTS_MAX_NEW_TOKENS_LIMIT",
    )
    model_dir = os.environ.get("ASR_MODEL_DIR", "/models/Qwen3-ASR-0.6B")
    _engine = ASREngine(ASRConfig(model_dir=model_dir))

    enable_tts = str(os.environ.get("ENABLE_TTS", "1")).strip().lower() not in {"0", "false", "no", "off"}
    if enable_tts:
        tts_model_dir = os.environ.get("TTS_MODEL_DIR", "/models/Qwen3-TTS-12Hz-0.6B-CustomVoice")
        tts_default_speaker = os.environ.get("TTS_DEFAULT_SPEAKER", "Vivian")
        tts_default_language = os.environ.get("TTS_DEFAULT_LANGUAGE", "Auto")
        tts_max_new_tokens = _parse_int(
            os.environ.get("TTS_MAX_NEW_TOKENS"),
            2048,
            field_name="TTS_MAX_NEW_TOKENS",
        )
        if tts_max_new_tokens > _tts_max_new_tokens_limit:
            _logger.warning(
                "TTS_MAX_NEW_TOKENS=%s exceeds TTS_MAX_NEW_TOKENS_LIMIT=%s, clamped.",
                tts_max_new_tokens,
                _tts_max_new_tokens_limit,
            )
            tts_max_new_tokens = _tts_max_new_tokens_limit
        tts_attn_impl = (os.environ.get("TTS_ATTN_IMPLEMENTATION") or "").strip() or None
        try:
            _tts_engine = TTSEngine(
                TTSConfig(
                    model_dir=tts_model_dir,
                    default_speaker=tts_default_speaker,
                    default_language=tts_default_language,
                    max_new_tokens=tts_max_new_tokens,
                    attn_implementation=tts_attn_impl,
                )
            )
            _tts_init_error = None
        except Exception as e:
            _tts_engine = None
            _tts_init_error = str(e)
    else:
        _tts_engine = None
        _tts_init_error = "disabled by ENABLE_TTS=0"

    _ensure_tmp_dir()
    _logger.info(
        "STARTUP_READY asr_device=%s tts_ready=%s tts_device=%s tts_default_max_new_tokens=%s tts_max_new_tokens_limit=%s",
        (_engine.device if _engine is not None else "-"),
        _tts_engine is not None,
        (_tts_engine.device if _tts_engine is not None else "-"),
        (_tts_engine.config.max_new_tokens if _tts_engine is not None else "-"),
        _tts_max_new_tokens_limit,
    )


@app.get("/healthz")
def healthz() -> dict:
    if _engine is None:
        raise HTTPException(status_code=503, detail="engine not ready")
    return {
        "status": "ok",
        "device": _engine.device,
        "asr_ready": _engine is not None,
        "tts_ready": _tts_engine is not None,
        "tts_error": _tts_init_error,
        "tts_device": (_tts_engine.device if _tts_engine is not None else None),
        "tts_max_new_tokens_default": (_tts_engine.config.max_new_tokens if _tts_engine is not None else None),
        "tts_max_new_tokens_limit": _tts_max_new_tokens_limit,
    }


@app.post("/v1/asr/transcribe")
def transcribe(
    file: UploadFile = File(...),
    max_new_tokens: int = Form(512),
    language: Optional[str] = Form(None),
    task: Optional[str] = Form(None),
) -> dict:
    if _engine is None:
        raise HTTPException(status_code=503, detail="engine not ready")

    # Root FS may be read-only; store uploads under ASR_TMP_DIR (default: /mnt/data/tmp).
    suffix = _normalize_suffix(os.path.splitext(file.filename or "")[1] or ".wav")
    raw_path = None
    asr_path = None
    try:
        tmp_dir = _ensure_tmp_dir()
        with tempfile.NamedTemporaryFile(prefix="asr_", suffix=suffix, dir=tmp_dir, delete=False) as tmp:
            shutil.copyfileobj(file.file, tmp)
            raw_path = tmp.name
        asr_path = _decode_audio_to_wav_path(raw_path)

        with _engine_lock:
            text = _engine.transcribe(
                asr_path,
                max_new_tokens=max_new_tokens,
                language=language,
                task=task,
            )
        return {"text": text}
    finally:
        for p in (asr_path, raw_path):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


@app.post("/v1/asr/transcribe_b64")
def transcribe_b64(payload: dict = Body(...)) -> dict:
    """
    JSON API for platforms that cannot send multipart/form-data.
    Body example:
      {"audio_b64":"...","suffix":".wav","max_new_tokens":512,"language":"zh","task":"transcribe"}
    """
    if _engine is None:
        raise HTTPException(status_code=503, detail="engine not ready")

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="invalid json body")

    audio_b64 = payload.get("audio_b64")
    if not audio_b64:
        raise HTTPException(status_code=400, detail="missing audio_b64")

    req_suffix = _normalize_suffix(str(payload.get("suffix") or ".wav"))
    req_suffix = _guess_suffix_from_data_uri(str(audio_b64), req_suffix)
    req = _TranscribeB64Request(
        audio_b64=str(audio_b64),
        suffix=req_suffix,
        max_new_tokens=int(payload.get("max_new_tokens") or 512),
        language=payload.get("language"),
        task=payload.get("task"),
    )

    raw_path = None
    asr_path = None
    try:
        b64 = _maybe_strip_data_uri(req.audio_b64)
        try:
            data = base64.b64decode(b64, validate=False)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"invalid base64: {e}") from e

        raw_path = _write_bytes_to_tmp(data, suffix=req.suffix)
        asr_path = _decode_audio_to_wav_path(raw_path)
        with _engine_lock:
            text = _engine.transcribe(
                asr_path,
                max_new_tokens=req.max_new_tokens,
                language=req.language,
                task=req.task,
            )
        return {"text": text}
    finally:
        for p in (asr_path, raw_path):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


@app.get("/v1/tts/speakers")
def tts_speakers() -> dict:
    if _tts_engine is None:
        raise HTTPException(status_code=503, detail=f"tts engine not ready: {_tts_init_error or 'not initialized'}")

    with _tts_engine_lock:
        speakers = _tts_engine.get_supported_speakers()
        languages = _tts_engine.get_supported_languages()
    return {"speakers": speakers, "languages": languages}


@app.post("/v1/tts/synthesize")
def tts_synthesize(request: Request, payload: dict = Body(...)) -> dict:
    """
    JSON TTS API.
    Body example:
      {
        "text":"你好，欢迎使用 Qwen3-TTS。",
        "speaker":"Vivian",
        "language":"zh",
        "instruct":"可选，0.6B 模型会忽略该字段",
        "max_new_tokens":2048
      }
    """
    route = "/v1/tts/synthesize"
    rid = _request_id_from(request)
    started = time.perf_counter()
    wav_bytes, sample_rate, final_speaker, final_language = _run_tts_infer(
        payload,
        request=request,
        route=route,
    )
    audio_format = str(payload.get("audio_format") or "wav").strip().lower()
    if audio_format not in {"wav", "opus"}:
        raise HTTPException(status_code=400, detail="audio_format must be wav or opus")

    output_bytes = wav_bytes
    format_name = "wav"
    mime_type = "audio/wav"
    if audio_format == "opus":
        opus_bitrate = str(payload.get("opus_bitrate") or "24k").strip()
        encode_started = time.perf_counter()
        output_bytes = _encode_wav_to_opus_bytes(wav_bytes, bitrate=opus_bitrate)
        _logger.info(
            "TTS_ENCODE_OPUS_DONE id=%s route=%s bitrate=%s encode_ms=%.1f opus_bytes=%s",
            rid,
            route,
            opus_bitrate,
            (time.perf_counter() - encode_started) * 1000.0,
            len(output_bytes),
        )
        format_name = "opus"
        mime_type = "audio/ogg"

    _logger.info(
        "TTS_RESP_READY id=%s route=%s format=%s total_ms=%.1f out_bytes=%s",
        rid,
        route,
        format_name,
        (time.perf_counter() - started) * 1000.0,
        len(output_bytes),
    )
    return {
        "audio_b64": base64.b64encode(output_bytes).decode("ascii"),
        "format": format_name,
        "mime_type": mime_type,
        "sample_rate": sample_rate,
        "speaker": final_speaker,
        "language": final_language,
    }


@app.post("/v1/tts/synthesize_wav")
def tts_synthesize_wav(request: Request, payload: dict = Body(...)) -> Response:
    """
    JSON TTS API that returns raw WAV bytes directly.
    """
    route = "/v1/tts/synthesize_wav"
    rid = _request_id_from(request)
    started = time.perf_counter()
    wav_bytes, sample_rate, final_speaker, final_language = _run_tts_infer(
        payload,
        request=request,
        route=route,
    )
    headers = {
        "X-TTS-Sample-Rate": str(sample_rate),
        "X-TTS-Speaker": str(final_speaker),
        "X-TTS-Language": str(final_language),
    }
    _logger.info(
        "TTS_RESP_READY id=%s route=%s format=wav total_ms=%.1f out_bytes=%s",
        rid,
        route,
        (time.perf_counter() - started) * 1000.0,
        len(wav_bytes),
    )
    return Response(content=wav_bytes, media_type="audio/wav", headers=headers)


@app.post("/v1/tts/synthesize_opus")
def tts_synthesize_opus(request: Request, payload: dict = Body(...)) -> Response:
    """
    JSON TTS API that returns Ogg Opus bytes directly.
    """
    route = "/v1/tts/synthesize_opus"
    rid = _request_id_from(request)
    started = time.perf_counter()
    wav_bytes, sample_rate, final_speaker, final_language = _run_tts_infer(
        payload,
        request=request,
        route=route,
    )
    opus_bitrate = str(payload.get("opus_bitrate") or "24k").strip()
    encode_started = time.perf_counter()
    opus_bytes = _encode_wav_to_opus_bytes(wav_bytes, bitrate=opus_bitrate)
    _logger.info(
        "TTS_ENCODE_OPUS_DONE id=%s route=%s bitrate=%s encode_ms=%.1f opus_bytes=%s",
        rid,
        route,
        opus_bitrate,
        (time.perf_counter() - encode_started) * 1000.0,
        len(opus_bytes),
    )
    headers = {
        "X-TTS-Codec": "opus",
        "X-TTS-Sample-Rate": str(sample_rate),
        "X-TTS-Speaker": str(final_speaker),
        "X-TTS-Language": str(final_language),
        "X-TTS-Bitrate": opus_bitrate,
    }
    _logger.info(
        "TTS_RESP_READY id=%s route=%s format=opus total_ms=%.1f out_bytes=%s",
        rid,
        route,
        (time.perf_counter() - started) * 1000.0,
        len(opus_bytes),
    )
    return Response(content=opus_bytes, media_type="audio/ogg", headers=headers)


@app.post("/v1/wecom/infer")
def wecom_infer(payload: dict = Body(...)) -> dict:
    """
    Compatibility endpoint for WeCom-style "input protocol" JSON.
    We treat `chatcontent` (or `chat`) as base64 audio by default.
    """
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="invalid json body")

    # Prefer explicit audio_b64; fall back to chatcontent/chat.
    audio_b64 = payload.get("audio_b64") or payload.get("chatcontent") or payload.get("chat")
    if not audio_b64:
        raise HTTPException(status_code=400, detail="missing audio_b64/chatcontent/chat")

    raw_path = None
    asr_path = None
    try:
        in_suffix = _normalize_suffix(str(payload.get("suffix") or ".wav"))
        in_suffix = _guess_suffix_from_data_uri(str(audio_b64), in_suffix)
        b64 = _maybe_strip_data_uri(str(audio_b64))
        try:
            data = base64.b64decode(b64, validate=False)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"invalid base64: {e}") from e

        max_new_tokens = int(payload.get("max_new_tokens") or payload.get("max_output_length") or 512)
        language = payload.get("language")
        task = payload.get("task")

        raw_path = _write_bytes_to_tmp(data, suffix=in_suffix)
        asr_path = _decode_audio_to_wav_path(raw_path)
        if _engine is None:
            raise HTTPException(status_code=503, detail="engine not ready")
        with _engine_lock:
            text = _engine.transcribe(
                asr_path,
                max_new_tokens=max_new_tokens,
                language=language,
                task=task,
            )

        # Match the typical WeCom "output protocol" shape.
        # Token counters are not meaningful for offline ASR; keep them as 0 unless you have a billing scheme.
        return {"result": text, "input_token": 0, "total_token": 0}
    finally:
        for p in (asr_path, raw_path):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


@app.post("/v1/asr/transcribe_callback")
def transcribe_callback(payload: dict = Body(...)) -> dict:
    """
    Async ASR with HTTP callback.
    Request example (JSON):
      {
        "audio_b64":"...",
        "suffix":".wav",
        "language":"zh",
        "task":"transcribe",
        "max_new_tokens":512,
        "callback_url":"https://example.com/callback",
        "callback_headers":{"Authorization":"Bearer ..."},
        "callback_timeout_s":10,
        "callback_retries":0,
        "callback_secret":"shared-secret",
        "request_id":"optional-client-id",
        "meta":{"biz_id":"xxx"}
      }
    Response:
      {"request_id":"...","status":"accepted"}
    """
    if _engine is None:
        raise HTTPException(status_code=503, detail="engine not ready")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="invalid json body")

    audio_b64 = payload.get("audio_b64")
    if not audio_b64:
        raise HTTPException(status_code=400, detail="missing audio_b64")

    callback_url = payload.get("callback_url")
    if not callback_url:
        raise HTTPException(status_code=400, detail="missing callback_url")
    callback_url = str(callback_url)
    if not (callback_url.startswith("http://") or callback_url.startswith("https://")):
        raise HTTPException(status_code=400, detail="callback_url must start with http:// or https://")

    request_id = payload.get("request_id")
    request_id = str(request_id).strip() if request_id else str(uuid.uuid4())
    if not request_id:
        request_id = str(uuid.uuid4())

    callback_headers = payload.get("callback_headers")
    if callback_headers is not None and not isinstance(callback_headers, dict):
        raise HTTPException(status_code=400, detail="callback_headers must be an object")
    headers_clean: Optional[dict[str, str]] = None
    if isinstance(callback_headers, dict):
        headers_clean = {str(k): str(v) for k, v in callback_headers.items() if v is not None}

    meta = payload.get("meta")
    if meta is not None and not isinstance(meta, dict):
        raise HTTPException(status_code=400, detail="meta must be an object")

    req = _TranscribeCallbackRequest(
        audio_b64=str(audio_b64),
        callback_url=callback_url,
        request_id=request_id,
        suffix=_guess_suffix_from_data_uri(
            str(audio_b64),
            _normalize_suffix(str(payload.get("suffix") or ".wav")),
        ),
        max_new_tokens=int(payload.get("max_new_tokens") or payload.get("max_output_length") or 512),
        language=payload.get("language"),
        task=payload.get("task"),
        meta=meta,
        callback_headers=headers_clean,
        callback_timeout_s=float(payload.get("callback_timeout_s") or 10.0),
        callback_retries=int(payload.get("callback_retries") or 0),
        callback_secret=(str(payload.get("callback_secret")) if payload.get("callback_secret") else None),
        callback_use_proxy=bool(payload.get("callback_use_proxy") or False),
    )

    now = time.time()
    with _jobs_lock:
        _purge_jobs_unlocked()
        # If the same request_id is reused, overwrite.
        _jobs[request_id] = _Job(
            request_id=request_id,
            status="pending",
            created_at=now,
            updated_at=now,
            meta=req.meta,
            callback_url=req.callback_url,
        )

    t = threading.Thread(target=_transcribe_and_callback, args=(req,), daemon=True)
    t.start()

    return {"request_id": request_id, "status": "accepted"}


@app.get("/v1/asr/jobs/{request_id}")
def get_job(request_id: str) -> dict:
    with _jobs_lock:
        job = _jobs.get(request_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        return _job_to_dict(job)


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("ASR_API_HOST", "0.0.0.0")
    port = int(os.environ.get("ASR_API_PORT", "8000"))
    uvicorn.run("api_server:app", host=host, port=port, workers=1)
