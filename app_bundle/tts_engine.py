import io
import inspect
import os
import wave
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch


@dataclass
class TTSConfig:
    model_dir: str = "/models/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    default_speaker: str = "Vivian"
    default_language: str = "Auto"
    max_new_tokens: int = 2048
    attn_implementation: Optional[str] = None


class TTSEngine:
    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or TTSConfig()

        if not os.path.isdir(self.config.model_dir):
            raise FileNotFoundError(f"Model dir not found: {self.config.model_dir}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        device_map = "cuda:0" if self.device == "cuda" else "cpu"

        kwargs: dict[str, Any] = {
            "device_map": device_map,
            "dtype": self.dtype,
        }
        if self.config.attn_implementation:
            kwargs["attn_implementation"] = self.config.attn_implementation

        try:
            from qwen_tts import Qwen3TTSModel
        except Exception as e:
            raise RuntimeError("qwen-tts is not installed, please install it first") from e

        # Use local directory for fully offline runtime.
        self.model = Qwen3TTSModel.from_pretrained(self.config.model_dir, **kwargs)
        self._speaker_alias = self._build_speaker_alias_map()

    @staticmethod
    def _normalize_language(language: Optional[str], default: str) -> str:
        if not language:
            return default
        lang = str(language).strip()
        if not lang:
            return default
        mapping = {
            "auto": "Auto",
            "zh": "Chinese",
            "zh-cn": "Chinese",
            "zh-hans": "Chinese",
            "en": "English",
            "ja": "Japanese",
            "ko": "Korean",
            "de": "German",
            "fr": "French",
            "ru": "Russian",
            "pt": "Portuguese",
            "es": "Spanish",
            "it": "Italian",
        }
        return mapping.get(lang.lower(), lang)

    @staticmethod
    def _canonical_speaker(name: str) -> str:
        parts = [p for p in str(name).replace("-", "_").split("_") if p]
        if not parts:
            return ""
        out: list[str] = []
        for p in parts:
            if p.lower() in {"fu", "anna"}:
                out.append(p.capitalize())
            else:
                out.append(p[:1].upper() + p[1:].lower())
        return "_".join(out)

    def _build_speaker_alias_map(self) -> dict[str, str]:
        aliases: dict[str, str] = {}
        supported = self.get_supported_speakers()
        for spk in supported:
            canonical = self._canonical_speaker(spk)
            aliases[spk.lower()] = canonical
            aliases[canonical.lower()] = canonical
            aliases[canonical.replace("_", "").lower()] = canonical
            aliases[canonical.replace("_", "-").lower()] = canonical
        return aliases

    def _normalize_speaker(self, speaker: Optional[str]) -> str:
        val = (speaker or "").strip()
        if not val:
            val = self.config.default_speaker
        key = val.lower()
        if key in self._speaker_alias:
            return self._speaker_alias[key]
        return self._canonical_speaker(val)

    @staticmethod
    def _to_wav_bytes(wav: Any, sample_rate: int) -> bytes:
        arr = np.asarray(wav, dtype=np.float32)
        if arr.ndim > 1:
            arr = np.mean(arr, axis=-1)
        arr = np.clip(arr, -1.0, 1.0)
        pcm16 = (arr * 32767.0).astype(np.int16)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(sample_rate))
            wf.writeframes(pcm16.tobytes())
        return buf.getvalue()

    def get_supported_speakers(self) -> list[str]:
        speakers = self.model.get_supported_speakers() or []
        out: list[str] = []
        for item in speakers:
            c = self._canonical_speaker(item)
            if c and c not in out:
                out.append(c)
        return out

    def get_supported_languages(self) -> list[str]:
        langs = self.model.get_supported_languages() or []
        out: list[str] = []
        for item in langs:
            if not item:
                continue
            normalized = str(item).strip().lower()
            if normalized and normalized not in out:
                out.append(normalized)
        return out

    def _generate_custom_voice_call(self, **kwargs: Any):
        sig = inspect.signature(self.model.generate_custom_voice)
        filtered: dict[str, Any] = {}
        for key, value in kwargs.items():
            if key not in sig.parameters:
                continue
            if value is None:
                continue
            filtered[key] = value
        return self.model.generate_custom_voice(**filtered)

    def synthesize_custom_voice(
        self,
        text: str,
        *,
        speaker: Optional[str] = None,
        language: Optional[str] = None,
        instruct: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        generation_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[bytes, int, str, str]:
        content = (text or "").strip()
        if not content:
            raise ValueError("text is empty")

        final_speaker = self._normalize_speaker(speaker)
        final_language = self._normalize_language(language, self.config.default_language)
        gen_kwargs: dict[str, Any] = dict(generation_kwargs or {})
        gen_kwargs["max_new_tokens"] = int(max_new_tokens or self.config.max_new_tokens)

        wavs, sample_rate = self._generate_custom_voice_call(
            text=content,
            language=final_language,
            speaker=final_speaker,
            instruct=instruct,
            **gen_kwargs,
        )
        if not wavs:
            raise RuntimeError("TTS returned empty audio")

        wav_bytes = self._to_wav_bytes(wavs[0], int(sample_rate))
        return wav_bytes, int(sample_rate), final_speaker, final_language
