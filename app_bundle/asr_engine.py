import os
from dataclasses import dataclass
from typing import Any
from typing import Optional
import torch
from qwen_asr import Qwen3ASRModel


@dataclass
class ASRConfig:
    model_dir: str = "/models/Qwen3-ASR-0.6B"
    max_new_tokens: int = 512
    max_inference_batch_size: int = 1


class ASREngine:
    def __init__(self, config: Optional[ASRConfig] = None):
        self.config = config or ASRConfig()

        if not os.path.isdir(self.config.model_dir):
            raise FileNotFoundError(f"Model dir not found: {self.config.model_dir}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        device_map = "cuda:0" if self.device == "cuda" else "cpu"

        # Use the official qwen-asr wrapper which provides the Qwen3-ASR model implementation.
        # Passing a local path keeps runtime fully offline (model files are pre-bundled in the image).
        self.model = Qwen3ASRModel.from_pretrained(
            self.config.model_dir,
            dtype=self.dtype,
            device_map=device_map,
            max_inference_batch_size=self.config.max_inference_batch_size,
            max_new_tokens=self.config.max_new_tokens,
        )

    @staticmethod
    def _normalize_language(language: Optional[str]) -> Optional[str]:
        if not language:
            return None
        lang = language.strip()
        # Accept common short codes and map to qwen-asr's language names.
        mapping = {
            "zh": "Chinese",
            "zh-cn": "Chinese",
            "zh-hans": "Chinese",
            "en": "English",
            "yue": "Cantonese",
            "ar": "Arabic",
            "de": "German",
            "fr": "French",
            "es": "Spanish",
            "pt": "Portuguese",
            "id": "Indonesian",
            "it": "Italian",
            "ko": "Korean",
            "ru": "Russian",
            "th": "Thai",
            "vi": "Vietnamese",
            "ja": "Japanese",
            "tr": "Turkish",
            "hi": "Hindi",
            "ms": "Malay",
            "nl": "Dutch",
            "sv": "Swedish",
            "da": "Danish",
            "fi": "Finnish",
            "pl": "Polish",
            "cs": "Czech",
            "fil": "Filipino",
            "fa": "Persian",
            "el": "Greek",
            "ro": "Romanian",
            "hu": "Hungarian",
            "mk": "Macedonian",
        }
        return mapping.get(lang.lower(), lang)

    def _transcribe_call(self, **kwargs: Any):
        # Some qwen-asr versions accept additional kwargs; only pass those supported.
        import inspect

        sig = inspect.signature(self.model.transcribe)
        filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return self.model.transcribe(**filtered)

    def transcribe(
        self,
        audio_path: str,
        *,
        max_new_tokens: int = 512,
        language: Optional[str] = None,
        task: Optional[str] = None,
    ) -> str:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # qwen-asr can take a local path directly; keep it simple and offline.
        # It also supports base64/URL inputs, but we keep file-path for uniformity.
        # `task` is not used by qwen-asr transformers backend; kept for API compatibility.
        language_name = self._normalize_language(language)
        results = self._transcribe_call(
            audio=audio_path,
            language=language_name,
            max_new_tokens=int(max_new_tokens),
        )

        # qwen-asr returns a list of results; each item has `.text`.
        if not results:
            return ""
        first = results[0]
        return getattr(first, "text", str(first)).strip()
