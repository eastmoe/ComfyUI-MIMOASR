# Copyright 2025 Xiaomi Corporation.
# ComfyUI node wrapper for MiMo-V2.5-ASR.

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torchaudio

try:
    import folder_paths
except ImportError:
    folder_paths = None

try:
    from comfy import model_management
except ImportError:
    model_management = None


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


ASR_REPO_ID = "XiaomiMiMo/MiMo-V2.5-ASR"
AUDIO_TOKENIZER_REPO_ID = "XiaomiMiMo/MiMo-Audio-Tokenizer"
ASR_DIR_NAME = "MiMo-V2.5-ASR"
AUDIO_TOKENIZER_DIR_NAME = "MiMo-Audio-Tokenizer"
QUANTIZATION_CACHE_DIR_NAME = "MiMo-V2.5-ASR-quantized-cache"

LANGUAGE_TAGS = {
    "auto": "",
    "zh": "<chinese>",
    "en": "<english>",
}

_MODEL_CACHE: dict[tuple[Any, ...], "MimoASRModelHandle"] = {}


def _comfy_models_dir() -> Path:
    if folder_paths is not None and getattr(folder_paths, "models_dir", None):
        return Path(folder_paths.models_dir)
    return Path.cwd() / "models"


def _resolve_model_root(model_root: str) -> Path:
    model_root = (model_root or "auto").strip()
    if not model_root or model_root.lower() == "auto":
        return _comfy_models_dir()
    return Path(model_root).expanduser()


def _looks_like_asr_model(path: Path) -> bool:
    return (
        (path / "config.json").is_file()
        and (
            (path / "model.safetensors.index.json").is_file()
            or (path / "model.safetensors").is_file()
        )
        and (path / "tokenizer.json").is_file()
    )


def _looks_like_audio_tokenizer(path: Path) -> bool:
    return (path / "config.json").is_file() and (path / "model.safetensors").is_file()


def _download_snapshot(
    repo_id: str,
    target_dir: Path,
    *,
    source: str,
    revision: str,
) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: huggingface_hub. Install requirements.txt in ComfyUI first."
        ) from exc

    target_dir.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "local_dir": str(target_dir),
    }
    if revision.strip():
        kwargs["revision"] = revision.strip()

    endpoint = None
    if source == "hf-mirror":
        endpoint = "https://hf-mirror.com"
        kwargs["endpoint"] = endpoint

    try:
        snapshot_download(**kwargs)
    except TypeError:
        if endpoint is None:
            raise
        previous = os.environ.get("HF_ENDPOINT")
        os.environ["HF_ENDPOINT"] = endpoint
        kwargs.pop("endpoint", None)
        try:
            snapshot_download(**kwargs)
        finally:
            if previous is None:
                os.environ.pop("HF_ENDPOINT", None)
            else:
                os.environ["HF_ENDPOINT"] = previous


def _ensure_models(
    model_root: Path,
    *,
    download_missing: bool,
    download_source: str,
    revision: str,
) -> tuple[Path, Path, Path]:
    asr_dir = model_root / ASR_DIR_NAME
    audio_tokenizer_dir = model_root / AUDIO_TOKENIZER_DIR_NAME
    quantization_cache_dir = model_root / QUANTIZATION_CACHE_DIR_NAME

    missing = []
    if not _looks_like_asr_model(asr_dir):
        missing.append((ASR_REPO_ID, asr_dir, "ASR model"))
    if not _looks_like_audio_tokenizer(audio_tokenizer_dir):
        missing.append((AUDIO_TOKENIZER_REPO_ID, audio_tokenizer_dir, "audio tokenizer"))

    if missing and download_source == "offline":
        names = ", ".join(label for _, _, label in missing)
        raise FileNotFoundError(
            f"Missing {names} under {model_root}. Enable download or place the model folders manually."
        )

    if missing and not download_missing:
        names = ", ".join(str(path) for _, path, _ in missing)
        raise FileNotFoundError(
            f"Missing MiMo model folders: {names}. Enable download_missing to fetch them."
        )

    for repo_id, target_dir, _ in missing:
        _download_snapshot(
            repo_id,
            target_dir,
            source=download_source,
            revision=revision,
        )

    if not _looks_like_asr_model(asr_dir):
        raise FileNotFoundError(f"ASR model is incomplete: {asr_dir}")
    if not _looks_like_audio_tokenizer(audio_tokenizer_dir):
        raise FileNotFoundError(f"Audio tokenizer model is incomplete: {audio_tokenizer_dir}")

    quantization_cache_dir.mkdir(parents=True, exist_ok=True)
    return asr_dir, audio_tokenizer_dir, quantization_cache_dir


def _normalize_device(device: str) -> str | None:
    device = (device or "auto").strip()
    if not device or device == "auto":
        return None
    return device


def _cache_key(
    *,
    asr_dir: Path,
    audio_tokenizer_dir: Path,
    quantization_cache_dir: Path | None,
    device: str,
    mel_device: str,
    audio_tokenizer_device: str,
    quantization: str,
    bnb_4bit_compute_dtype: str,
    bnb_4bit_double_quant: bool,
    quantized_linear_chunk_size: int,
    progress: bool,
    progress_interval: float,
) -> tuple[Any, ...]:
    return (
        str(asr_dir.resolve()),
        str(audio_tokenizer_dir.resolve()),
        str(quantization_cache_dir.resolve()) if quantization_cache_dir else None,
        device,
        mel_device,
        audio_tokenizer_device,
        quantization,
        bnb_4bit_compute_dtype,
        bool(bnb_4bit_double_quant),
        int(quantized_linear_chunk_size),
        bool(progress),
        float(progress_interval),
    )


class MimoASRModelHandle:
    def __init__(
        self,
        *,
        model: Any,
        cache_key: tuple[Any, ...],
        asr_dir: Path,
        audio_tokenizer_dir: Path,
        quantization_cache_dir: Path | None,
    ) -> None:
        self.model = model
        self.cache_key = cache_key
        self.asr_dir = asr_dir
        self.audio_tokenizer_dir = audio_tokenizer_dir
        self.quantization_cache_dir = quantization_cache_dir

    @property
    def loaded(self) -> bool:
        return self.model is not None

    @property
    def target_sample_rate(self) -> int:
        return int(self.model.mimo_audio_tokenizer.config.sampling_rate)

    def release(self, *, clear_cuda_cache: bool = True) -> str:
        if self.cache_key in _MODEL_CACHE:
            _MODEL_CACHE.pop(self.cache_key, None)
        if self.model is None:
            return "MiMo ASR model was already released."
        self.model = None
        gc.collect()
        if clear_cuda_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except RuntimeError:
                pass
        if clear_cuda_cache and model_management is not None:
            model_management.soft_empty_cache()
        return "MiMo ASR model weights released."

    def transcribe(
        self,
        audio: torch.Tensor,
        *,
        language: str,
        max_new_tokens: int,
        audio_start_seconds: float,
        audio_duration_seconds: float,
    ) -> str:
        if self.model is None:
            raise RuntimeError("MiMo ASR model has been released. Run the loader node again.")
        duration = None if audio_duration_seconds <= 0 else float(audio_duration_seconds)
        with torch.inference_mode():
            return self.model.asr_sft(
                audio,
                audio_tag=LANGUAGE_TAGS[language],
                max_new_tokens=int(max_new_tokens),
                audio_start_seconds=float(audio_start_seconds),
                audio_duration_seconds=duration,
            )


def _audio_from_comfy(audio: dict[str, Any], *, target_sample_rate: int, batch_index: int) -> torch.Tensor:
    if not isinstance(audio, dict) or "waveform" not in audio:
        raise TypeError("Expected ComfyUI AUDIO input with waveform and sample_rate.")

    waveform = audio["waveform"]
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.as_tensor(waveform)
    waveform = waveform.detach().float().cpu()

    if waveform.ndim == 3:
        batch_index = max(0, min(int(batch_index), waveform.shape[0] - 1))
        waveform = waveform[batch_index]
    elif waveform.ndim not in {1, 2}:
        raise ValueError(f"Unsupported AUDIO waveform shape: {tuple(waveform.shape)}")

    sample_rate = int(audio.get("sample_rate") or target_sample_rate)
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=sample_rate,
            new_freq=target_sample_rate,
        )
    return waveform.contiguous()


class MimoASRLoadModel:
    CATEGORY = "audio/ComfyUI-MIMOASR"
    DESCRIPTION = "Load MiMo-V2.5-ASR and MiMo-Audio-Tokenizer for transcription."
    RETURN_TYPES = ("MIMOASR_MODEL",)
    RETURN_NAMES = ("mimo_asr_model",)
    FUNCTION = "load_model"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_root": (
                    "STRING",
                    {
                        "default": "auto",
                        "tooltip": "Root folder that contains MiMo model folders. auto uses ComfyUI/models.",
                    },
                ),
                "download_missing": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Download missing model folders automatically."},
                ),
                "download_source": (
                    ["huggingface", "hf-mirror", "offline"],
                    {"default": "huggingface", "tooltip": "Model download source."},
                ),
                "revision": (
                    "STRING",
                    {"default": "", "tooltip": "Optional Hugging Face revision, branch, or commit."},
                ),
                "device": (
                    ["auto", "cuda", "cpu"],
                    {"default": "auto", "tooltip": "Device for the ASR language model."},
                ),
                "mel_device": (
                    ["cpu", "cuda", "auto"],
                    {"default": "cpu", "tooltip": "Device for mel spectrogram processing."},
                ),
                "audio_tokenizer_device": (
                    ["auto", "cpu", "cuda"],
                    {"default": "auto", "tooltip": "Device for MiMo-Audio-Tokenizer."},
                ),
                "quantization": (
                    ["none", "int8", "int4", "nf4", "fp4", "fp8"],
                    {"default": "none", "tooltip": "Weight-only quantization mode for the ASR model."},
                ),
                "use_quantization_cache": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Store and reuse quantized weights under ComfyUI/models."},
                ),
                "rebuild_quantization_cache": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Ignore existing quantized cache and rebuild it."},
                ),
                "bnb_4bit_compute_dtype": (
                    ["auto", "float16", "bfloat16", "float32"],
                    {"default": "auto", "tooltip": "Compatibility option for bitsandbytes-style settings."},
                ),
                "bnb_4bit_double_quant": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Compatibility option for bitsandbytes-style settings."},
                ),
                "quantized_linear_chunk_size": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 0,
                        "max": 65536,
                        "step": 64,
                        "tooltip": "Rows per chunk for large quantized Linear layers. Use 0 to disable.",
                    },
                ),
                "progress": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Print detailed load and inference progress to the console."},
                ),
                "progress_interval": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 0.1,
                        "max": 60.0,
                        "step": 0.1,
                        "tooltip": "Minimum seconds between progress updates.",
                    },
                ),
            }
        }

    def load_model(
        self,
        model_root: str,
        download_missing: bool,
        download_source: str,
        revision: str,
        device: str,
        mel_device: str,
        audio_tokenizer_device: str,
        quantization: str,
        use_quantization_cache: bool,
        rebuild_quantization_cache: bool,
        bnb_4bit_compute_dtype: str,
        bnb_4bit_double_quant: bool,
        quantized_linear_chunk_size: int,
        progress: bool,
        progress_interval: float,
    ):
        from src.mimo_audio.mimo_audio import MimoAudio

        root = _resolve_model_root(model_root)
        asr_dir, audio_tokenizer_dir, default_cache_dir = _ensure_models(
            root,
            download_missing=bool(download_missing),
            download_source=download_source,
            revision=revision,
        )
        cache_dir = default_cache_dir if use_quantization_cache and quantization != "none" else None
        key = _cache_key(
            asr_dir=asr_dir,
            audio_tokenizer_dir=audio_tokenizer_dir,
            quantization_cache_dir=cache_dir,
            device=device,
            mel_device=mel_device,
            audio_tokenizer_device=audio_tokenizer_device,
            quantization=quantization,
            bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
            bnb_4bit_double_quant=bool(bnb_4bit_double_quant),
            quantized_linear_chunk_size=int(quantized_linear_chunk_size),
            progress=bool(progress),
            progress_interval=float(progress_interval),
        )
        cached = _MODEL_CACHE.get(key)
        if cached is not None and cached.loaded:
            return (cached,)

        model = MimoAudio(
            model_path=str(asr_dir),
            mimo_audio_tokenizer_path=str(audio_tokenizer_dir),
            device=_normalize_device(device),
            mel_device=mel_device,
            audio_tokenizer_device=audio_tokenizer_device,
            quantization=quantization,
            quantized_linear_chunk_size=int(quantized_linear_chunk_size),
            quantization_cache_dir=str(cache_dir) if cache_dir is not None else None,
            rebuild_quantization_cache=bool(rebuild_quantization_cache),
            bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
            bnb_4bit_use_double_quant=bool(bnb_4bit_double_quant),
            progress=bool(progress),
            progress_interval=float(progress_interval),
        )
        handle = MimoASRModelHandle(
            model=model,
            cache_key=key,
            asr_dir=asr_dir,
            audio_tokenizer_dir=audio_tokenizer_dir,
            quantization_cache_dir=cache_dir,
        )
        _MODEL_CACHE[key] = handle
        return (handle,)


class MimoASRAudioToText:
    CATEGORY = "audio/ComfyUI-MIMOASR"
    DESCRIPTION = "Transcribe ComfyUI AUDIO input with a loaded MiMo ASR model."
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "transcribe"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mimo_asr_model": (
                    "MIMOASR_MODEL",
                    {"tooltip": "Loaded MiMo ASR model from the loader node."},
                ),
                "audio": ("AUDIO", {"tooltip": "ComfyUI native audio input."}),
                "language": (
                    ["auto", "zh", "en"],
                    {"default": "auto", "tooltip": "Language hint for recognition."},
                ),
                "max_new_tokens": (
                    "INT",
                    {
                        "default": 8192,
                        "min": 1,
                        "max": 65536,
                        "step": 1,
                        "tooltip": "Maximum generated text token groups.",
                    },
                ),
                "audio_start_seconds": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 86400.0,
                        "step": 0.1,
                        "tooltip": "Start offset in seconds.",
                    },
                ),
                "audio_duration_seconds": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 86400.0,
                        "step": 0.1,
                        "tooltip": "Duration in seconds. Use 0 for full remaining audio.",
                    },
                ),
                "batch_index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "Batch index to transcribe from batched AUDIO.",
                    },
                ),
            }
        }

    def transcribe(
        self,
        mimo_asr_model: MimoASRModelHandle,
        audio,
        language: str,
        max_new_tokens: int,
        audio_start_seconds: float,
        audio_duration_seconds: float,
        batch_index: int,
    ):
        waveform = _audio_from_comfy(
            audio,
            target_sample_rate=mimo_asr_model.target_sample_rate,
            batch_index=int(batch_index),
        )
        text = mimo_asr_model.transcribe(
            waveform,
            language=language,
            max_new_tokens=int(max_new_tokens),
            audio_start_seconds=float(audio_start_seconds),
            audio_duration_seconds=float(audio_duration_seconds),
        )
        return (text,)


class MimoASRUnloadModel:
    CATEGORY = "audio/ComfyUI-MIMOASR"
    DESCRIPTION = "Release a loaded MiMo ASR model and optionally clear CUDA cache."
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "release_model"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mimo_asr_model": (
                    "MIMOASR_MODEL",
                    {"tooltip": "Loaded MiMo ASR model to release."},
                ),
                "clear_cuda_cache": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Clear CUDA cache after releasing the model."},
                ),
            }
        }

    def release_model(self, mimo_asr_model: MimoASRModelHandle, clear_cuda_cache: bool):
        return (mimo_asr_model.release(clear_cuda_cache=bool(clear_cuda_cache)),)


NODE_CLASS_MAPPINGS = {
    "MimoASRLoadModel": MimoASRLoadModel,
    "MimoASRAudioToText": MimoASRAudioToText,
    "MimoASRUnloadModel": MimoASRUnloadModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MimoASRLoadModel": "MiMo ASR Load Model",
    "MimoASRAudioToText": "MiMo ASR Audio To Text",
    "MimoASRUnloadModel": "MiMo ASR Release Model",
}
