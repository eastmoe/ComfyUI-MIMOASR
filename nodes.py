# Copyright 2025 Xiaomi Corporation.
# ComfyUI node wrapper for MiMo-V2.5-ASR.

from __future__ import annotations

import gc
import json
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
_SILERO_VAD_CACHE: dict[str, Any] = {}


def _comfy_models_dir() -> Path:
    if folder_paths is not None and getattr(folder_paths, "models_dir", None):
        return Path(folder_paths.models_dir)
    return Path.cwd() / "models"


def _comfy_output_dir() -> Path:
    if folder_paths is not None and hasattr(folder_paths, "get_output_directory"):
        return Path(folder_paths.get_output_directory())
    return Path.cwd() / "output"


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


def _select_comfy_waveform(audio: dict[str, Any], *, batch_index: int) -> tuple[torch.Tensor, int]:
    if not isinstance(audio, dict) or "waveform" not in audio:
        raise TypeError("Expected ComfyUI AUDIO input with waveform and sample_rate.")

    waveform = audio["waveform"]
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.as_tensor(waveform)
    waveform = waveform.detach().float().cpu()

    if waveform.ndim == 3:
        batch_index = max(0, min(int(batch_index), waveform.shape[0] - 1))
        waveform = waveform[batch_index]
    elif waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim != 2:
        raise ValueError(f"Unsupported AUDIO waveform shape: {tuple(waveform.shape)}")

    sample_rate = int(audio.get("sample_rate") or 16000)
    return waveform.contiguous(), sample_rate


def _audio_from_comfy(audio: dict[str, Any], *, target_sample_rate: int, batch_index: int) -> torch.Tensor:
    waveform, sample_rate = _select_comfy_waveform(audio, batch_index=batch_index)
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=sample_rate,
            new_freq=target_sample_rate,
        )
    return waveform.contiguous()


def _resolve_vad_model_path(vad_model_path: str) -> Path:
    vad_model_path = (vad_model_path or "auto").strip()
    if not vad_model_path or vad_model_path.lower() == "auto":
        return REPO_ROOT / "vad" / "silero_vad.jit"
    path = Path(vad_model_path).expanduser()
    if path.is_absolute():
        return path
    repo_relative = REPO_ROOT / path
    if repo_relative.exists():
        return repo_relative
    return path.resolve()


def _load_silero_vad_model(vad_model_path: str):
    model_path = _resolve_vad_model_path(vad_model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Silero VAD model not found: {model_path}")
    cache_key = str(model_path.resolve())
    cached = _SILERO_VAD_CACHE.get(cache_key)
    if cached is not None:
        return cached
    model = torch.jit.load(str(model_path), map_location="cpu")
    model.eval()
    _SILERO_VAD_CACHE[cache_key] = model
    return model


def _reset_silero_states(model) -> None:
    try:
        model.reset_states()
    except Exception:
        pass


def _silero_predict(model, chunk: torch.Tensor, sampling_rate: int) -> float:
    try:
        out = model(chunk, sampling_rate)
    except RuntimeError:
        out = model(chunk.unsqueeze(0), sampling_rate)
    return float(out.item())


@torch.no_grad()
def _get_speech_timestamps(
    audio: torch.Tensor,
    model,
    *,
    threshold: float,
    sampling_rate: int,
    min_speech_duration_ms: int,
    max_speech_duration_s: float,
    min_silence_duration_ms: int,
    speech_pad_ms: int,
    neg_threshold: float | None = None,
) -> list[dict[str, int]]:
    if audio.ndim != 1:
        audio = audio.squeeze()
    if audio.ndim != 1:
        raise ValueError("Silero VAD expects a mono waveform.")
    if sampling_rate not in {8000, 16000}:
        raise ValueError("Silero VAD supports 8000 Hz or 16000 Hz audio.")

    window_size_samples = 512 if sampling_rate == 16000 else 256
    max_speech_duration_s = float(max_speech_duration_s or 0.0)
    if max_speech_duration_s <= 0:
        max_speech_duration_s = float("inf")

    _reset_silero_states(model)
    min_speech_samples = sampling_rate * int(min_speech_duration_ms) / 1000
    speech_pad_samples = sampling_rate * int(speech_pad_ms) / 1000
    max_speech_samples = sampling_rate * max_speech_duration_s - window_size_samples - 2 * speech_pad_samples
    min_silence_samples = sampling_rate * int(min_silence_duration_ms) / 1000
    min_silence_samples_at_max_speech = sampling_rate * 98 / 1000
    audio_length_samples = int(audio.shape[-1])
    if audio_length_samples == 0:
        return []

    speech_probs: list[float] = []
    for current_start_sample in range(0, audio_length_samples, window_size_samples):
        chunk = audio[current_start_sample : current_start_sample + window_size_samples]
        if chunk.shape[-1] < window_size_samples:
            chunk = torch.nn.functional.pad(chunk, (0, window_size_samples - chunk.shape[-1]))
        speech_probs.append(_silero_predict(model, chunk, sampling_rate))

    triggered = False
    speeches: list[dict[str, int]] = []
    current_speech: dict[str, int] = {}
    if neg_threshold is None:
        neg_threshold = max(float(threshold) - 0.15, 0.01)
    temp_end = 0
    prev_end = 0
    next_start = 0
    possible_ends: list[tuple[int, int]] = []

    for i, speech_prob in enumerate(speech_probs):
        cur_sample = window_size_samples * i

        if speech_prob >= threshold and temp_end:
            silence_duration = cur_sample - temp_end
            if silence_duration > min_silence_samples_at_max_speech:
                possible_ends.append((temp_end, silence_duration))
            temp_end = 0
            if next_start < prev_end:
                next_start = cur_sample

        if speech_prob >= threshold and not triggered:
            triggered = True
            current_speech["start"] = cur_sample
            continue

        if triggered and cur_sample - current_speech["start"] > max_speech_samples:
            if possible_ends:
                prev_end, silence_duration = max(possible_ends, key=lambda item: item[1])
                current_speech["end"] = prev_end
                speeches.append(current_speech)
                current_speech = {}
                next_start = prev_end + silence_duration
                if next_start < cur_sample:
                    current_speech["start"] = next_start
                else:
                    triggered = False
                prev_end = next_start = temp_end = 0
                possible_ends = []
            else:
                current_speech["end"] = cur_sample
                speeches.append(current_speech)
                current_speech = {}
                prev_end = next_start = temp_end = 0
                triggered = False
                possible_ends = []
            continue

        if speech_prob < neg_threshold and triggered:
            if not temp_end:
                temp_end = cur_sample
            if cur_sample - temp_end < min_silence_samples:
                continue
            current_speech["end"] = temp_end
            if current_speech["end"] - current_speech["start"] > min_speech_samples:
                speeches.append(current_speech)
            current_speech = {}
            prev_end = next_start = temp_end = 0
            triggered = False
            possible_ends = []

    if current_speech and audio_length_samples - current_speech["start"] > min_speech_samples:
        current_speech["end"] = audio_length_samples
        speeches.append(current_speech)

    for i, speech in enumerate(speeches):
        if i == 0:
            speech["start"] = int(max(0, speech["start"] - speech_pad_samples))
        if i != len(speeches) - 1:
            silence_duration = speeches[i + 1]["start"] - speech["end"]
            if silence_duration < 2 * speech_pad_samples:
                speech["end"] += int(silence_duration // 2)
                speeches[i + 1]["start"] = int(max(0, speeches[i + 1]["start"] - silence_duration // 2))
            else:
                speech["end"] = int(min(audio_length_samples, speech["end"] + speech_pad_samples))
                speeches[i + 1]["start"] = int(max(0, speeches[i + 1]["start"] - speech_pad_samples))
        else:
            speech["end"] = int(min(audio_length_samples, speech["end"] + speech_pad_samples))

    return speeches


def _format_timestamp(seconds: float, *, srt: bool = False) -> str:
    seconds = max(float(seconds), 0.0)
    total_millis = int(round(seconds * 1000))
    hours, remainder = divmod(total_millis, 3600 * 1000)
    minutes, remainder = divmod(remainder, 60 * 1000)
    whole_seconds, millis = divmod(remainder, 1000)
    separator = "," if srt else "."
    return f"{hours:02}:{minutes:02}:{whole_seconds:02}{separator}{millis:03}"


class MimoASRLoadModel:
    CATEGORY = "audio/Comfy-MIMOASR"
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
                    {"default": True, "tooltip": "Print detailed load and inference progress to the console."},
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
    CATEGORY = "audio/Comfy-MIMOASR"
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
                        "default": 128,
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


class SileroVADAudioSegmenter:
    CATEGORY = "audio/Comfy-MIMOASR"
    DESCRIPTION = "Split ComfyUI AUDIO into speech segments with Silero VAD timestamps."
    RETURN_TYPES = ("MIMOASR_TIMED_AUDIO",)
    RETURN_NAMES = ("timed_audio",)
    FUNCTION = "segment"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "ComfyUI native audio input."}),
                "vad_model_path": (
                    "STRING",
                    {
                        "default": "auto",
                        "tooltip": "Path to silero_vad.jit. auto uses this extension's vad/silero_vad.jit.",
                    },
                ),
                "threshold": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.01,
                        "max": 0.99,
                        "step": 0.01,
                        "tooltip": "Speech probability threshold. Higher values are stricter.",
                    },
                ),
                "min_speech_duration_ms": (
                    "INT",
                    {
                        "default": 250,
                        "min": 0,
                        "max": 10000,
                        "step": 10,
                        "tooltip": "Drop detected speech chunks shorter than this duration.",
                    },
                ),
                "max_speech_duration_s": (
                    "FLOAT",
                    {
                        "default": 30.0,
                        "min": 0.0,
                        "max": 3600.0,
                        "step": 0.5,
                        "tooltip": "Split long speech chunks near silence. Use 0 for unlimited.",
                    },
                ),
                "min_silence_duration_ms": (
                    "INT",
                    {
                        "default": 100,
                        "min": 0,
                        "max": 10000,
                        "step": 10,
                        "tooltip": "Silence duration needed before closing a speech chunk.",
                    },
                ),
                "speech_pad_ms": (
                    "INT",
                    {
                        "default": 30,
                        "min": 0,
                        "max": 5000,
                        "step": 10,
                        "tooltip": "Padding added to both sides of each detected speech chunk.",
                    },
                ),
                "batch_index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "Batch index to segment from batched AUDIO.",
                    },
                ),
            }
        }

    def segment(
        self,
        audio,
        vad_model_path: str,
        threshold: float,
        min_speech_duration_ms: int,
        max_speech_duration_s: float,
        min_silence_duration_ms: int,
        speech_pad_ms: int,
        batch_index: int,
    ):
        waveform, sample_rate = _select_comfy_waveform(audio, batch_index=int(batch_index))
        vad_waveform = waveform.mean(dim=0)
        vad_sample_rate = 16000
        if sample_rate != vad_sample_rate:
            vad_waveform = torchaudio.functional.resample(
                vad_waveform,
                orig_freq=sample_rate,
                new_freq=vad_sample_rate,
            )
        vad_waveform = vad_waveform.detach().float().cpu().contiguous()

        vad_model = _load_silero_vad_model(vad_model_path)
        timestamps = _get_speech_timestamps(
            vad_waveform,
            vad_model,
            threshold=float(threshold),
            sampling_rate=vad_sample_rate,
            min_speech_duration_ms=int(min_speech_duration_ms),
            max_speech_duration_s=float(max_speech_duration_s),
            min_silence_duration_ms=int(min_silence_duration_ms),
            speech_pad_ms=int(speech_pad_ms),
        )

        total_samples = int(waveform.shape[-1])
        duration = total_samples / sample_rate if sample_rate else 0.0
        segments = []
        for index, speech in enumerate(timestamps, start=1):
            start_seconds = max(float(speech["start"]) / vad_sample_rate, 0.0)
            end_seconds = min(float(speech["end"]) / vad_sample_rate, duration)
            start_sample = max(0, min(total_samples, int(round(start_seconds * sample_rate))))
            end_sample = max(start_sample, min(total_samples, int(round(end_seconds * sample_rate))))
            if end_sample <= start_sample:
                continue
            segment_waveform = waveform[:, start_sample:end_sample].contiguous()
            segment_audio = {
                "waveform": segment_waveform.unsqueeze(0),
                "sample_rate": sample_rate,
            }
            segments.append(
                {
                    "index": index,
                    "start": start_seconds,
                    "end": end_seconds,
                    "duration": end_seconds - start_seconds,
                    "start_sample": start_sample,
                    "end_sample": end_sample,
                    "sample_rate": sample_rate,
                    "audio": segment_audio,
                }
            )

        timed_audio = {
            "type": "MIMOASR_TIMED_AUDIO",
            "sample_rate": sample_rate,
            "duration": duration,
            "batch_index": int(batch_index),
            "vad_sample_rate": vad_sample_rate,
            "vad_model_path": str(_resolve_vad_model_path(vad_model_path)),
            "segments": segments,
        }
        return (timed_audio,)


class MimoASRTimedAudioToText:
    CATEGORY = "audio/Comfy-MIMOASR"
    DESCRIPTION = "Transcribe a timed audio segment group with MiMo ASR and keep timestamps."
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("timestamped_text",)
    FUNCTION = "transcribe_timed"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mimo_asr_model": (
                    "MIMOASR_MODEL",
                    {"tooltip": "Loaded MiMo ASR model from the loader node."},
                ),
                "timed_audio": (
                    "MIMOASR_TIMED_AUDIO",
                    {"tooltip": "Timed audio group from the Silero VAD segmenter node."},
                ),
                "language": (
                    ["auto", "zh", "en"],
                    {"default": "auto", "tooltip": "Language hint for recognition."},
                ),
                "max_new_tokens": (
                    "INT",
                    {
                        "default": 128,
                        "min": 1,
                        "max": 65536,
                        "step": 1,
                        "tooltip": "Maximum generated text token groups for each segment.",
                    },
                ),
                "timestamp_format": (
                    ["bracket", "srt", "jsonl"],
                    {
                        "default": "bracket",
                        "tooltip": "Output format for segment timestamps.",
                    },
                ),
                "skip_empty_segments": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Skip segments that return empty text."},
                ),
            }
        }

    def transcribe_timed(
        self,
        mimo_asr_model: MimoASRModelHandle,
        timed_audio: dict[str, Any],
        language: str,
        max_new_tokens: int,
        timestamp_format: str,
        skip_empty_segments: bool,
    ):
        if not isinstance(timed_audio, dict) or "segments" not in timed_audio:
            raise TypeError("Expected MIMOASR_TIMED_AUDIO input from the Silero VAD segmenter node.")

        rendered: list[str] = []
        srt_index = 1
        for output_index, segment in enumerate(timed_audio.get("segments") or [], start=1):
            segment_audio = segment.get("audio")
            if not isinstance(segment_audio, dict):
                continue
            waveform = _audio_from_comfy(
                segment_audio,
                target_sample_rate=mimo_asr_model.target_sample_rate,
                batch_index=0,
            )
            text = mimo_asr_model.transcribe(
                waveform,
                language=language,
                max_new_tokens=int(max_new_tokens),
                audio_start_seconds=0.0,
                audio_duration_seconds=0.0,
            ).strip()
            if skip_empty_segments and not text:
                continue

            start = float(segment.get("start", 0.0))
            end = float(segment.get("end", start))
            if timestamp_format == "srt":
                rendered.extend(
                    [
                        str(srt_index),
                        f"{_format_timestamp(start, srt=True)} --> {_format_timestamp(end, srt=True)}",
                        text,
                        "",
                    ]
                )
                srt_index += 1
            elif timestamp_format == "jsonl":
                rendered.append(
                    json.dumps(
                        {
                            "index": output_index,
                            "start": start,
                            "end": end,
                            "text": text,
                        },
                        ensure_ascii=False,
                    )
                )
            else:
                rendered.append(f"[{_format_timestamp(start)} --> {_format_timestamp(end)}] {text}")

        return ("\n".join(rendered).strip(),)


class MimoASRUnloadModel:
    CATEGORY = "audio/Comfy-MIMOASR"
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


class MimoASRSaveText:
    CATEGORY = "audio/Comfy-MIMOASR"
    DESCRIPTION = "Save input text as a numbered txt file."
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_path",)
    FUNCTION = "save_text"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True, "tooltip": "Text to save."}),
                "filename_prefix": (
                    "STRING",
                    {
                        "default": "Comfy-MIMOASR/transcript",
                        "tooltip": "File prefix. Subfolders are supported, for example transcripts/session.",
                    },
                ),
                "output_directory": (
                    "STRING",
                    {
                        "default": "auto",
                        "tooltip": "Save directory. auto uses ComfyUI/output; absolute and relative custom paths are supported.",
                    },
                ),
            }
        }

    def save_text(self, text: str, filename_prefix: str = "Comfy-MIMOASR/transcript", output_directory: str = "auto"):
        save_root = (output_directory or "auto").strip()
        if not save_root or save_root.lower() == "auto":
            output_dir = _comfy_output_dir()
        else:
            output_dir = Path(save_root).expanduser()
            if not output_dir.is_absolute():
                output_dir = (Path.cwd() / output_dir).resolve()

        filename_prefix = (filename_prefix or "Comfy-MIMOASR/transcript").strip() or "Comfy-MIMOASR/transcript"
        if folder_paths is not None and hasattr(folder_paths, "get_save_image_path"):
            full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
                filename_prefix,
                str(output_dir),
            )
            output_folder = Path(full_output_folder)
        else:
            normalized_prefix = Path(filename_prefix)
            subfolder = str(normalized_prefix.parent) if str(normalized_prefix.parent) != "." else ""
            filename = normalized_prefix.name
            output_folder = output_dir / subfolder
            output_folder.mkdir(parents=True, exist_ok=True)
            existing = []
            for path in output_folder.glob(f"{filename}_*.txt"):
                suffix = path.stem.removeprefix(f"{filename}_").split("_", 1)[0]
                if suffix.isdigit():
                    existing.append(int(suffix))
            counter = max(existing, default=0) + 1

        output_folder.mkdir(parents=True, exist_ok=True)
        file = f"{filename}_{counter:05}_.txt"
        saved_path = output_folder / file
        saved_path.write_text(str(text), encoding="utf-8")

        return {
            "ui": {
                "text": [str(saved_path)],
                "files": [{"filename": file, "subfolder": subfolder, "type": "output"}],
            },
            "result": (str(saved_path),),
        }


NODE_CLASS_MAPPINGS = {
    "MimoASRLoadModel": MimoASRLoadModel,
    "MimoASRAudioToText": MimoASRAudioToText,
    "SileroVADAudioSegmenter": SileroVADAudioSegmenter,
    "MimoASRTimedAudioToText": MimoASRTimedAudioToText,
    "MimoASRUnloadModel": MimoASRUnloadModel,
    "MimoASRSaveText": MimoASRSaveText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MimoASRLoadModel": "MiMo ASR Load Model",
    "MimoASRAudioToText": "MiMo ASR Audio To Text",
    "SileroVADAudioSegmenter": "SILERO-VAD Audio Segmenter",
    "MimoASRTimedAudioToText": "MiMo ASR Timed Audio To Text",
    "MimoASRUnloadModel": "MiMo ASR Release Model",
    "MimoASRSaveText": "MiMo ASR Save Text",
}
