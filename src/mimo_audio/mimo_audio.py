# Copyright 2025 Xiaomi Corporation.
import gc
import hashlib
import json
import os
import subprocess
import time
import random
from collections import defaultdict
from importlib.metadata import PackageNotFoundError, version
import torch
import torch.nn.functional as F
import torchaudio

from typing import Union
from torchaudio.transforms import MelSpectrogram
from transformers import (
    AutoTokenizer,
    GenerationConfig
)
try:
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
except ImportError:
    from transformers import PreTrainedTokenizerFast

from .process_speechdata import InputSegment
from ..mimo_audio_tokenizer import MiMoAudioTokenizer
from .templates import asr_en_templates, asr_zh_templates
from .modeling_mimo_audio import (
    MiMoAudioArguments,
    MiMoAudioConfig,
    MiMoAudioForCausalLM,
    MiMoSampler,
    MiMoStopper,
)

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    _tqdm = None


def _transformers_major_version() -> int:
    try:
        return int(version("transformers").split(".", 1)[0])
    except (PackageNotFoundError, ValueError):
        return 0


def _cuda_memory_text(device: torch.device | str) -> str:
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return ""
    allocated = torch.cuda.memory_allocated(device) / 1024**2
    reserved = torch.cuda.memory_reserved(device) / 1024**2
    peak = torch.cuda.max_memory_allocated(device) / 1024**2
    return f"cuda alloc/res/peak {allocated:.0f}/{reserved:.0f}/{peak:.0f} MiB"


def _normalize_quantization(quantization: str | None) -> str | None:
    if quantization is None:
        return None

    aliases = {
        "": None,
        "none": None,
        "no": None,
        "off": None,
        "false": None,
        "8bit": "int8",
        "bnb-int8": "int8",
        "int8": "int8",
        "4bit": "int4",
        "bnb-int4": "int4",
        "int4": "int4",
        "nf4": "nf4",
        "fp4": "fp4",
        "fbgemm-fp8": "fp8",
        "fp8": "fp8",
    }
    key = quantization.strip().lower()
    if key not in aliases:
        valid = "none, int8, int4, nf4, fp4, fp8"
        raise ValueError(
            f"Unsupported quantization mode: {quantization!r}. Choose one of: {valid}."
        )
    return aliases[key]


class _WeightOnlyQuantizedLinear(torch.nn.Module):
    def __init__(
        self,
        linear: torch.nn.Linear,
        mode: str,
        *,
        codebook: torch.Tensor | None = None,
        chunk_size: int | None = 1024,
    ) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.mode = mode
        self.chunk_size = chunk_size

        weight = linear.weight.detach().float().cpu()
        if linear.bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", linear.bias.detach().clone())

        if mode == "int8":
            scale = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127
            qweight = torch.round(weight / scale).clamp(-128, 127).to(torch.int8)
            self.register_buffer("qweight", qweight)
            self.register_buffer("scale", scale)
            return

        if mode == "fp8":
            if not hasattr(torch, "float8_e4m3fn"):
                raise RuntimeError("fp8 quantization requires PyTorch float8 support.")
            self.register_buffer("qweight", weight.to(torch.float8_e4m3fn))
            self.register_buffer("scale", None)
            return

        if codebook is None:
            raise ValueError(f"Missing codebook for {mode} quantization.")

        scale = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        normalized = (weight / scale).clamp(-1, 1)
        distances = (normalized.unsqueeze(-1) - codebook.view(1, 1, -1)).abs()
        qweight = distances.argmin(dim=-1).to(torch.uint8)
        if qweight.shape[1] % 2:
            qweight = F.pad(qweight, (0, 1))
        packed = qweight[:, 0::2] | (qweight[:, 1::2] << 4)

        self.register_buffer("qweight", packed.contiguous())
        self.register_buffer("scale", scale)
        self.register_buffer("codebook", codebook.float())

    @classmethod
    def from_quantized_tensors(
        cls,
        *,
        in_features: int,
        out_features: int,
        mode: str,
        qweight: torch.Tensor,
        scale: torch.Tensor | None,
        codebook: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        chunk_size: int | None = 1024,
    ) -> "_WeightOnlyQuantizedLinear":
        module = cls.__new__(cls)
        torch.nn.Module.__init__(module)
        module.in_features = in_features
        module.out_features = out_features
        module.mode = mode
        module.chunk_size = chunk_size
        module.register_buffer("bias", bias)
        module.register_buffer("qweight", qweight)
        module.register_buffer("scale", scale)
        if codebook is not None:
            module.register_buffer("codebook", codebook)
        return module

    def _dequantize_weight(
        self,
        dtype: torch.dtype,
        device: torch.device,
        row_start: int = 0,
        row_end: int | None = None,
    ) -> torch.Tensor:
        row_end = self.out_features if row_end is None else row_end
        if self.mode == "int8":
            return self.qweight[row_start:row_end].to(device=device, dtype=dtype) * self.scale[row_start:row_end].to(
                device=device,
                dtype=dtype,
            )

        if self.mode == "fp8":
            return self.qweight[row_start:row_end].to(device=device, dtype=dtype)

        packed = self.qweight[row_start:row_end].to(device)
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        indices = torch.stack((low, high), dim=-1).flatten(1)[:, : self.in_features]
        values = self.codebook.to(device=device, dtype=dtype)
        return values[indices.long()] * self.scale[row_start:row_end].to(device=device, dtype=dtype)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.chunk_size and self.chunk_size > 0 and self.out_features > 16384:
            chunk_rows = self.chunk_size
            flat_inputs = inputs.reshape(-1, self.in_features)
            output_chunks = []
            for row_start in range(0, self.out_features, chunk_rows):
                row_end = min(row_start + chunk_rows, self.out_features)
                weight = self._dequantize_weight(
                    inputs.dtype,
                    inputs.device,
                    row_start,
                    row_end,
                )
                bias = (
                    None
                    if self.bias is None
                    else self.bias[row_start:row_end].to(
                        device=inputs.device,
                        dtype=inputs.dtype,
                    )
                )
                output_chunks.append(F.linear(flat_inputs, weight, bias))
            return torch.cat(output_chunks, dim=-1).reshape(*inputs.shape[:-1], self.out_features)

        weight = self._dequantize_weight(inputs.dtype, inputs.device)
        bias = None if self.bias is None else self.bias.to(device=inputs.device, dtype=inputs.dtype)
        return F.linear(inputs, weight, bias)


def _four_bit_codebook(mode: str) -> torch.Tensor:
    if mode in {"int4", "nf4"}:
        return torch.tensor(
            [
                -1.0,
                -0.6961928,
                -0.52507305,
                -0.3949175,
                -0.28444138,
                -0.18477343,
                -0.09105004,
                0.0,
                0.0795803,
                0.1609302,
                0.2461123,
                0.33791524,
                0.44070983,
                0.562617,
                0.72295684,
                1.0,
            ],
            dtype=torch.float32,
        )
    if mode == "fp4":
        return torch.linspace(-1.0, 1.0, steps=16, dtype=torch.float32)
    raise ValueError(f"Unsupported 4-bit quantization mode: {mode}")


def _replace_linear_modules(
    module: torch.nn.Module,
    mode: str,
    *,
    codebook: torch.Tensor | None = None,
    chunk_size: int | None = 1024,
) -> int:
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.Linear):
            setattr(
                module,
                name,
                _WeightOnlyQuantizedLinear(
                    child,
                    mode,
                    codebook=codebook,
                    chunk_size=chunk_size,
                ),
            )
            replaced += 1
        else:
            replaced += _replace_linear_modules(
                child,
                mode,
                codebook=codebook,
                chunk_size=chunk_size,
            )
    return replaced


def _apply_weight_only_quantization(
    quantization: str | None,
    *,
    model: torch.nn.Module,
    bnb_4bit_compute_dtype: str | None = "auto",
    bnb_4bit_use_double_quant: bool = False,
    quantized_linear_chunk_size: int | None = 1024,
) -> int:
    mode = _normalize_quantization(quantization)
    if mode is None:
        return 0

    if bnb_4bit_compute_dtype not in {None, "auto"}:
        print(
            "--bnb-4bit-compute-dtype is ignored by the built-in weight-only "
            "quantizer; computation follows the model input dtype."
        )
    if bnb_4bit_use_double_quant:
        print(
            "--bnb-4bit-double-quant is ignored by the built-in weight-only "
            "quantizer."
        )

    codebook = _four_bit_codebook(mode) if mode in {"int4", "nf4", "fp4"} else None
    return _replace_linear_modules(
        model,
        mode,
        codebook=codebook,
        chunk_size=quantized_linear_chunk_size,
    )


def _maybe_warn_quantization_compat_options(
    bnb_4bit_compute_dtype: str | None,
    bnb_4bit_use_double_quant: bool,
) -> None:
    if bnb_4bit_compute_dtype not in {None, "auto"}:
        print(
            "--bnb-4bit-compute-dtype is ignored by the built-in weight-only "
            "quantizer; computation follows the model input dtype."
        )
    if bnb_4bit_use_double_quant:
        print(
            "--bnb-4bit-double-quant is ignored by the built-in weight-only "
            "quantizer."
        )


def _quantization_chunk_rows(weight: torch.Tensor, mode: str) -> int:
    if weight.ndim != 2 or weight.shape[1] == 0:
        return 1
    max_temp_bytes = int(os.environ.get("MIMO_QUANTIZE_MAX_TEMP_MB", "256")) * 1024 * 1024
    multiplier = 16 if mode in {"int4", "nf4", "fp4"} else 2
    bytes_per_row = max(1, weight.shape[1] * multiplier * 4)
    return max(1, min(weight.shape[0], max_temp_bytes // bytes_per_row))


def _quantize_weight_tensor(
    weight: torch.Tensor,
    mode: str,
    *,
    codebook: torch.Tensor | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if weight.ndim != 2:
        raise ValueError(f"Only 2D linear weights can be quantized, got shape {tuple(weight.shape)}")

    weight = weight.detach().cpu()
    out_features, in_features = weight.shape
    chunk_rows = _quantization_chunk_rows(weight, mode)

    if mode == "int8":
        qweight = torch.empty((out_features, in_features), dtype=torch.int8)
        scale = torch.empty((out_features, 1), dtype=torch.float32)
        for start in range(0, out_features, chunk_rows):
            end = min(start + chunk_rows, out_features)
            chunk = weight[start:end].float()
            chunk_scale = chunk.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127
            qweight[start:end] = torch.round(chunk / chunk_scale).clamp(-128, 127).to(torch.int8)
            scale[start:end] = chunk_scale
        return qweight.to(device), scale.to(device), None

    if mode == "fp8":
        if not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError("fp8 quantization requires PyTorch float8 support.")
        return weight.to(device=device, dtype=torch.float8_e4m3fn), None, None

    if codebook is None:
        raise ValueError(f"Missing codebook for {mode} quantization.")

    cpu_codebook = codebook.float().cpu()
    packed_width = (in_features + 1) // 2
    packed = torch.empty((out_features, packed_width), dtype=torch.uint8)
    scale = torch.empty((out_features, 1), dtype=torch.float32)

    for start in range(0, out_features, chunk_rows):
        end = min(start + chunk_rows, out_features)
        chunk = weight[start:end].float()
        chunk_scale = chunk.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        normalized = (chunk / chunk_scale).clamp(-1, 1)
        distances = (normalized.unsqueeze(-1) - cpu_codebook.view(1, 1, -1)).abs()
        qweight = distances.argmin(dim=-1).to(torch.uint8)
        if in_features % 2:
            qweight = F.pad(qweight, (0, 1))
        packed[start:end] = qweight[:, 0::2] | (qweight[:, 1::2] << 4)
        scale[start:end] = chunk_scale

    return packed.to(device), scale.to(device), cpu_codebook.to(device)


def _split_tensor_name(name: str) -> tuple[str, str]:
    if "." not in name:
        return "", name
    return name.rsplit(".", 1)


def _set_module_tensor(
    model: torch.nn.Module,
    tensor_name: str,
    tensor: torch.Tensor,
) -> None:
    module_name, attr_name = _split_tensor_name(tensor_name)
    module = model.get_submodule(module_name) if module_name else model
    if attr_name in module._parameters:
        old_param = module._parameters[attr_name]
        requires_grad = bool(old_param.requires_grad) if old_param is not None else False
        module._parameters[attr_name] = torch.nn.Parameter(tensor, requires_grad=requires_grad)
        return
    if attr_name in module._buffers:
        module._buffers[attr_name] = tensor
        return
    raise KeyError(f"Cannot place tensor {tensor_name!r}: target attribute was not found.")


def _replace_module(
    model: torch.nn.Module,
    module_name: str,
    module: torch.nn.Module,
) -> None:
    parent_name, child_name = _split_tensor_name(module_name)
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child_name, module)


def _load_safetensors_index(model_path: str) -> tuple[dict[str, str], list[str]]:
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as index_file:
            weight_map = json.load(index_file)["weight_map"]
        return weight_map, sorted(set(weight_map.values()))

    single_file = os.path.join(model_path, "model.safetensors")
    if os.path.exists(single_file):
        from safetensors.torch import safe_open

        with safe_open(single_file, framework="pt", device="cpu") as shard:
            keys = list(shard.keys())
        return {key: "model.safetensors" for key in keys}, ["model.safetensors"]

    raise FileNotFoundError(
        f"No safetensors checkpoint found in {model_path!r}. "
        "Streaming quantized loading requires model.safetensors or model.safetensors.index.json."
    )


def _checkpoint_signature(model_path: str) -> dict:
    weight_map, shard_names = _load_safetensors_index(model_path)
    files = ["config.json", "generation_config.json", "model.safetensors.index.json"]
    files.extend(shard_names)

    file_signatures = []
    for filename in sorted(set(files)):
        path = os.path.join(model_path, filename)
        if not os.path.exists(path):
            continue
        stat = os.stat(path)
        file_signatures.append(
            {
                "name": filename,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )

    return {
        "model_path": os.path.abspath(model_path),
        "weight_count": len(weight_map),
        "files": file_signatures,
    }


def _signature_digest(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _normalise_quantization_cache_dir(cache_dir: str | os.PathLike | None) -> str | None:
    if cache_dir is None:
        return None
    cache_dir = os.fspath(cache_dir).strip()
    if not cache_dir:
        return None
    if cache_dir.lower() == "auto":
        return os.path.join("temp", "quantized-cache")
    return cache_dir


def _quantization_cache_path(
    cache_dir: str,
    model_path: str,
    *,
    mode: str,
    dtype: torch.dtype,
) -> tuple[str, dict]:
    signature = _checkpoint_signature(model_path)
    key_payload = {
        "cache_version": 1,
        "signature": signature,
        "mode": mode,
        "dtype": _dtype_name(dtype),
        "format": "mimo-weight-only-state-dict",
    }
    filename = f"{os.path.basename(os.path.abspath(model_path))}-{mode}-{_dtype_name(dtype)}-{_signature_digest(key_payload)}.pt"
    return os.path.join(cache_dir, filename), key_payload


def _empty_quantized_linear_from_linear(
    linear: torch.nn.Linear,
    mode: str,
    *,
    chunk_size: int | None,
) -> _WeightOnlyQuantizedLinear:
    out_features = linear.out_features
    in_features = linear.in_features
    if mode in {"int4", "nf4", "fp4"}:
        qweight = torch.empty(
            (out_features, (in_features + 1) // 2),
            dtype=torch.uint8,
            device="meta",
        )
        scale = torch.empty((out_features, 1), dtype=torch.float32, device="meta")
        codebook = torch.empty((16,), dtype=torch.float32, device="meta")
    elif mode == "int8":
        qweight = torch.empty(
            (out_features, in_features),
            dtype=torch.int8,
            device="meta",
        )
        scale = torch.empty((out_features, 1), dtype=torch.float32, device="meta")
        codebook = None
    elif mode == "fp8":
        fp8_dtype = getattr(torch, "float8_e4m3fn", torch.uint8)
        qweight = torch.empty(
            (out_features, in_features),
            dtype=fp8_dtype,
            device="meta",
        )
        scale = None
        codebook = None
    else:
        raise ValueError(f"Unsupported quantization cache mode: {mode}")

    bias = None
    if linear.bias is not None:
        bias = torch.empty((out_features,), dtype=linear.bias.dtype, device="meta")

    return _WeightOnlyQuantizedLinear.from_quantized_tensors(
        in_features=in_features,
        out_features=out_features,
        mode=mode,
        qweight=qweight,
        scale=scale,
        codebook=codebook,
        bias=bias,
        chunk_size=chunk_size,
    )


def _replace_quantized_modules_from_names(
    model: torch.nn.Module,
    module_names: list[str],
    *,
    mode: str,
    chunk_size: int | None,
) -> None:
    for module_name in module_names:
        module = model.get_submodule(module_name)
        if not isinstance(module, torch.nn.Linear):
            raise RuntimeError(
                f"Quantization cache expected Linear module {module_name!r}, "
                f"got {type(module).__name__}."
            )
        _replace_module(
            model,
            module_name,
            _empty_quantized_linear_from_linear(
                module,
                mode,
                chunk_size=chunk_size,
            ),
        )


def _torch_load_weights(path: str, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _local_config(path: str) -> dict | None:
    config_path = os.path.join(os.fspath(path), "config.json")
    if not os.path.isfile(config_path):
        return None

    try:
        with open(config_path, "r", encoding="utf-8") as config_file:
            return json.load(config_file)
    except (OSError, json.JSONDecodeError):
        return None


def _local_config_model_type(path: str) -> str | None:
    config = _local_config(path)
    if config is None:
        return None

    model_type = config.get("model_type")
    return str(model_type) if model_type is not None else None


def _is_mimo_audio_tokenizer_config(config: dict | None) -> bool:
    if not config:
        return False

    if config.get("model_type") == "mimo_audio_tokenizer":
        return True

    required_keys = {
        "encoder_layers",
        "decoder_layers",
        "nfft",
        "n_mels",
        "sampling_rate",
        "hop_length",
        "window_size",
        "num_quantizers",
        "codebook_size",
    }
    return required_keys.issubset(config)


def _same_local_path(left: str, right: str) -> bool:
    try:
        return os.path.normcase(os.path.abspath(left)) == os.path.normcase(os.path.abspath(right))
    except (OSError, TypeError, ValueError):
        return False


def _is_explicit_local_path(path: str) -> bool:
    path = os.fspath(path)
    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded) or os.path.splitdrive(expanded)[0]:
        return True

    first_part = expanded.replace("\\", "/").split("/", 1)[0]
    if first_part and os.path.isdir(first_part):
        return True

    local_prefixes = (".", f".{os.sep}", f"..{os.sep}", "./", "../", "~")
    if os.altsep:
        local_prefixes += (f".{os.altsep}", f"..{os.altsep}")
    return expanded.startswith(local_prefixes)


def _nearby_existing_path_hint(path: str) -> str:
    candidates = []
    normalized = os.path.normpath(path)

    parts = normalized.split(os.sep)
    for index, part in enumerate(parts):
        if part == "models":
            candidate_parts = parts.copy()
            candidate_parts[index] = "model"
            candidates.append(os.sep.join(candidate_parts))
        elif part == "model":
            candidate_parts = parts.copy()
            candidate_parts[index] = "models"
            candidates.append(os.sep.join(candidate_parts))

    basename = os.path.basename(normalized)
    candidates.extend([
        os.path.join("model", basename),
        os.path.join("models", basename),
    ])

    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            return f" Did you mean {candidate!r}?"
    return ""


def _validate_mimo_audio_tokenizer_path(path: str | None, model_path: str) -> None:
    if not path:
        raise ValueError(
            "MiMo audio tokenizer path is required. Download it separately with "
            "`hf download XiaomiMiMo/MiMo-Audio-Tokenizer --local-dir ./model/MiMo-Audio-Tokenizer` "
            "and pass `--audio-tokenizer ./model/MiMo-Audio-Tokenizer`."
        )

    if not os.path.exists(path) and _is_explicit_local_path(path):
        raise FileNotFoundError(
            f"MiMo audio tokenizer path {path!r} does not exist."
            f"{_nearby_existing_path_hint(path)} "
            "Pass the directory downloaded from XiaomiMiMo/MiMo-Audio-Tokenizer."
        )

    config = _local_config(path)
    model_type = str(config["model_type"]) if config and config.get("model_type") is not None else None
    if _is_mimo_audio_tokenizer_config(config):
        return

    if os.path.isdir(path) and config is None:
        raise ValueError(
            f"MiMo audio tokenizer path {path!r} does not contain a readable config.json. "
            "Pass the directory downloaded from XiaomiMiMo/MiMo-Audio-Tokenizer."
        )

    if model_type is not None:
        hint = (
            " This is the ASR model directory, not the MiMo audio tokenizer directory."
            if _same_local_path(path, model_path)
            else ""
        )
        raise ValueError(
            f"MiMo audio tokenizer path {path!r} has model_type={model_type!r}, "
            "expected 'mimo_audio_tokenizer'."
            f"{hint} Download XiaomiMiMo/MiMo-Audio-Tokenizer separately and pass it with --audio-tokenizer."
        )

    if os.path.isdir(path):
        raise ValueError(
            f"MiMo audio tokenizer path {path!r} contains config.json, but it does not look like "
            "a MiMo audio tokenizer config. Pass the directory downloaded from "
            "XiaomiMiMo/MiMo-Audio-Tokenizer."
        )


def _load_audio_with_ffmpeg(path: str, sample_rate: int) -> tuple[torch.Tensor, int]:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        path,
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "pipe:1",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "torchaudio could not decode this audio file and ffmpeg was not found. "
            "Install ffmpeg or a TorchCodec version compatible with your PyTorch build."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace").strip()
        detail = f": {stderr}" if stderr else "."
        raise RuntimeError(f"ffmpeg failed to decode audio file {path!r}{detail}") from exc

    if not completed.stdout:
        raise RuntimeError(f"ffmpeg decoded no audio samples from {path!r}.")

    wav = torch.frombuffer(bytearray(completed.stdout), dtype=torch.float32).clone()
    return wav, sample_rate


def _materialize_meta_rotary_buffers(model: torch.nn.Module, device: torch.device) -> None:
    try:
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS as HF_ROPE_INIT_FUNCTIONS
    except ImportError:
        HF_ROPE_INIT_FUNCTIONS = {}

    for module in model.modules():
        inv_freq = getattr(module, "inv_freq", None)
        if not isinstance(inv_freq, torch.Tensor) or not inv_freq.is_meta:
            continue
        if not hasattr(module, "config") or not hasattr(module, "rope_type"):
            continue

        if module.rope_type == "default" and hasattr(module, "compute_default_rope_parameters"):
            rope_fn = module.compute_default_rope_parameters
        else:
            rope_fn = HF_ROPE_INIT_FUNCTIONS.get(module.rope_type)
        if rope_fn is None:
            continue

        new_inv_freq, attention_scaling = rope_fn(module.config, device=device)
        module.attention_scaling = attention_scaling
        module.register_buffer("inv_freq", new_inv_freq, persistent=False)
        if hasattr(module, "original_inv_freq"):
            module.register_buffer("original_inv_freq", new_inv_freq.clone(), persistent=False)


def _raise_if_model_has_meta_tensors(model: torch.nn.Module) -> None:
    meta_names = []
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if tensor.is_meta:
            meta_names.append(name)
            if len(meta_names) >= 5:
                break
    if meta_names:
        raise RuntimeError(
            "Streaming checkpoint load left tensors on the meta device. "
            f"First meta tensors: {', '.join(meta_names)}"
        )


def _normalize_device_arg(value: str | torch.device | None) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    if not value or value.lower() == "auto":
        return None
    return value


def _load_weight_only_quantized_model_streaming(
    model_path: str,
    model_args: MiMoAudioArguments,
    *,
    quantization: str,
    device: str,
    dtype: torch.dtype,
    quantized_linear_chunk_size: int | None = 1024,
    quantization_cache_dir: str | os.PathLike | None = None,
    rebuild_quantization_cache: bool = False,
    bnb_4bit_compute_dtype: str | None = "auto",
    bnb_4bit_use_double_quant: bool = False,
) -> tuple[MiMoAudioForCausalLM, int]:
    from safetensors.torch import safe_open

    _maybe_warn_quantization_compat_options(
        bnb_4bit_compute_dtype,
        bnb_4bit_use_double_quant,
    )

    mode = _normalize_quantization(quantization)
    if mode is None:
        raise ValueError("Streaming quantized loading requires an enabled quantization mode.")

    target_device = torch.device(device)
    codebook = _four_bit_codebook(mode) if mode in {"int4", "nf4", "fp4"} else None
    config = MiMoAudioConfig.from_pretrained(model_path)

    cache_dir = _normalise_quantization_cache_dir(quantization_cache_dir)
    cache_path = None
    expected_cache_metadata = None
    if cache_dir is not None:
        cache_path, expected_cache_metadata = _quantization_cache_path(
            cache_dir,
            model_path,
            mode=mode,
            dtype=dtype,
        )
        if os.path.exists(cache_path) and not rebuild_quantization_cache:
            try:
                cache_start = time.monotonic()
                print(f"Loading quantized model cache: {cache_path}", flush=True)
                payload = _torch_load_weights(cache_path, map_location=target_device)
                metadata = payload.get("metadata")
                if metadata != expected_cache_metadata:
                    raise RuntimeError("cache metadata does not match the current model files")

                quantized_module_names = payload["quantized_module_names"]
                with torch.device("meta"):
                    model = MiMoAudioForCausalLM(config, model_args)
                _replace_quantized_modules_from_names(
                    model,
                    quantized_module_names,
                    mode=mode,
                    chunk_size=quantized_linear_chunk_size,
                )
                model.load_state_dict(payload["state_dict"], strict=True, assign=True)
                _materialize_meta_rotary_buffers(model, target_device)
                _raise_if_model_has_meta_tensors(model)
                print(
                    "Quantized model cache loaded in "
                    f"{time.monotonic() - cache_start:.2f} seconds.",
                    flush=True,
                )
                return model, len(quantized_module_names)
            except Exception as exc:
                print(
                    f"Could not use quantized model cache ({exc}); rebuilding from safetensors.",
                    flush=True,
                )

    with torch.device("meta"):
        model = MiMoAudioForCausalLM(config, model_args)

    weight_map, shard_names = _load_safetensors_index(model_path)
    keys_by_shard: dict[str, list[str]] = defaultdict(list)
    for key, shard_name in weight_map.items():
        keys_by_shard[shard_name].append(key)

    expected_keys = set(model.state_dict().keys())
    loaded_keys: set[str] = set()
    quantized_layers = 0
    quantized_module_names: list[str] = []

    for shard_name in shard_names:
        shard_path = os.path.join(model_path, shard_name)
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            for key in keys_by_shard[shard_name]:
                if key not in expected_keys:
                    continue

                tensor = shard.get_tensor(key)
                module_name, attr_name = _split_tensor_name(key)
                module = model.get_submodule(module_name) if module_name else model

                if attr_name == "weight" and isinstance(module, torch.nn.Linear):
                    qweight, scale, q_codebook = _quantize_weight_tensor(
                        tensor,
                        mode,
                        codebook=codebook,
                        device=target_device,
                    )
                    quantized = _WeightOnlyQuantizedLinear.from_quantized_tensors(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        mode=mode,
                        qweight=qweight,
                        scale=scale,
                        codebook=q_codebook,
                        bias=None,
                        chunk_size=quantized_linear_chunk_size,
                    )
                    _replace_module(model, module_name, quantized)
                    quantized_layers += 1
                    quantized_module_names.append(module_name)
                else:
                    if tensor.is_floating_point():
                        tensor = tensor.to(device=target_device, dtype=dtype)
                    else:
                        tensor = tensor.to(device=target_device)
                    _set_module_tensor(model, key, tensor)

                loaded_keys.add(key)
                del tensor

        gc.collect()
        if target_device.type == "cuda":
            torch.cuda.empty_cache()

    missing_keys = expected_keys - loaded_keys
    if missing_keys:
        sample = ", ".join(sorted(missing_keys)[:5])
        raise RuntimeError(
            f"Streaming checkpoint load left {len(missing_keys)} tensors unloaded. "
            f"First missing keys: {sample}"
        )

    _materialize_meta_rotary_buffers(model, target_device)
    _raise_if_model_has_meta_tensors(model)

    if cache_path is not None and expected_cache_metadata is not None:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            cache_start = time.monotonic()
            tmp_path = f"{cache_path}.tmp"
            torch.save(
                {
                    "metadata": expected_cache_metadata,
                    "quantized_module_names": quantized_module_names,
                    "state_dict": model.state_dict(),
                },
                tmp_path,
            )
            os.replace(tmp_path, cache_path)
            print(
                f"Saved quantized model cache in {time.monotonic() - cache_start:.2f} seconds: "
                f"{cache_path}",
                flush=True,
            )
        except Exception as exc:
            print(f"Warning: failed to save quantized model cache: {exc}", flush=True)

    return model, quantized_layers


class MimoAudio:

    def __init__(
        self,
        model_path: str,
        mimo_audio_tokenizer_path: str | None = None,
        device: str | None = None,
        *,
        tokenizer_path: str | None = None,
        quantization: str | None = None,
        mel_device: str | torch.device | None = "cpu",
        audio_tokenizer_device: str | torch.device | None = "auto",
        quantized_linear_chunk_size: int | None = 1024,
        quantization_cache_dir: str | os.PathLike | None = None,
        rebuild_quantization_cache: bool = False,
        bnb_4bit_compute_dtype: str | None = "auto",
        bnb_4bit_use_double_quant: bool = False,
        progress: bool = False,
        progress_interval: float = 2.0,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.progress = bool(progress)
        self.progress_interval = max(float(progress_interval), 0.1)
        self._progress_start_time = time.monotonic()
        self.quantization = _normalize_quantization(quantization)
        if quantized_linear_chunk_size is not None and quantized_linear_chunk_size < 0:
            raise ValueError("--quantized-linear-chunk-size must be >= 0.")
        if quantized_linear_chunk_size == 0:
            quantized_linear_chunk_size = None
            if self.quantization and torch.device(self.device).type == "cuda":
                print(
                    "Warning: --quantized-linear-chunk-size 0 disables chunked "
                    "dequantization. On 16GB GPUs this can make generation very "
                    "slow or run near OOM; try the default 1024 first.",
                    flush=True,
                )

        self.path = model_path
        self.mimo_audio_tokenizer_path = mimo_audio_tokenizer_path or tokenizer_path
        _validate_mimo_audio_tokenizer_path(self.mimo_audio_tokenizer_path, self.path)

        self.tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
            self.path
        )
        self.padding_idx = int(self.tokenizer.pad_token_id)

        special_tokens = [
            "<|sosp|>",
            "<|eosp|>",
            "<|empty|>",
            "<|Human|>",
            "<|SpeechLM|>",
            "<|sostm|>",
            "<|eostm|>",
            "<|eot|>",
        ]
        for token in special_tokens:
            if token not in self.tokenizer.get_vocab():
                print(f"Add special tokens {token} to tokenizer.vocab")
                self.tokenizer.add_tokens([token], special_tokens=True)

        self.sosp_idx = self.tokenizer.convert_tokens_to_ids("<|sosp|>")
        self.eosp_idx = self.tokenizer.convert_tokens_to_ids("<|eosp|>")
        self.empty_token = self.tokenizer.convert_tokens_to_ids("<|empty|>")
        self.sostm_idx = self.tokenizer.convert_tokens_to_ids("<|sostm|>")
        self.eostm_idx = self.tokenizer.convert_tokens_to_ids("<|eostm|>")
        self.eot_idx = self.tokenizer.convert_tokens_to_ids("<|eot|>")
        self.im_start_idx = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end_idx = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

        model_args = MiMoAudioArguments(
            model_name_or_path=self.path,
            sosp_idx=self.sosp_idx,
            eosp_idx=self.eosp_idx,
            empty_idx=self.empty_token,
            sostm_idx=self.sostm_idx,
            eostm_idx=self.eostm_idx,
            eot_idx=self.eot_idx,
        )

        start_loading_time = time.monotonic()
        dtype_key = "dtype" if _transformers_major_version() >= 5 else "torch_dtype"
        model_dtype = torch.bfloat16 if self.device != "cpu" else torch.float32
        model_load_kwargs = {
            "args": model_args,
            dtype_key: model_dtype,
        }

        if self.quantization:
            self.model, quantized_layers = _load_weight_only_quantized_model_streaming(
                self.path,
                model_args,
                quantization=self.quantization,
                device=self.device,
                dtype=model_dtype,
                quantized_linear_chunk_size=quantized_linear_chunk_size,
                quantization_cache_dir=quantization_cache_dir,
                rebuild_quantization_cache=rebuild_quantization_cache,
                bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
                bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
            )
        else:
            self.model = MiMoAudioForCausalLM.from_pretrained(
                self.path,
                **model_load_kwargs,
            )
            quantized_layers = 0
            self.model.to(self.device)

        self.group_size=self.model.config.group_size
        self.audio_channels=self.model.config.audio_channels
        self.delay_pattern = self.model.config.delay_pattern
        self.vocab_size = self.model.config.vocab_size

        self.speech_zeroemb_idx = self.model.speech_empty_ids

        self.model.eval()
        quant_suffix = (
            f", quantization: {self.quantization} ({quantized_layers} linear layers)"
            if self.quantization
            else ""
        )
        print(
            f"Model loaded in {time.monotonic() - start_loading_time:.2f} seconds, "
            f"device: {self.device}{quant_suffix}"
        )

        self.generate_kwargs = {
            "max_length": 8192,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        self.default_global_sampler = MiMoSampler(
            do_sample=True, temperature=0.6, top_k=50, top_p=0.95
        )
        self.default_local_sampler = MiMoSampler(
            do_sample=True, temperature=0.9, top_k=50, top_p=0.95
        )

        self.task_sampler_configs = {
            "asr": {
                "global": MiMoSampler(do_sample=False, temperature=1.0, top_p=1.0),
                "local": MiMoSampler(do_sample=True, temperature=0.9, top_p=0.95)
            },
        }

        start_loading_mimo_audio_tokenizer_time = time.monotonic()
        self.mimo_audio_tokenizer = MiMoAudioTokenizer.from_pretrained(self.mimo_audio_tokenizer_path)

        self.audio_tokenizer_device = _normalize_device_arg(audio_tokenizer_device)
        if self.audio_tokenizer_device is None:
            self.audio_tokenizer_device = self.device
        if (
            _normalize_device_arg(audio_tokenizer_device) is None
            and torch.device(self.device).type == "cuda"
            and torch.version.hip
        ):
            self.audio_tokenizer_device = "cpu"
            print(
                "ROCm/HIP runtime detected; keeping MiMo-Audio Tokenizer on CPU "
                "to avoid MIOpen Conv1d failures."
            )

        self.mel_device = _normalize_device_arg(mel_device) or self.audio_tokenizer_device

        self.mimo_audio_tokenizer.eval().bfloat16().to(self.audio_tokenizer_device)
        print(
            "MiMo-Audio Tokenizer loaded in "
            f"{time.monotonic() - start_loading_mimo_audio_tokenizer_time:.2f} seconds, "
            f"device: {self.audio_tokenizer_device}"
        )

        # Initialize mel spectrogram transform for consistent processing
        self.mel_transform = MelSpectrogram(
            sample_rate=self.mimo_audio_tokenizer.config.sampling_rate,
            n_fft=self.mimo_audio_tokenizer.config.nfft,
            hop_length=self.mimo_audio_tokenizer.config.hop_length,
            win_length=self.mimo_audio_tokenizer.config.window_size,
            f_min=self.mimo_audio_tokenizer.config.fmin,
            f_max=self.mimo_audio_tokenizer.config.fmax,
            n_mels=self.mimo_audio_tokenizer.config.n_mels,
            power=1.0,
            center=True,
        ).to(self.mel_device)
        print(f"Mel spectrogram device: {self.mel_device}")

    def _progress(self, message: str) -> None:
        if not self.progress:
            return
        elapsed = time.monotonic() - self._progress_start_time
        print(f"[{elapsed:8.2f}s] {message}", flush=True)

    def _progress_iter(self, iterable, **kwargs):
        if not self.progress or _tqdm is None:
            return iterable
        return _tqdm(iterable, dynamic_ncols=True, leave=True, **kwargs)

    def _sync_progress_device(self, device: torch.device | str) -> None:
        if self.progress and torch.device(device).type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)

    def get_task_sampler(self, task_name):
        if task_name not in self.task_sampler_configs:
            return {
                "global": self.default_global_sampler,
                "local": self.default_local_sampler
            }
        return self.task_sampler_configs[task_name]

    def wav2mel(self, wav):
        spec = self.mel_transform(wav[None, :])
        return torch.log(torch.clip(spec, min=1e-7)).squeeze()

    def resample_audio_if_needed(self, wav_tensor: torch.Tensor, original_sr: int):
        target_sr = self.mimo_audio_tokenizer.config.sampling_rate
        if original_sr != target_sr:
            wav_tensor = torchaudio.functional.resample(
                wav_tensor, original_sr, target_sr
            )
        return wav_tensor

    def group_by_length(self, features: torch.Tensor, lengths: torch.Tensor, max_length: int):
        if features.size(0) != lengths.sum().item():
            raise ValueError(f"Feature size mismatch: {features.size(0)} vs {lengths.sum().item()}")

        split_points = []
        current_sum = 0

        for i, seq_len in enumerate(lengths):
            if current_sum + seq_len > max_length and current_sum > 0:
                split_points.append(i)
                current_sum = seq_len.item()
            else:
                current_sum += seq_len.item()

        # Convert split points to group sizes
        group_sizes = []
        prev = 0
        for point in split_points:
            group_sizes.append(point - prev)
            prev = point
        if prev < len(lengths):
            group_sizes.append(len(lengths) - prev)

        len_groups = torch.split(lengths, group_sizes)
        feature_sizes = [group.sum().item() for group in len_groups]
        feature_groups = torch.split(features, feature_sizes)

        return feature_groups, len_groups

    def encode_batch(self, input_features: torch.Tensor, input_lens: torch.Tensor, max_length: int = 256000):
        feature_groups, len_groups = self.group_by_length(input_features, input_lens, max_length)

        encoded_parts = []
        tokenizer_dtype = next(self.mimo_audio_tokenizer.parameters()).dtype
        groups = list(zip(feature_groups, len_groups))
        for group_idx, (features, lengths) in enumerate(groups, start=1):
            self._progress(
                "Audio tokenizer encode "
                f"group {group_idx}/{len(groups)}: frames={features.size(0)}, "
                f"device={self.audio_tokenizer_device}, dtype={tokenizer_dtype}"
            )
            group_start = time.monotonic()
            with torch.no_grad():
                codes, _ = self.mimo_audio_tokenizer.encoder.encode(
                    input_features=features.to(
                        device=self.audio_tokenizer_device,
                        dtype=tokenizer_dtype,
                    ),
                    input_lens=lengths.to(self.audio_tokenizer_device),
                    return_codes_only=True
                )
                encoded_parts.append(codes)
            self._sync_progress_device(self.audio_tokenizer_device)
            self._progress(
                "Audio tokenizer encode "
                f"group {group_idx}/{len(groups)} done in {time.monotonic() - group_start:.2f}s "
                f"{_cuda_memory_text(self.audio_tokenizer_device)}"
            )

        return torch.cat(encoded_parts, dim=-1)

    def preprocess_input(
        self,
        input: Union[str, torch.Tensor],
        *,
        start_seconds: float = 0.0,
        duration_seconds: float | None = None,
    ):
        target_sr = self.mimo_audio_tokenizer.config.sampling_rate
        if isinstance(input, torch.Tensor):
            self._progress(
                f"Audio input is tensor: shape={tuple(input.shape)}, target_sr={target_sr}"
            )
            wav = input
        else:
            decode_start = time.monotonic()
            self._progress(f"Audio decode start: {input!r}")
            try:
                wav, sr = torchaudio.load(input)
                self._progress(
                    f"torchaudio decoded in {time.monotonic() - decode_start:.2f}s: "
                    f"shape={tuple(wav.shape)}, sr={sr}"
                )
            except (ImportError, RuntimeError, OSError) as exc:
                print(
                    f"torchaudio failed to decode {input!r} "
                    f"({exc.__class__.__name__}); falling back to ffmpeg."
                )
                ffmpeg_start = time.monotonic()
                wav, sr = _load_audio_with_ffmpeg(input, target_sr)
                self._progress(
                    f"ffmpeg decoded in {time.monotonic() - ffmpeg_start:.2f}s "
                    f"(total decode {time.monotonic() - decode_start:.2f}s): "
                    f"shape={tuple(wav.shape)}, sr={sr}"
                )
            resample_start = time.monotonic()
            wav = self.resample_audio_if_needed(wav, sr)
            self._progress(
                f"Audio resample/check done in {time.monotonic() - resample_start:.2f}s: "
                f"shape={tuple(wav.shape)}, target_sr={target_sr}"
            )
        if wav.ndim == 2:
            self._progress(f"Audio downmix to mono from {wav.shape[0]} channels")
            wav = wav.mean(dim=0)

        start_seconds = max(float(start_seconds or 0.0), 0.0)
        if duration_seconds is not None and duration_seconds <= 0:
            raise ValueError("--audio-duration-seconds must be > 0 when provided.")
        if start_seconds or duration_seconds is not None:
            start_sample = min(int(start_seconds * target_sr), wav.shape[-1])
            end_sample = wav.shape[-1]
            if duration_seconds is not None:
                end_sample = min(
                    wav.shape[-1],
                    start_sample + int(float(duration_seconds) * target_sr),
                )
            self._progress(
                f"Audio clip selected: start={start_seconds:.2f}s, "
                f"duration={(end_sample - start_sample) / target_sr:.2f}s"
            )
            wav = wav[..., start_sample:end_sample]
            if wav.numel() == 0:
                raise ValueError("Selected audio clip is empty.")

        wav = wav.detach().to(self.mel_device)
        self._sync_progress_device(self.mel_device)
        self._progress(
            f"Waveform ready: samples={wav.shape[-1]}, seconds={wav.shape[-1] / target_sr:.2f}, "
            f"mel_device={self.mel_device} {_cuda_memory_text(self.mel_device)}"
        )

        # Split waveform into 30s chunks, tokenize each separately, then concatenate codes
        chunk_samples = 30 * target_sr
        n_fft = self.mimo_audio_tokenizer.config.nfft

        total_samples = wav.shape[-1]
        code_parts = []
        start = 0
        chunk_ranges = []
        while start < total_samples:
            end = min(start + chunk_samples, total_samples)
            # Merge a too-short trailing chunk (would break mel reflect padding)
            # into the current one.
            if 0 < total_samples - end < n_fft:
                end = total_samples
            chunk_ranges.append((start, end))
            start = end

        self._progress(
            f"Audio preprocessing split into {len(chunk_ranges)} chunk(s), "
            f"chunk_seconds={chunk_samples / target_sr:.0f}"
        )
        for chunk_idx, (start, end) in enumerate(
            self._progress_iter(chunk_ranges, desc="Audio chunks", unit="chunk"),
            start=1,
        ):
            chunk_start = time.monotonic()
            chunk = wav[start:end]
            # Zero-pad if the entire audio is shorter than n_fft.
            if chunk.shape[-1] < n_fft:
                chunk = torch.nn.functional.pad(chunk, (0, n_fft - chunk.shape[-1]))
            self._progress(
                f"Chunk {chunk_idx}/{len(chunk_ranges)} mel start: "
                f"samples={end - start}, seconds={(end - start) / target_sr:.2f}"
            )
            mel_start = time.monotonic()
            mel = self.wav2mel(chunk).transpose(0, 1)  # (seq_len, n_mels)
            self._sync_progress_device(self.mel_device)
            self._progress(
                f"Chunk {chunk_idx}/{len(chunk_ranges)} mel done in "
                f"{time.monotonic() - mel_start:.2f}s: frames={mel.size(0)}, "
                f"n_mels={mel.size(1)} {_cuda_memory_text(self.mel_device)}"
            )
            codes_chunk = self.encode_batch(
                input_features=mel,
                input_lens=torch.tensor([mel.size(0)]),
            )
            code_parts.append(codes_chunk)
            self._progress(
                f"Chunk {chunk_idx}/{len(chunk_ranges)} done in "
                f"{time.monotonic() - chunk_start:.2f}s: codes_shape={tuple(codes_chunk.shape)}"
            )

        codes_packed = torch.cat(code_parts, dim=-1)
        codes = codes_packed.transpose(0, 1).detach().cpu()
        audio_codes = codes[:, :self.audio_channels]
        self._progress(
            f"Audio codes packed: shape={tuple(audio_codes.shape)}, "
            f"channels={self.audio_channels}"
        )

        # Pad the sequence to be a multiple of group_size by repeating the last frame
        num_timesteps = audio_codes.shape[0]
        if num_timesteps % self.group_size != 0:
            padding_needed = self.group_size - (num_timesteps % self.group_size)
            last_tokens = audio_codes[-1:, :] # Keep dim for repeat
            padding_tokens = last_tokens.repeat(padding_needed, 1)
            audio_codes = torch.cat([audio_codes, padding_tokens], dim=0)

        audio_tokenized = audio_codes.reshape(-1)
        self._progress(
            f"Audio tokens ready: timesteps={audio_codes.shape[0]}, "
            f"flat_tokens={audio_tokenized.numel()}, group_size={self.group_size}"
        )

        return audio_tokenized

    def get_input_ids(self, prompt):
        self._progress(f"Prompt assembly start: segments={len(prompt)}")
        input_ids = [
            seg.to_input_id(
                self.tokenizer,
                self.group_size,
                self.audio_channels,
            )
            for seg in prompt
        ]
        input_ids = torch.cat(input_ids, dim=1)
        self._progress(
            f"Prompt assembly done: shape={tuple(input_ids.shape)}, "
            f"device={self.device}"
        )
        return input_ids.to(self.device)


    def get_asr_sft_prompt(
        self,
        input: Union[None, str] = None,
        audio_tag="",
        *,
        audio_start_seconds: float = 0.0,
        audio_duration_seconds: float | None = None,
    ):
        self._progress(f"ASR prompt start: audio_tag={audio_tag or '<auto>'}")
        audio_tokenized = self.preprocess_input(
            input,
            start_seconds=audio_start_seconds,
            duration_seconds=audio_duration_seconds,
        )

        if '<chinese>' in audio_tag:
            template = random.choice(asr_zh_templates)
        elif '<english>' in audio_tag:
            template = random.choice(asr_en_templates)
        else:
            template = random.choice(asr_zh_templates + asr_en_templates)

        lm_prompt = [
            InputSegment(
                text=f"<|im_start|>user\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                audio=audio_tokenized,
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                text=template,
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                text=f"<|im_end|>\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                text=f"<|im_start|>assistant\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                text=f"<think>\n\n</think>\n{audio_tag}",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            )
        ]
        input_ids = self.get_input_ids(lm_prompt)
        self._progress(f"ASR prompt ready: input_ids_shape={tuple(input_ids.shape)}")
        return input_ids


    @torch.no_grad()
    def forward(
        self,
        input_ids,
        stopping_criteria=None,
        min_new_tokens=0,
        max_new_tokens=8192,
        task_name=None,
    ):

        task_sampler = self.get_task_sampler(task_name)

        generation_kwargs = self.generate_kwargs.copy()
        generation_config = GenerationConfig(**generation_kwargs)

        input_ids = input_ids.T.reshape(1, -1) # [B, flattened(T, audio_channels + 1)]

        prompt_length = input_ids.shape[1] // (self.audio_channels+1)

        max_length = prompt_length // self.group_size + max_new_tokens
        min_length = prompt_length // self.group_size + min_new_tokens
        generation_config.max_length = max_length
        self._progress(
            f"Generation start: prompt_groups={prompt_length // self.group_size}, "
            f"max_new_tokens={max_new_tokens}, max_groups={max_length}, "
            f"device={self.device} {_cuda_memory_text(self.device)}"
        )

        if stopping_criteria is not None:
            for criterion in stopping_criteria:
                if isinstance(criterion, MiMoStopper):
                    criterion.max_length = max_length
                    criterion.min_length = min_length

        generated_text_token_ids: list[int] = []

        def generation_progress(event):
            if event["event"] == "start":
                self._progress(
                    f"Generation loop entered: budget={event['total_new_tokens']} "
                    f"{event.get('cuda_memory', '')}"
                )
                return None
            if event["event"] == "end":
                self._progress(
                    f"Generation loop finished: generated={event['generated']}, "
                    f"current_groups={event['current_length']} {event.get('cuda_memory', '')}"
                )
                return None
            if event["event"] != "step":
                return None

            token_id = int(event["text_token_id"])
            if token_id != self.empty_token:
                generated_text_token_ids.append(token_id)

            if not event.get("should_report"):
                return None

            preview = ""
            if generated_text_token_ids:
                preview = self.tokenizer.decode(
                    generated_text_token_ids[-80:],
                    skip_special_tokens=False,
                )
                preview = (
                    preview.replace("<|empty|>", "")
                    .replace("<|eot|>", "")
                    .replace("<|eostm|>", "")
                    .strip()
                )
            if len(preview) > 80:
                preview = "..." + preview[-77:]
            return (
                f"{event['mode']}, step {event['step_seconds']:.2f}s, "
                f"local {event['local_seconds']:.2f}s, id={token_id}, text={preview!r}"
            )

        generated_ids = self.model.generate(
            input_ids,
            generation_config,
            stopping_criteria=stopping_criteria,
            global_sampler=task_sampler["global"],
            local_sampler=task_sampler["local"],
            progress=self.progress,
            progress_interval=self.progress_interval,
            progress_callback=generation_progress,
        )

        generated_ids = generated_ids.int().cpu().reshape(-1, self.audio_channels+1).T[:, prompt_length:]

        raw_text_tokens = generated_ids[0, ::self.group_size]
        text = raw_text_tokens[:-1]
        detokenized_text = self.tokenizer.decode(text, skip_special_tokens=False).strip().replace("<|empty|>", "").replace("<|eot|>", "").replace("<|eostm|>", "")
        print("Text channel:\t", detokenized_text)
        if not detokenized_text:
            token_count = int(text.numel())
            empty_count = int((text == self.empty_token).sum().item()) if token_count else 0
            unique_preview = sorted(set(int(token) for token in text[:32].tolist()))
            print(
                "Warning: generated text channel is empty. "
                f"{empty_count}/{token_count} generated text tokens were <|empty|> "
                f"(empty_token_id={self.empty_token}); first unique token ids: {unique_preview}. "
                "For ASR debugging, try --lang zh, a shorter clip such as "
                "--audio-duration-seconds 30, or a larger --max-new-tokens.",
                flush=True,
            )

        return detokenized_text

    def asr_sft(
        self,
        audio,
        audio_tag="",
        max_new_tokens=8192,
        audio_start_seconds: float = 0.0,
        audio_duration_seconds: float | None = None,
    ):
        stopping_criteria = [
            MiMoStopper(
                stop_tokens=[self.tokenizer.eos_token_id, self.im_end_idx],
                group_size=self.group_size,
                audio_channels=self.audio_channels,
            )
        ]
        input_ids = self.get_asr_sft_prompt(
            audio,
            audio_tag,
            audio_start_seconds=audio_start_seconds,
            audio_duration_seconds=audio_duration_seconds,
        )
        result = self.forward(
            input_ids,
            stopping_criteria=stopping_criteria,
            task_name="asr",
            max_new_tokens=max_new_tokens,
        )
        if '<chinese>' in result or '<english>' in result:
            result = result.replace('<chinese>', '').replace('<english>', '').strip()
        return result
