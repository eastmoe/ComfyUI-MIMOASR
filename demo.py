# Copyright 2025 Xiaomi Corporation.
import argparse
import time


LANGUAGE_TAGS = {
    "auto": "",
    "zh": "<chinese>",
    "en": "<english>",
}


def main():
    parser = argparse.ArgumentParser(description="Simple MiMo-V2.5-ASR demo")
    parser.add_argument("audio", help="Path to the audio file")
    parser.add_argument("--model", default="model", help="Path to the ASR model")
    parser.add_argument(
        "--audio-tokenizer",
        default=None,
        help="Path to the MiMo audio tokenizer, e.g. ./model/MiMo-Audio-Tokenizer.",
    )
    parser.add_argument(
        "--lang",
        choices=LANGUAGE_TAGS,
        default="auto",
        help="Language hint: auto, zh, or en",
    )
    parser.add_argument("--device", default=None, help="cuda, cpu, or leave unset")
    parser.add_argument(
        "--mel-device",
        default="cpu",
        help="Device for mel spectrogram calculation: cpu, cuda, or auto. Default: cpu.",
    )
    parser.add_argument(
        "--audio-tokenizer-device",
        default="auto",
        help=(
            "Device for MiMo-Audio-Tokenizer: cpu, cuda, or auto. "
            "Default: auto (uses model device, but CPU on ROCm/HIP)."
        ),
    )
    parser.add_argument(
        "--quantization",
        choices=["none", "int8", "int4", "nf4", "fp4", "fp8"],
        default="none",
        help=(
            "Optional built-in weight-only quantization for the ASR model."
        ),
    )
    parser.add_argument(
        "--bnb-4bit-compute-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Compatibility option; built-in quantization follows the model input dtype.",
    )
    parser.add_argument(
        "--bnb-4bit-double-quant",
        action="store_true",
        help="Compatibility option; ignored by built-in quantization.",
    )
    parser.add_argument(
        "--quantized-linear-chunk-size",
        type=int,
        default=1024,
        help=(
            "Rows per chunk when dequantizing large quantized Linear layers. "
            "Use 0 to disable chunked dequantization. Default: 1024."
        ),
    )
    parser.add_argument(
        "--quantization-cache-dir",
        default=None,
        help=(
            "Directory for reusable built-in quantized model cache. "
            "Use 'auto' for ./temp/quantized-cache. Disabled by default."
        ),
    )
    parser.add_argument(
        "--rebuild-quantization-cache",
        action="store_true",
        help="Ignore an existing quantized cache file and rebuild it.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8192,
        help="Maximum generated text-token groups. Lower this for debugging long runs. Default: 8192.",
    )
    parser.add_argument(
        "--progress",
        dest="progress",
        action="store_true",
        default=True,
        help="Show detailed stage progress while decoding and generating. Default: enabled.",
    )
    parser.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="Disable detailed progress output.",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=2.0,
        help="Minimum seconds between progress detail updates. Default: 2.0.",
    )
    parser.add_argument(
        "--audio-start-seconds",
        type=float,
        default=0.0,
        help="Start offset for ASR input audio. Useful for debugging long files. Default: 0.",
    )
    parser.add_argument(
        "--audio-duration-seconds",
        type=float,
        default=None,
        help="Limit ASR to this many seconds of audio. Useful for debugging empty output.",
    )
    args = parser.parse_args()

    try:
        from src.mimo_audio.mimo_audio import MimoAudio
    except ModuleNotFoundError as exc:
        missing = exc.name or "required package"
        raise SystemExit(
            f"Missing dependency: {missing}. Run `pip install -r requirements.txt` first."
        ) from exc

    start = time.time()
    try:
        model = MimoAudio(
            model_path=args.model,
            mimo_audio_tokenizer_path=args.audio_tokenizer,
            device=args.device,
            mel_device=args.mel_device,
            audio_tokenizer_device=args.audio_tokenizer_device,
            quantization=args.quantization,
            quantized_linear_chunk_size=args.quantized_linear_chunk_size,
            quantization_cache_dir=args.quantization_cache_dir,
            rebuild_quantization_cache=args.rebuild_quantization_cache,
            bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
            bnb_4bit_use_double_quant=args.bnb_4bit_double_quant,
            progress=args.progress,
            progress_interval=args.progress_interval,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc

    text = model.asr_sft(
        args.audio,
        audio_tag=LANGUAGE_TAGS[args.lang],
        max_new_tokens=args.max_new_tokens,
        audio_start_seconds=args.audio_start_seconds,
        audio_duration_seconds=args.audio_duration_seconds,
    )
    print(text)
    print(f"\nDone in {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
