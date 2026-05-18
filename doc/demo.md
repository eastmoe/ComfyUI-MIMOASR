# MiMo-V2.5-ASR Demo Notes

This file preserves the command-line and Python usage from the original project.
For ComfyUI usage, see the repository root `README.md`.

## Command Line

```bash
python demo.py path/to/audio.wav \
    --model ./model/MiMo-V2.5-ASR \
    --audio-tokenizer ./model/MiMo-Audio-Tokenizer
```

Language hints are optional:

```bash
python demo.py path/to/audio.wav \
    --model ./model/MiMo-V2.5-ASR \
    --audio-tokenizer ./model/MiMo-Audio-Tokenizer \
    --lang zh
```

Built-in weight-only quantization can reduce memory usage for the ASR model:

```bash
python demo.py path/to/audio.wav \
    --model ./model/MiMo-V2.5-ASR \
    --audio-tokenizer ./model/MiMo-Audio-Tokenizer \
    --quantization int4 \
    --quantization-cache-dir auto
```

Useful low-VRAM options:

```bash
python demo.py path/to/audio.wav \
    --model ./model/MiMo-V2.5-ASR \
    --audio-tokenizer ./model/MiMo-Audio-Tokenizer \
    --quantization int4 \
    --mel-device cpu \
    --audio-tokenizer-device cpu \
    --quantized-linear-chunk-size 512
```

## Demo Options

| Option | Default | Description |
| --- | --- | --- |
| `audio` | required | Path to the input audio file. |
| `--model` | `model` | Path to the ASR model directory. |
| `--audio-tokenizer` | `None` | Path to the MiMo-Audio-Tokenizer directory. |
| `--lang` | `auto` | Language hint: `auto`, `zh`, or `en`. |
| `--device` | auto | Device for the ASR language model. |
| `--mel-device` | `cpu` | Device for mel spectrogram calculation. |
| `--audio-tokenizer-device` | `auto` | Device for MiMo-Audio-Tokenizer. |
| `--quantization` | `none` | Choices: `none`, `int8`, `int4`, `nf4`, `fp4`, `fp8`. |
| `--bnb-4bit-compute-dtype` | `auto` | Compatibility option. |
| `--bnb-4bit-double-quant` | disabled | Compatibility option. |
| `--quantized-linear-chunk-size` | `1024` | Rows per chunk for quantized Linear dequantization. |
| `--quantization-cache-dir` | disabled | Directory for reusable quantized model cache. |
| `--rebuild-quantization-cache` | disabled | Rebuild the quantized cache. |
| `--max-new-tokens` | `8192` | Maximum generated text-token groups. |
| `--progress` | enabled | Show detailed progress. |
| `--no-progress` | disabled | Disable detailed progress. |
| `--progress-interval` | `2.0` | Seconds between progress updates. |
| `--audio-start-seconds` | `0.0` | Start offset for ASR input audio. |
| `--audio-duration-seconds` | `None` | Limit ASR to this many seconds of audio. |

## Python API

```python
from src.mimo_audio.mimo_audio import MimoAudio

model = MimoAudio(
    model_path="./model/MiMo-V2.5-ASR",
    mimo_audio_tokenizer_path="./model/MiMo-Audio-Tokenizer",
    quantization="int4",
    mel_device="cpu",
    audio_tokenizer_device="auto",
    quantized_linear_chunk_size=1024,
    quantization_cache_dir="temp/quantized-cache",
    rebuild_quantization_cache=False,
    progress=True,
    progress_interval=2.0,
)

text = model.asr_sft("path/to/audio.wav")
text_zh = model.asr_sft("path/to/audio.wav", audio_tag="<chinese>")
text_clip = model.asr_sft(
    "path/to/audio.wav",
    audio_tag="<chinese>",
    max_new_tokens=512,
    audio_start_seconds=0.0,
    audio_duration_seconds=30.0,
)
```
