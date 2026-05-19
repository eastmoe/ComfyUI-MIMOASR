# Comfy-MIMOASR

Comfy-MIMOASR wraps Xiaomi MiMo-V2.5-ASR as native ComfyUI audio nodes.

## Nodes

| Node | Purpose |
| --- | --- |
| `MiMo ASR Load Model` | Loads MiMo-V2.5-ASR and MiMo-Audio-Tokenizer. |
| `MiMo ASR Audio To Text` | Converts ComfyUI `AUDIO` input to a text transcript. |
| `SILERO-VAD Audio Segmenter` | Splits ComfyUI `AUDIO` into timestamped speech segments. |
| `MiMo ASR Timed Audio To Text` | Runs ASR over each timestamped segment and returns timestamped text. |
| `MiMo ASR Release Model` | Releases the loaded model weights and clears CUDA cache. |
| `MiMo ASR Save Transcript` | Saves transcripts as auto-numbered `.txt`, `.json`, `.jsonl`, `.srt`, `.ass`, or custom-extension files. |

## Model Layout

By default the loader uses ComfyUI's main `models` folder:

```text
ComfyUI/
  custom_nodes/
    Comfy-MIMOASR/
      vad/
        silero_vad.jit
  models/
    MiMo-Audio-Tokenizer/
    MiMo-V2.5-ASR/
    MiMo-V2.5-ASR-quantized-cache/
```

If either model folder is missing, the loader can download from Hugging Face or
`hf-mirror`:

- `XiaomiMiMo/MiMo-V2.5-ASR`
- `XiaomiMiMo/MiMo-Audio-Tokenizer`

Set `download_source` to `offline` if you want missing models to fail instead of
downloading.

## Install

Place this folder under `ComfyUI/custom_nodes/Comfy-MIMOASR`, then install the
requirements in the same Python environment used by ComfyUI:

```bash
pip install -r requirements.txt
```

Restart ComfyUI after installation.

## Usage

1. Add `MiMo ASR Load Model`.
2. Connect its `mimo_asr_model` output to `MiMo ASR Audio To Text`.
3. Connect a ComfyUI native `AUDIO` output to the `audio` input.
4. Use the returned `STRING` output as transcript text.
5. Optionally connect the transcript to `MiMo ASR Save Transcript` to write a transcript file.
6. Optionally run `MiMo ASR Release Model` when you want to unload weights.

The loader exposes the same core loading controls as the original demo:
device placement, mel/audio-tokenizer device, quantization, quantization cache,
chunk size, progress logging, and bitsandbytes-compatible options.

The transcription node exposes language hint, generation token limit, audio
start offset, duration clipping, and batch selection for batched ComfyUI audio.
The default generation token limit is 128.

For timestamped transcription:

1. Connect ComfyUI native `AUDIO` to `SILERO-VAD Audio Segmenter`.
2. Keep `vad_model_path` as `auto` when `vad/silero_vad.jit` exists in this
   extension folder.
3. Connect `timed_audio` to `MiMo ASR Timed Audio To Text`.
4. Choose `bracket`, `srt`, or `jsonl` timestamp output. Use `jsonl` when you want the save node to generate `.srt` or `.ass` subtitles.

The timed audio output is a dictionary with a `segments` list. Each segment
contains `start`, `end`, `duration`, sample coordinates, and a ComfyUI-style
`audio` dictionary for that slice.

The save transcript node defaults to ComfyUI's `output` directory and names
files like `transcript_00001_.txt`. Use `filename_prefix` for output subfolders
or `output_directory` for a custom save directory. `file_format` controls the
extension and content conversion. `srt` and `ass` are generated from timestamped
JSONL lines with `start`, `end`, and `text` fields, such as the `jsonl` output
from `MiMo ASR Timed Audio To Text`.

Long-running MiMo ASR nodes check ComfyUI's interrupt flag during VAD scanning,
audio preprocessing, segment loops, and generation, so the ComfyUI cancel button
can stop queued work instead of waiting for the full transcription to finish.

## Localization

Simplified Chinese node definitions are included under `locales/zh/nodeDefs.json`
with a `locales/zh-CN/nodeDefs.json` compatibility copy. They translate node
names, input and output labels, tooltips, and combo options in ComfyUI's
language system.

## Documentation

- Original model README: [doc/mimo-v2.5-asr.md](doc/mimo-v2.5-asr.md)
- Original demo/API notes: [doc/demo.md](doc/demo.md)
