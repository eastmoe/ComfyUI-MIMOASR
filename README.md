# ComfyUI-MIMOASR

ComfyUI-MIMOASR wraps Xiaomi MiMo-V2.5-ASR as native ComfyUI audio nodes.

## Nodes

| Node | Purpose |
| --- | --- |
| `MiMo ASR Load Model` | Loads MiMo-V2.5-ASR and MiMo-Audio-Tokenizer. |
| `MiMo ASR Audio To Text` | Converts ComfyUI `AUDIO` input to a text transcript. |
| `MiMo ASR Release Model` | Releases the loaded model weights and clears CUDA cache. |

## Model Layout

By default the loader uses ComfyUI's main `models` folder:

```text
ComfyUI/
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

Place this folder under `ComfyUI/custom_nodes/ComfyUI-MIMOASR`, then install the
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
5. Optionally run `MiMo ASR Release Model` when you want to unload weights.

The loader exposes the same core loading controls as the original demo:
device placement, mel/audio-tokenizer device, quantization, quantization cache,
chunk size, progress logging, and bitsandbytes-compatible options.

The transcription node exposes language hint, generation token limit, audio
start offset, duration clipping, and batch selection for batched ComfyUI audio.

## Localization

Simplified Chinese node definitions are included under `locales/zh/nodeDefs.json`
with a `locales/zh-CN/nodeDefs.json` compatibility copy. They translate node
names, input and output labels, tooltips, and combo options in ComfyUI's
language system.

## Documentation

- Original model README: [doc/mimo-v2.5-asr.md](doc/mimo-v2.5-asr.md)
- Original demo/API notes: [doc/demo.md](doc/demo.md)
