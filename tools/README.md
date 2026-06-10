# VibeVoice ASR Split Tools

This folder contains the current split client/server workflow for VibeVoice ASR.
The intended runtime path is:

```text
client:
  audio file
  minimal HF-keyed audio bundle
  HF acoustic feature extractor
  HF acoustic/semantic encoders + projector
  -> projected audio rows

server:
  vLLM Chat Completions
  normal text strings
  prompt_embeds content part containing projected audio rows
  server-side tokenizer + WTE
  -> transcript
```

The client does not need the Qwen decoder, Qwen WTE, tokenizer files, or ASR
processor files for this mixed-embedding path.

## Export The Audio Bundle

Use `tools/split-encode-lm/export_hf_audio_encoder.py` to convert the older
repo-keyed split into the minimal HF-keyed audio bundle:

```bash
UV_CACHE_DIR=/tmp/vibevoice-uv-cache uv run --script \
  tools/split-encode-lm/export_hf_audio_encoder.py \
  --input-dir out/audioonly-checkpoint \
  --output-dir out/audioonly-hf-mixed-checkpoint
```

By default the output contains only:

```text
audio_encoder.safetensors
config.json
bundle_config.json
```

The default export intentionally omits WTE and tokenizer/processor files.

Optional compatibility flags:

```bash
--include-wte
--include-processor-files
```

These are for older full-`inputs_embeds` experiments and parity checks, not the
preferred mixed vLLM runtime path.

## Run The Mixed vLLM Client

`tools/asr_mixed_embed_client.py` is the copyable runtime prototype. It defaults
to the Hugging Face bundle `jkeisling/vibevoice-encoder-only`, and also accepts
a local bundle path.

Start vLLM with prompt embeds enabled. The mixed Chat Completions content-part
path requires a recent vLLM, verified with `v0.22.1`:

```bash
vllm serve /models/vibevoice --enable-prompt-embeds
```

Then run:

```bash
UV_CACHE_DIR=/tmp/vibevoice-uv-cache uv run --script \
  tools/asr_mixed_embed_client.py \
  flight.wav \
  --url http://localhost:8000 \
  --model /models/vibevoice
```

For local bundle validation:

```bash
UV_CACHE_DIR=/tmp/vibevoice-uv-cache uv run --script \
  tools/asr_mixed_embed_client.py \
  flight.wav \
  --checkpoint out/audioonly-hf-mixed-checkpoint \
  --device cpu \
  --dtype float32 \
  --max-tokens 64
```

The script has `EXTENSION POINT` comments for batching, VAD/long-audio splits,
lazy module loading, multiple content parts, and alternate wire formats.

## Validate The Export

`tools/test-encode-split/check_audioonly_npz.py` is the validation/parity script.
It is broader than the runtime client: it can still compare old fixtures and can
opt into the legacy full-`inputs_embeds` path.

Mixed-mode smoke against the minimal bundle:

```bash
UV_CACHE_DIR=/tmp/vibevoice-uv-cache uv run --script \
  tools/test-encode-split/check_audioonly_npz.py \
  --checkpoint out/audioonly-hf-mixed-checkpoint \
  --processor-path out/audioonly-hf-checkpoint \
  --audio flight.wav \
  --reference-npz '' \
  --output-npz out/test-encode-split/audioonly_mixed_checkpoint_check.npz \
  --device cpu \
  --dtype float32 \
  --no-default-compare-keys \
  --post-mixed-vllm \
  --max-tokens 64
```

`--processor-path` is only needed by this validation script when checking a
minimal bundle, because the validator can still exercise tokenizer-era parity
paths. The standalone mixed client does not need processor/tokenizer files.

## Legacy Tools

These remain useful for historical comparison or debugging, but they are not the
current runtime contract:

- `tools/export_asr_prefix.py`: old full prefix / `inputs_embeds` exporter.
- `tools/post_asr_prompt_embeds.py`: old Completions `prompt_embeds` poster.
- `tools/smoke_spliced_audio_npz.py`: local full-prefix generation smoke.
- `tools/export_audio_features_npz.py`: older repo-module audio feature export.

Prefer the minimal exporter plus `asr_mixed_embed_client.py` for new work.
