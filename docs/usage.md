## Startup

```bash
docker run --runtime nvidia --gpus all -v ./out/textonly-checkpoint:/models/vibevoice -p 8000:8000 --ipc=host vllm/vllm-openai:latest /models/vibevoice
```

## Export ASR Prefixes

Use `tools/export_asr_prefix.py` when the audio encoder and text-only LM are
served separately and you need the exact prefix tensor that the normal
VibeVoice ASR demo would pass to the language model.

The exporter writes an `.npz` with:

- `input_ids`
- `attention_mask`
- `acoustic_input_mask`
- `speech_masks`
- `audio_features`
- `inputs_embeds`

`inputs_embeds` is the final LM prefix after the `<|box_start|>` placeholder
embeddings have been replaced by encoded audio features.

```bash
.venv/bin/python tools/export_asr_prefix.py \
  --model-path microsoft/VibeVoice-ASR \
  --textonly-model-path out/textonly-checkpoint \
  --audio flight.wav \
  --output out/flight_prefix.npz \
  --device cpu \
  --dtype bfloat16 \
  --seed 0
```

If the ASR checkpoint and text-only checkpoint are already local, add
`--local-files-only`.

To include ASR hotwords or other transcription context, pass the same text you
would send to the demo:

```bash
.venv/bin/python tools/export_asr_prefix.py \
  --audio flight.wav \
  --context-info "Hotwords: VibeVoice, Qwen2.5" \
  --output out/flight_prefix_with_context.npz
```

To verify a new export against a fixture, use `--compare-npz`. Token and mask
arrays must match exactly; floating arrays report shape, dtype, allclose,
maximum absolute difference, and mean absolute difference.

```bash
.venv/bin/python tools/export_asr_prefix.py \
  --audio flight.wav \
  --output out/flight_prefix.npz \
  --compare-npz tests/fixtures/asr/flight.npz \
  --local-files-only \
  --device cpu \
  --dtype bfloat16 \
  --seed 0
```

For a heavier end-to-end check against the full ASR model path used by the demo,
add `--compare-demo-path`.

## POST ASR Prefix Embeddings to vLLM

Use `tools/post_asr_prompt_embeds.py` for a quick end-to-end validation against a
vLLM Completions server started with prompt embeds enabled:

```bash
docker run --runtime nvidia --gpus all \
  -v ./out/textonly-checkpoint:/models/vibevoice \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai:latest /models/vibevoice \
  --enable-prompt-embeds
```

Then build the same prefix tensor as `tools/export_asr_prefix.py`, POST it as
`prompt_embeds`, stop on `<|endoftext|>`, and print the generated text:

```bash
.venv/bin/python tools/post_asr_prompt_embeds.py \
  flight.wav \
  --local-files-only \
  --device cpu \
  --dtype bfloat16
```

The request defaults to `http://localhost:8000/v1/completions` and model
`/models/vibevoice`. Use `--url` or `--model` if your vLLM server differs.

The prefix includes the normal assistant generation marker
`<|im_start|>assistant\n`, so the completion should start directly with the
transcription JSON rather than first emitting `assistant`. A local `flight.wav`
validation produced:

```text
prompt_length=621 speech_tokens=558 hidden_size=3584
finish_reason=stop
usage={"completion_tokens": 431, "prompt_tokens": 621, "total_tokens": 1052}
```
