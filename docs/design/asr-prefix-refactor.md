# ASR Prefix Bundle: Portable Client Design

## Goal

Replace the repo-local prefix export pipeline with a portable ASR prefix bundle that can be consumed by external clients (including future MLX ports) without importing this repository.

**Current pipeline to replace:**
- `tools/export_asr_prefix.py` — builds prefix by importing local `VibeVoiceASR*` classes
- `tools/post_asr_prompt_embeds.py` — thin HTTP wrapper that delegates ML work back to `export_asr_prefix.py`

**Target architecture:**
```
client (outside this repo):
  audio + optional prompt/context
  -> tokenizer/processor metadata
  -> WTE rows + audio rows
  -> prompt_embeds or mixed prompt_embeds request

server:
  vLLM or Transformers decoder on constrained GPU
  -> generate transcript
```

The current scripts should become parity oracles and fixture generators. The runtime client should load a small, versioned, documented bundle.

---

## Verified HF Transformers Surface

Do not rely on documentation alone. These versions were checked against `microsoft/VibeVoice-ASR-HF`:

| Transformers | `VibeVoiceAsrForConditionalGeneration` | `VibeVoiceAsrProcessor` |
|-------------|--------------------------------------|------------------------|
| 4.51.3      | No                                   | No                     |
| 4.57.6      | No                                   | No                     |
| 5.10.2      | Yes                                  | Yes                    |

Model revision verified: `microsoft/VibeVoice-ASR-HF` @ `f22241c2062b3b25272bf117397e03d73381037a`

**Verified API in 5.10.2:**
```python
from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration

model_id = "microsoft/VibeVoice-ASR-HF"
processor = AutoProcessor.from_pretrained(model_id)
model = VibeVoiceAsrForConditionalGeneration.from_pretrained(model_id)

inputs = processor.apply_transcription_request(audio=..., prompt=...)
output_ids = model.generate(**inputs)
```

**Observed behavior:**
- `AutoProcessor.from_pretrained` returns `VibeVoiceAsrProcessor`
- `AutoTokenizer.from_pretrained` returns plain `Qwen2Tokenizer`; ASR special tokens and chat template live in repo serialization files
- `apply_transcription_request` returns `input_values`, `padding_mask`, `input_ids`, `attention_mask`
- 1 second of 24kHz audio pads to `input_values` shape `[1, 1, 25600]` with `padding_mask` `[1, 25600]`
- 1 second of zero audio produces 8 audio placeholder tokens
- ASR token IDs:
  - audio BOS `<|object_ref_start|>`: `151646`
  - audio EOS `<|object_ref_end|>`: `151647`
  - audio placeholder `<|box_start|>`: `151648`
- `VibeVoiceAsrForConditionalGeneration.forward` accepts `inputs_embeds`, `input_values`, `padding_mask`, `acoustic_tokenizer_chunk_size`
- `VibeVoiceAsrModel.get_audio_features` exists and returns projected audio embeddings via `pooler_output`
- WTE provenance: `{"lm_head.weight": "model.language_model.embed_tokens.weight"}`
- Config values:
  - `model_type`: `vibevoice_asr`
  - text config: `qwen2`, hidden size `3584`, vocab size `152064`
  - acoustic chunk size: `1440000`
  - acoustic encoder hidden size: `64`
  - semantic encoder hidden size: `128`
  - hop length: `3200`
  - acoustic VAE std: `0.625`

**Conclusion:** The upstream API already covers prompt construction (`AutoProcessor`) and decoder execution from `inputs_embeds`. The gap is not "can Transformers tokenize this?" but "how does a client build the prefix without importing this repo or carrying full decoder weights?"

---

## Bundle Structure

```
asr-prefix-bundle/
  README.md
  bundle_config.json
  config.json
  tokenizer.json
  tokenizer_config.json
  chat_template.jinja
  processor_config.json
  audio_encoder.safetensors      (includes WTE under original key)
  audio_encoder_config.json
```

`bundle_config.json`:
```json
{
  "format": "vibevoice-asr-prefix-bundle-v1",
  "source_model": "microsoft/VibeVoice-ASR-HF",
  "source_revision": "<commit used to produce this bundle>",
  "decoder_model": "<server-side model id or compatible family>",
  "hidden_size": 3584,
  "sample_rate": 24000,
  "speech_token_compress_ratio": 3200,
  "speech_tokens": {
    "start": "<|object_ref_start|>",
    "pad": "<|box_start|>",
    "end": "<|object_ref_end|>"
  },
  "default_bundle": "jkeisling/vibevoice-asr-encoder",
  "wte_key": "model.language_model.embed_tokens.weight",
  "audio_encoder_weight_format": "hf-vibevoice-asr-audio-v1"
}
```

Note: `wte.safetensors` is no longer a separate file. The WTE tensor lives inside `audio_encoder.safetensors` under its original checkpoint key. Clients that need WTE for full prompt embeddings (Mode B) can load it from the audio encoder directly. Mode A (mixed prompt) clients still don't need WTE at all.

The runtime client should load one immutable bundle version. The bundle records provenance; it should not ask the user to resolve a source checkpoint at runtime.

---

## Component Details

### WTE Extraction

`tools/extract_lm_trunk.py` extracts the WTE tensor (`model.language_model.embed_tokens.weight` for the current source checkpoint) and includes it in both outputs:
- **LM trunk:** renamed to `model.embed_tokens.weight` for standalone Qwen2 compatibility
- **Audio encoder:** preserved with its original checkpoint key

**Key mapping:**
```
source ASR key:  model.language_model.embed_tokens.weight
LM trunk key:    model.embed_tokens.weight          (Qwen2 standalone)
audio_encoder key: model.language_model.embed_tokens.weight   (original preserved)
```

For HF-format checkpoints, confirm the source key from the model index. In `transformers==5.10.2`, `lm_head.weight` is tied to `model.language_model.embed_tokens.weight`, so that is the expected source key.

**Size:** For the 7B decoder, WTE is ~1.0 GiB in bf16/fp16 (`152064 * 3584 * 2`). This is much smaller than the full decoder but not tiny. If the client can use vLLM's mixed text-token-plus-supplied-embedding path, it may not need WTE at all.

**Recommended changes:**
1. Add `tools/extract_asr_prefix_bundle.py`
2. Add `--components wte,audio-encoder,tokenizer` so local experiments can build partial bundles
3. Keep WTE in audio_encoder with original key for clients building full prompt embeddings
4. Keep `extract_lm_trunk.py` for full standalone decoder extraction, but stop pointing the prefix client at the full trunk by default
5. For MLX or non-Python clients, also export an optional `wte.npz` or document the safetensors key and dtype so clients can read it directly
6. Every split checkpoint that carries WTE or decoder weights — including the LM trunk — must also ship the merged `tokenizer.json` and `tokenizer_config.json` so it can be loaded with vanilla `AutoTokenizer` without re-merging against the base Qwen repo

### Tokenizer and Processor

The ASR special tokens (`<|object_ref_start|>`, `<|object_ref_end|>`, `<|box_start|>`) must be baked into the serialized tokenizer files, not injected at runtime by a custom class.

**How the tokenizer is produced:**

`tools/export_asr_text_tokenizer.py` loads the base Qwen tokenizer, calls `add_special_tokens()` to inject the ASR tokens, sets the chat template, and then calls `save_pretrained()`. This writes the extended vocabulary and added-token metadata into ordinary `tokenizer.json` / `tokenizer_config.json` files.

After that one-time export, every consumer loads it with vanilla tools:

**HF Processor Mode (default for Torch/Transformers clients):**
```python
processor = AutoProcessor.from_pretrained(bundle_or_model_id)
inputs = processor.apply_transcription_request(audio=audio, prompt=prompt)
```

Requires Transformers >= 5.10.2.

**Bare Tokenizer Mode (for MLX or non-Python clients):**
```python
from tokenizers import Tokenizer
tok = Tokenizer.from_file("bundle/tokenizer.json")
```

The bundle must include:
- `tokenizer.json`
- `tokenizer_config.json`
- `chat_template.jinja` (or the template string in `tokenizer_config.json`)
- speech token strings and IDs in `bundle_config.json`

Runtime code must not import `VibeVoiceASRTextTokenizerFast`. Read token IDs from the manifest or look them up with `convert_tokens_to_ids()`. Do not depend on custom class properties like `speech_start_id`.

**Split checkpoint rule:**

Every split checkpoint that needs to tokenize or detokenize — the LM trunk, the audio-encoder bundle, the full-prefix bundle, etc. — must ship its own copy of the merged tokenizer files and load them with `AutoTokenizer.from_pretrained(local_path)`. Do not point at the base Qwen repo and re-apply the special-token graft.

**Testing note:** `test_asr_tokenizer_roundtrip.py` already proves that `Tokenizer.from_file` on the exported `tokenizer.json` produces byte-for-byte identical IDs to the custom class. Use it to guard regressions, not to justify keeping the custom class.

### Audio Encoder Weights

The audio path includes:
- audio loading, mono conversion, resampling, normalization
- acoustic tokenizer encode
- stochastic acoustic latent sampling (or mean mode)
- semantic tokenizer encode
- streaming caches for long audio
- connector projections into decoder hidden size
- frame mask handling and shape checks

`tools/export_audio_features_npz.py`, `export_asr_prefix.py`, and `vllm_plugin/model.py` currently duplicate large parts of this logic. Centralize it behind one API and one fixture suite before any MLX port.

**Bundle contents:**
```
audio_encoder.safetensors:
  model.acoustic_tokenizer.*          (original keys preserved, no prefix stripping)
  model.semantic_tokenizer.*
  model.acoustic_connector.*
  model.semantic_connector.*
  model.language_model.embed_tokens.weight   (WTE, original key)

audio_encoder_config.json:
  acoustic_tokenizer_encoder_config
  semantic_tokenizer_encoder_config
  text_hidden_size
  hidden_size
  sample_rate
  streaming_chunk_samples
  acoustic_sampling_mode
```

**Key format:** As of the `extract_lm_trunk.py` refactor, audio encoder keys are preserved exactly as they appear in the source checkpoint — no `model.` prefix is stripped. The WTE tensor (`model.language_model.embed_tokens.weight` in the current source checkpoint) is also copied into `audio_encoder.safetensors` with its original key, so clients building full prompt embeddings can source it from the audio encoder without needing a separate WTE file.

**Recommended Python API:**
```python
features = audio_encoder.encode(
    input_values,
    padding_mask=None,
    chunk_size=1440000,
    seed=0,
)
```

For the first portable Torch client, use the Transformers implementation rather than repo-local audio classes. `VibeVoiceAsrModel.get_audio_features` is verified in `transformers==5.10.2`. The open engineering question is how to instantiate/load only `acoustic_tokenizer_encoder`, `semantic_tokenizer_encoder`, and `multi_modal_projector` without carrying decoder weights.

---

## Revision Handling

Runtime clients should not need `--revision` when using a vendored bundle.

Use revisions in exactly two places:
1. **Bundle creation:** resolve the source checkpoint at a specific commit
2. **Bundle metadata:** record the source commit and compatible decoder commit

Avoid a pinned default revision on generic `--model-path`. A commit hash for `microsoft/VibeVoice-ASR` is not meaningful for `microsoft/VibeVoice-ASR-HF` or for a local bundle path.

```
--source-model microsoft/VibeVoice-ASR-HF
--source-revision <optional>

If omitted:
  use the model's current resolved revision at bundle creation time
  write the resolved commit into bundle_config.json

Runtime:
  no revision argument
```

---

## Runtime Client Modes

### Mode A: Mixed Prompt, Server Embeds Text (preferred long-term)

```
client:
  processor/tokenizer -> input_ids + acoustic_input_mask
  audio encoder -> audio embeddings only

server:
  WTE embeds normal token IDs
  supplied audio rows replace placeholder rows
```

The client does not need WTE. Relies on vLLM accepting mixed token IDs plus embeddings. The existing `docs/design/prompt-embeds-vibevoice-notes.md` says vLLM supports this in the native/offline prompt interface, while OpenAI-compatible Completions is mainly a full-prompt embedding path.

**Challenge:** wire/API shape, not model math.

### Mode B: Full Prompt Embeddings

```
client:
  processor/tokenizer -> input_ids
  WTE -> text rows
  audio encoder -> audio rows
  splice -> full inputs_embeds

server:
  Completions prompt_embeds only
```

Easiest server integration. Matches the existing `post_asr_prompt_embeds.py` approach. Requires WTE on the client.

### Mode C: Server-Side Transformers Thin Client

```
client:
  sends audio and prompt

server:
  AutoProcessor + VibeVoiceAsrForConditionalGeneration
```

Official HF path. Correctness baseline. Does not solve VRAM split by itself unless customized.

---

## Implementation Phases

### Completed: Tokenizer Runtime Cleanup (2026-06-05)

Done for this incremental refactor:
- Kept `tools/export_asr_text_tokenizer.py` and
  `tools/test_asr_tokenizer_roundtrip.py` as one-off development tooling for
  creating and validating vendored tokenizer files.
- Added `docs/development/asr-prefix-dev-workflow.md` with the tokenizer export,
  roundtrip check, and `flight.wav` fixture regeneration process.
- Confirmed the exported ASR tokenizer
  manifest and can be loaded as a plain `Qwen2TokenizerFast`.
- Removed the custom `VibeVoiceASRTextTokenizerFast` import from
  `tools/export_asr_prefix.py`; prefix export now loads tokenizer files through
  `AutoTokenizer`.
- Updated `VibeVoiceASRProcessor` so plain tokenizer instances resolve the ASR
  speech token IDs from serialized token strings.
- Verified `tools/test_asr_tokenizer_roundtrip.py` passes byte-for-byte with
  local cached files.
- Verified the `flight.wav` prefix exporter output is identical between the
  custom-tokenizer loader and the vanilla `AutoTokenizer` loader for
  `input_ids`, masks, audio features, and final `inputs_embeds`.

### Completed: Audio-Only Bundle Runtime Increment (2026-06-05)

Done for this incremental refactor:
- Changed `tools/export_asr_prefix.py` and `tools/post_asr_prompt_embeds.py` to
  default to `jkeisling/vibevoice-asr-encoder` as the ASR prefix bundle.
- Removed runtime `--revision`, text-only checkpoint, and base language-model
  fallback defaults from the prefix export/post path.
- Fixed `tools/extract_lm_trunk.py` so the audio encoder split includes the
  actual source WTE key (`model.language_model.embed_tokens.weight`) and fails
  if no known WTE key exists.
- Added bundle-only config inference from `audio_encoder.safetensors`, so a
  local/remote bundle without `config.json` can instantiate the audio modules
  from connector tensor shapes.
- Updated `tools/export_asr_text_tokenizer.py` so future exported tokenizer
  configs advertise `Qwen2TokenizerFast`.
- Verified `out/audioonly-checkpoint` contains audio encoder weights, merged
  tokenizer files, and embedded WTE, and can reproduce the `flight.wav` prefix
  fixture exactly without reading the source ASR checkpoint or text-only split.

### Phase 0: Stop The Bleeding

- Fix `post_asr_prompt_embeds.py` so it passes `revision` or so `build_prefix()` tolerates absent revision
- Remove pinned revision as a default for arbitrary model IDs
- Rename current scripts/docs to clarify their role: `export_asr_prefix.py` is a parity/export tool, not the future runtime client

### Phase 1: Create The Bundle Builder

Add `tools/extract_asr_prefix_bundle.py`. It should:
- resolve a source model once
- export tokenizer/processor files
- export WTE-only weights
- export audio encoder weights and config
- write `bundle_config.json`
- optionally create a README snippet showing both full-embeds and mixed-mode usage

This replaces the expectation that a client points at `out/textonly-checkpoint`.

### Phase 2: Split Runtime Code From Parity Code

Create a small importable module:
```
tools/asr_prefix_client/
  bundle.py
  processor.py
  wte.py
  audio_encoder.py
  prompt_embeds.py
  vllm_wire.py
```

Public API:
```python
bundle = AsrPrefixBundle.from_pretrained(path)
request = bundle.prepare_request(audio=path, prompt=prompt)
embeds = bundle.build_prompt_embeds(request)
payload = bundle.to_vllm_prompt_embeds_payload(embeds)
```

The current `export_asr_prefix.py` should call this library and compare its output against the full demo/HF path.

### Phase 3: Switch Torch Runtime To HF-First APIs

Replace local tokenizer/processor usage with:
```python
AutoProcessor.from_pretrained(...)
processor.apply_transcription_request(...)
```

Use `VibeVoiceAsrForConditionalGeneration` only for the parity oracle and for checking that generated output still matches the official path.

Require a verified Transformers version in this path:
```
transformers==4.51.3: no
transformers==4.57.6: no
transformers==5.10.2: yes
```

Do not block the bundle on perfect removal of every local import for old-format checkpoints. It is fine for a legacy bundle builder to remain repo-aware temporarily. The runtime client is the piece that must be repo-independent.

### Phase 4: MLX Audio Encoder Port

Port only after the Torch bundle client is stable.

Required fixtures:
- short audio direct path
- long audio streaming path
- audio normalization and resampling
- `input_ids` and prompt bytes for contexts with punctuation/newlines/Unicode
- acoustic sampling with fixed seed, or explicit mean-mode if chosen
- final `inputs_embeds` allclose against Torch

---

## Testing Strategy

Keep parity tests and runtime tests separate.

**Parity tests:**
- Compare against full HF/repo model path
- Expensive, optional, fixture-generating
- Own "exact same prefix as the demo"

**Runtime tests:**
- Load a bundle from disk
- Build prompt IDs and masks from tokenizer files
- Load WTE-only weights
- Load audio-encoder-only weights
- Produce full `inputs_embeds`
- POST or serialize without requiring the full ASR checkpoint

**Regression cases:**
- no context
- hotwords/context
- multiline context
- Unicode context
- audio just under and just over the streaming threshold
- local bundle path with no `revision`
- HF model path with optional source revision during bundle creation

---

## Decisions

1. WTE-only export is worth doing even though it is ~1 GiB for 7B.
2. The runtime bundle should vendor tokenizer/processor metadata and weights, then record source revisions as provenance.
3. Runtime code should not import `vibevoice.*`.
4. `AutoProcessor`/`apply_transcription_request` on `microsoft/VibeVoice-ASR-HF` with Transformers 5.10.2 is the verified tokenizer/processor path for Torch clients. Future Transformers versions should stay on this path, but CI should check it explicitly.
5. The ASR special-token extension is done once at bundle-build time (`add_special_tokens()` + `save_pretrained()`). Runtime clients load the pre-baked tokenizer with vanilla `AutoTokenizer` or `tokenizers.Tokenizer`. No custom tokenizer class at runtime.
6. Every split checkpoint (LM trunk, audio encoder, prefix bundle) ships its own copy of the merged tokenizer files. Consumers load them locally with `AutoTokenizer.from_pretrained(bundle_path)`, not by re-merging against the base Qwen repo.
7. A bare client should use serialized tokenizer files plus manifest constants, not custom tokenizer class properties.
8. The current exporter remains valuable as a parity oracle, but should stop being the implementation that `post_asr_prompt_embeds.py` depends on forever.

---

## Open Questions

1. Does the current Transformers implementation expose a stable enough audio encoder submodule to load only audio weights without instantiating the full decoder?
2. Should the portable bundle support stochastic acoustic sampling, mean-mode, or both? Mean-mode is easier for cross-runtime determinism, but may not match the official generation path.
3. Should vLLM integration target full-prompt `prompt_embeds` first, then mixed-mode token IDs plus audio embeddings, or jump directly to mixed mode to avoid shipping WTE to clients?
4. What is the acceptable client artifact size for macOS? WTE-only is much smaller than the full decoder but still large.
5. Should non-Python clients emit PyTorch `torch.save` base64 for vLLM compatibility, or should the server grow a safetensors/NPY/raw tensor input?
