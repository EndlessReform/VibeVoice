# ASR Prefix Dev Workflow

This note is for maintainers working on the split ASR prefix path. The runtime
goal is simple: vendored checkpoint or bundle folders carry the merged tokenizer
files, and runtime code loads those files with vanilla `AutoTokenizer` or
`tokenizers.Tokenizer`. The custom `VibeVoiceASRTextTokenizerFast` path is only
used as one-off development tooling to create and verify those serialized files.

## Export The Merged Tokenizer

Use this when preparing a text-only checkpoint, prefix bundle, or other vendored
folder that needs to tokenize ASR prompts without importing this repository's
custom tokenizer class.

```bash
PYTHONPATH=. .venv/bin/python tools/export_asr_text_tokenizer.py \
  --model-path microsoft/VibeVoice-ASR \
  --language-model Qwen/Qwen2.5-7B \
  --output-dir out/textonly-checkpoint \
  --local-files-only
```

Drop `--local-files-only` if the source tokenizer or ASR config is not already
cached locally.

The export writes ordinary tokenizer assets such as:

- `tokenizer.json`
- `tokenizer_config.json`
- `chat_template.jinja`
- `added_tokens.json`
- `special_tokens_map.json`
- `vibevoice_asr_tokenizer_export.json`

The important ASR tokens must be present in the saved files:

```text
<|object_ref_start|> 151646
<|object_ref_end|>   151647
<|box_start|>        151648
```

After export, runtime code should load the folder directly:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("out/textonly-checkpoint")
```

Do not re-merge against the base Qwen tokenizer at runtime.

## Verify Tokenizer Roundtrip

Run the roundtrip check after changing tokenizer export code, prompt rendering,
chat templates, ASR token IDs, or vendored tokenizer files.

```bash
PYTHONPATH=. .venv/bin/python tools/test_asr_tokenizer_roundtrip.py \
  --model-path microsoft/VibeVoice-ASR \
  --language-model Qwen/Qwen2.5-7B \
  --local-files-only
```

This test compares the existing custom-tokenizer processor path against a bare
`tokenizers.Tokenizer.from_file("tokenizer.json")` path. It checks exact input
IDs, decoded UTF-8 prompt bytes, audio placeholder counts, multiline context,
Unicode context, and a long placeholder span.

## Regenerate Flight Prefix Fixtures

Use this when prompt construction, tokenizer files, audio encoder loading, WTE
loading, or prefix splicing changes.

```bash
PYTHONPATH=. .venv/bin/python tools/export_asr_prefix.py \
  --audio flight.wav \
  --output out/flight_prefix_fixture_check.npz \
  --local-files-only \
  --device cpu \
  --dtype bfloat16 \
  --seed 0

PYTHONPATH=. .venv/bin/python tools/export_asr_prefix.py \
  --audio flight.wav \
  --output out/flight_prefix_test.npz \
  --compare-npz out/flight_prefix_fixture_check.npz \
  --local-files-only \
  --device cpu \
  --dtype bfloat16 \
  --seed 0

PYTHONPATH=. .venv/bin/python tools/export_asr_prefix.py \
  --audio flight.wav \
  --output out/flight_prefix_test_fp32.npz \
  --local-files-only \
  --device cpu \
  --dtype float32 \
  --seed 0

PYTHONPATH=. .venv/bin/python tools/export_asr_prefix.py \
  --audio flight.wav \
  --output out/flight_prefix_demo_compare.npz \
  --compare-demo-path \
  --local-files-only \
  --device cpu \
  --dtype bfloat16 \
  --seed 0
```

Expected current `flight.wav` metadata:

```text
prompt_length=621
speech_tokens=558
hidden_size=3584
input_ids=(621,)
inputs_embeds=(621, 3584)
```

The 621-token prefix includes the assistant generation marker
`<|im_start|>assistant\n`. Older 618-token fixtures omitted that marker and
should be considered stale.

## Quick Sanity Checks

Confirm a vendored tokenizer folder loads as a plain tokenizer and contains the
ASR tokens:

```bash
PYTHONPATH=. .venv/bin/python - <<'PY'
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("out/textonly-checkpoint", local_files_only=True)
print(type(tok))
for token in ("<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>"):
    print(token, tok.convert_tokens_to_ids(token))
PY
```

Check generated fixture shapes:

```bash
PYTHONPATH=. .venv/bin/python - <<'PY'
from pathlib import Path
import numpy as np

for path in sorted(Path("out").glob("flight_prefix_*.npz")):
    data = np.load(path)
    print(
        f"{path}: input_ids={data['input_ids'].shape} "
        f"inputs_embeds={data['inputs_embeds'].shape} "
        f"speech_tokens={int(data['speech_token_count'])} "
        f"hidden_size={int(data['hidden_size'])}"
    )
PY
```

## What Each Tool Is For

- `tools/export_asr_text_tokenizer.py`: one-off exporter that creates vendored
  tokenizer files by applying the ASR token merge and saving the result.
- `tools/test_asr_tokenizer_roundtrip.py`: regression oracle proving the
  exported `tokenizer.json` produces the same prompt IDs and bytes as the
  custom tokenizer path.
- `tools/export_asr_prefix.py`: parity/export tool that builds the full LM
  prefix tensor. It should load tokenizer files with `AutoTokenizer`.
- `tools/post_asr_prompt_embeds.py`: local vLLM smoke helper that delegates to
  `export_asr_prefix.py` until the portable runtime client exists.

