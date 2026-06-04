# Exporting The VibeVoice ASR LM Prefix

## Purpose

We need one small, reliable tool that takes an audio file and produces the exact tensor that the language model receives in the normal VibeVoice ASR demo.

The normal demo path is in `demo/vibevoice_asr_gradio_demo.py`.

It does this:

```python
inputs = self.processor(
    audio=audio_path,
    sampling_rate=sample_rate,
    return_tensors="pt",
    add_generation_prompt=True,
    context_info=context_info,
)

output_ids = self.model.generate(**inputs, **generation_config)
```

Inside the model, the raw audio is turned into audio embeddings. Those audio embeddings are inserted into the text prompt embeddings. After that insertion, the language model sees one big prefix embedding tensor.

That final tensor is what we want to export.

The export tool should answer this question:

> Given the same audio file and the same optional context text, did our script build exactly the same LM prefix that the upstream demo would build?

This document describes the missing tool and the checks it must perform.

## Important Vocabulary

### Prompt

The prompt is the text sequence sent to the model before it starts generating the transcript.

For ASR, the prompt contains:

1. A system message.
2. A user message.
3. A placeholder region where audio will go.
4. A written instruction such as "please transcribe it".
5. An assistant generation marker.

### Token IDs

The tokenizer converts the prompt text into integer IDs.

Example:

```text
"hello" -> [14990]
```

The exact IDs are model-specific. We should not hand-write them. We should let the tokenizer produce them.

### Embeddings

The language model does not directly read token IDs. First, token IDs are passed through the model's word embedding table.

This turns:

```text
input_ids: [batch, prompt_length]
```

into:

```text
text_embeddings: [batch, prompt_length, hidden_size]
```

For the VibeVoice ASR 7B model, `hidden_size` is `3584`.

### Prefix

In this document, "prefix" means the full tensor passed into the language model for the first generation step.

For VibeVoice ASR, that prefix contains both:

1. Normal text embeddings.
2. Audio embeddings inserted into special audio placeholder positions.

The final prefix tensor has this shape:

```text
inputs_embeds: [batch, prompt_length, hidden_size]
```

This is the tensor we want to export.

## The Three ASR Audio Tokens

VibeVoice ASR represents the audio area inside the text prompt using three special tokenizer tokens.

They look confusing because they reuse Qwen/Qwen2.5 token names that were originally meant for vision or object-reference tasks.

For VibeVoice ASR, treat them like this:

| Token string | Meaning in VibeVoice ASR |
|---|---|
| `<|object_ref_start|>` | Marks the start of the audio region. |
| `<|box_start|>` | One placeholder slot for one audio embedding frame. |
| `<|object_ref_end|>` | Marks the end of the audio region. |

So when you see this:

```text
<|object_ref_start|><|box_start|><|box_start|><|box_start|><|object_ref_end|>
```

read it as:

```text
audio starts here
audio frame slot 1
audio frame slot 2
audio frame slot 3
audio ends here
```

The token `<|box_start|>` is not a literal box for ASR. It is just the token string used as the repeated audio placeholder.

Only the `<|box_start|>` positions are replaced with audio embeddings.

The start and end tokens stay as normal text-token embeddings.

## How Many `<|box_start|>` Tokens Are Needed?

The processor computes one audio placeholder for every encoded audio frame.

The formula is:

```text
audio_frame_count = ceil(number_of_audio_samples / 3200)
```

The audio is resampled to 24 kHz first.

So one second of audio has:

```text
24000 samples / 3200 = 7.5
ceil(7.5) = 8 audio frames
```

That means one second of audio gets 8 copies of `<|box_start|>` in the prompt.

Example:

```text
<|object_ref_start|>
<|box_start|> repeated 8 times
<|object_ref_end|>
```

## What The Normal Demo Builds

For an audio file, the demo processor builds these values:

```text
input_ids
attention_mask
acoustic_input_mask
speech_tensors
speech_masks
```

Their meanings are:

| Name | Meaning |
|---|---|
| `input_ids` | Token IDs for the full ASR prompt. |
| `attention_mask` | Which prompt positions are real tokens instead of padding. |
| `acoustic_input_mask` | Boolean mask that is `True` exactly at `<|box_start|>` positions. |
| `speech_tensors` | The normalized 24 kHz audio waveform. |
| `speech_masks` | Boolean mask for valid encoded audio frames. |

Then the model does this:

```python
inputs_embeds = model.get_input_embeddings()(input_ids)
```

That creates normal text embeddings for every token, including the audio placeholder tokens.

Then the model encodes the audio:

```python
audio_features = model.encode_speech(
    speech_tensors=speech_tensors,
    speech_masks=speech_masks,
)
```

That creates:

```text
audio_features: [number_of_audio_frames, hidden_size]
```

Then the model replaces only the `<|box_start|>` embeddings:

```python
inputs_embeds = inputs_embeds.clone()
inputs_embeds[acoustic_input_mask] = audio_features
```

After this replacement, `inputs_embeds` is the final LM prefix.

That is the artifact we need to export.

## What Exists Today

### `tools/test_asr_tokenizer_roundtrip.py`

This checks the text/tokenizer part.

It verifies:

```text
input_ids
decoded prompt bytes
number of audio placeholder tokens
acoustic_input_mask
```

This is useful, but it does not encode audio and it does not export the final prefix tensor.

### `tools/export_audio_features_npz.py`

This exports only the audio feature block:

```text
output: [batch, audio_frames, hidden_size]
```

This output is the data that should replace the `<|box_start|>` token embeddings.

This is useful, but it is not the full LM prefix.

It does not include:

```text
input_ids
attention_mask
normal text embeddings
final spliced inputs_embeds
```

### `tools/smoke_spliced_audio_npz.py`

This test loads the audio feature block, builds the ASR prompt, inserts the audio features into the prompt embeddings, and runs a small Hugging Face decode test.

This is useful, but it is not a clean exporter.

It does not produce a reusable prefix artifact for another script.

## Missing Tool

Add a script named something like:

```text
tools/export_asr_prefix.py
```

The script should accept:

```text
--model-path microsoft/VibeVoice-ASR
--textonly-model-path out/textonly-checkpoint
--audio path/to/audio.wav
--context-info "optional hotwords or other notes"
--output out/prefix_fixture.npz
--format npz
--seed 0
--device cpu
--dtype float32
```

The script should write a file containing:

```text
input_ids
attention_mask
acoustic_input_mask
speech_masks
audio_features
inputs_embeds
```

The most important output is:

```text
inputs_embeds
```

That is the full prefix tensor after text embedding and audio insertion.

## Step-By-Step Behavior For The Exporter

The exporter should do the following work in this order.

### 1. Load And Normalize The Audio

Use the same audio loading behavior as the demo processor:

1. Read the audio file.
2. Convert it to mono audio.
3. Resample it to 24 kHz if needed.
4. Convert it to `float32`.
5. Apply the same VibeVoice audio normalizer.

The result is the waveform that the demo would put in `speech_tensors`.

### 2. Build The Same Prompt As The Demo

Call `VibeVoiceASRProcessor` the same way the demo does:

```python
inputs = processor(
    audio=audio_path,
    sampling_rate=sample_rate,
    return_tensors="pt",
    add_generation_prompt=True,
    context_info=context_info,
)
```

This produces the official reference prompt fields:

```text
input_ids
attention_mask
acoustic_input_mask
speech_tensors
speech_masks
```

The exporter should save at least:

```text
input_ids
attention_mask
acoustic_input_mask
speech_masks
```

Saving `speech_tensors` is optional. It is useful for debugging, but it may make the output file larger.

### 3. Encode The Audio

Run the separated audio encoder and produce:

```text
audio_features: [batch, audio_frames, hidden_size]
```

For a single input file, `batch` is normally `1`.

The number of `audio_frames` must equal the number of `True` values in `acoustic_input_mask`.

If those counts differ, the exporter should fail. That means the audio features cannot be inserted into the prompt correctly.

### 4. Embed The Text Prompt

Use the language model's token embedding table:

```python
inputs_embeds = embed_tokens(input_ids)
```

This produces text embeddings for every token in the prompt.

At this point, the `<|box_start|>` positions still contain ordinary token embeddings. They have not been replaced by audio yet.

### 5. Insert The Audio Features

Replace the embeddings at the `<|box_start|>` positions:

```python
inputs_embeds = inputs_embeds.clone()
inputs_embeds[acoustic_input_mask] = audio_features.reshape(-1, hidden_size)
```

After this line, `inputs_embeds` is the final LM prefix.

This is the main output of the tool.

## Required Comparison Mode

The exporter needs a comparison mode.

Example:

```text
tools/export_asr_prefix.py \
  --audio path/to/audio.wav \
  --context-info "Project names: VibeVoice, Qwen2.5" \
  --output out/prefix_fixture.npz \
  --compare-demo-path
```

This mode should build the reference using the exact same processor call as the demo:

```python
reference_inputs = processor(
    audio=audio_path,
    sampling_rate=sample_rate,
    return_tensors="pt",
    add_generation_prompt=True,
    context_info=context_info,
)
```

Then it should compare these values from the exporter against the reference:

```text
input_ids
attention_mask
acoustic_input_mask
speech_tensors
speech_masks
```

After that, it should compare the final prefix embeddings.

Reference construction:

```python
reference_embeds = model.get_input_embeddings()(
    reference_inputs["input_ids"]
).clone()

reference_features = model.encode_speech(
    speech_tensors=reference_inputs["speech_tensors"],
    speech_masks=reference_inputs["speech_masks"],
)

reference_embeds[reference_inputs["acoustic_input_mask"]] = reference_features
```

The exporter passes if:

```text
exported inputs_embeds == reference_embeds
```

Because floating-point math can vary by dtype and device, comparison mode should print:

```text
shape
dtype
allclose result
max absolute difference
mean absolute difference
```

For reproducible local tests, set the same random seed before both calls to `encode_speech`. This matters because the acoustic encoder samples from a distribution.

## NPZ Output

The first version should support `.npz`.

Recommended contents:

```text
input_ids: int64
attention_mask: int64
acoustic_input_mask: bool
speech_masks: bool
audio_features: float32
inputs_embeds: float32
```

Also include simple metadata values:

```text
source_model_path
textonly_model_path
audio_path
context_info
sample_rate
num_audio_samples
seed
dtype
speech_token_count
hidden_size
prompt_length
```

NPZ is acceptable for the first implementation because it is easy to inspect with Python and NumPy.

## Safetensors Output

A later version can also support `.safetensors`.

Recommended tensor contents:

```text
input_ids
attention_mask
acoustic_input_mask
speech_masks
audio_features
inputs_embeds
```

Recommended metadata:

```text
source_model_path
textonly_model_path
audio_path
context_info
sample_rate
num_audio_samples
seed
dtype
speech_token_count
hidden_size
prompt_length
```

Safetensors is useful when another runtime wants a tensor-only artifact without NumPy loading.

## What This Tool Should Not Do

This tool should not run generation.

This tool should not test vLLM.

This tool should not decide how another runtime accepts pre-embedded inputs.

This tool should only prove one thing:

> The exported prefix tensor is the same prefix tensor that the known demo path would construct before the LM starts generating.

---

## Ground Truth NPZ Format (Reference)

For validation, the `demo/vibevoice_asr_gradio_demo.py` script can be configured to dump ground truth fixtures into the `out/` directory. These files follow this format:

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `input_ids` | `[seq_len]` | `int64` | Full prompt token IDs |
| `attention_mask` | `[seq_len]` | `int64` | Padding mask for the LM |
| `acoustic_input_mask` | `[seq_len]` | `bool` | Mask marking `<|box_start|>` positions |
| `speech_masks` | `[num_frames]` | `bool` | Valid frame mask for the audio encoder |
| `audio_features` | `[batch, num_frames, hidden_size]` | `float32` | Raw output of the speech encoder |
| `inputs_embeds` | `[seq_len, hidden_size]` | `float32` | Final spliced prefix tensor passed to LM |
