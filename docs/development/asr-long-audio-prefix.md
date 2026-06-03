# VibeVoice ASR Long-Audio Prefix Mechanics

This note explains how the default VibeVoice ASR path handles long audio and why a 20 minute input can work without breaking the LM prefix or KV cache.

## Short Version

The language model still sees the audio as one up-front prefix.

For long clips, the raw waveform is segmented only inside the audio encoder. Those encoder segments are not LM turns. They are a preprocessing strategy used to compute the same kind of dense audio feature sequence that would otherwise be computed from the whole waveform at once.

After audio encoding finishes, the model has:

- normal text token embeddings from WTE
- one dense audio embedding per `<|box_start|>` speech placeholder
- one contiguous initial prefix embedding tensor

Then the LM does a normal prefill over that entire prefix and generation proceeds autoregressively with KV cache.

## Default Gradio Path

The ASR Gradio demo calls:

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

There is one call to `model.generate(...)` per transcription request. The `TextIteratorStreamer` in the demo streams generated text tokens to the UI, but it does not split the audio into conversational turns.

So for a 20 minute input:

1. The processor builds one prompt with hundreds or thousands of speech placeholder tokens.
2. The model encodes the entire audio into matching dense embeddings.
3. The LM prefills once over the full prompt.
4. The LM decodes text tokens with normal KV cache.

## Processor Stage

`VibeVoiceASRProcessor` does the audio loading and prompt construction.

For file input, it:

1. Loads mono audio with ffmpeg at the file's native sample rate.
2. Resamples to 24 kHz with `librosa` if needed.
3. Converts to `float32`.
4. Applies `AudioNormalizer`.
5. Computes the number of speech placeholder tokens:

```python
vae_tok_len = math.ceil(len(audio_array) / speech_tok_compress_ratio)
```

The default compression ratio is `3200`, so the audio token rate is:

```text
24000 / 3200 = 7.5 audio frames per second
```

A 20 minute clip is approximately:

```text
20 * 60 * 7.5 = 9000 speech placeholder tokens
```

The prompt contains:

```text
system message
user message:
  <|object_ref_start|>
  <|box_start|> repeated vae_tok_len times
  <|object_ref_end|>
  "This is a ... seconds audio, please transcribe it..."
assistant generation prompt
```

The processor also returns:

- `input_ids`: the full text/speech-placeholder prompt
- `acoustic_input_mask`: `True` exactly at the `<|box_start|>` slots
- `speech_tensors`: the normalized 24 kHz waveform
- `speech_masks`: valid audio-frame positions

There is no transcript truncation at this stage unless a caller passes truncation/max-length options through batching. The normal Gradio path does not chunk the prompt into multiple LM calls.

## Audio Encoder Stage

`VibeVoiceASRForConditionalGeneration.forward(...)` starts by WTE-embedding the text prompt:

```python
inputs_embeds = self.get_input_embeddings()(input_ids)
```

Then, if `speech_tensors` and `acoustic_input_mask` are present, it calls:

```python
speech_features = self.encode_speech(
    speech_tensors=speech_tensors,
    speech_masks=speech_masks,
    speech_semantic_tensors=speech_semantic_tensors,
)

inputs_embeds = inputs_embeds.clone()
inputs_embeds[acoustic_input_mask] = speech_features
```

That replacement is the splice point.

The tensor spliced into WTE output has shape:

```text
[num_speech_slots, hidden_size]
```

For a single example before flattening, it is naturally:

```text
[batch, audio_frames, hidden_size]
```

For the local `flight.wav` smoke test, that was:

```text
[1, 558, 3584]
```

## Acoustic Plus Semantic Features

The ASR audio feature at each speech slot is:

```python
combined = acoustic_connector(acoustic_tokens) + semantic_connector(semantic_tokens)
```

The acoustic side samples from the acoustic tokenizer output. The semantic side uses the mean:

```python
audio_tokens = acoustic_encoder_output.sample(
    dist_type=acoustic_tokenizer.std_dist_type
)[0]

semantic_tokens = semantic_tokenizer.encode(...).mean
```

This matters for fixture generation: if you want an allclose fixture, you must control the RNG seed before acoustic sampling.

## Long Audio: Encoder Streaming, Not LM Streaming

For audio longer than `streaming_segment_duration` seconds, default `60.0`, `encode_speech(...)` switches to the streaming encoder path:

```python
segment_samples = int(streaming_segment_duration * 24000)
use_streaming = total_samples > segment_samples
```

In that path, the waveform is processed in 60 second chunks by the acoustic and semantic tokenizers. Each tokenizer receives a `VibeVoiceTokenizerStreamingCache`.

Conceptually:

1. Split raw waveform into 60 second chunks.
2. Run acoustic encoder chunk by chunk with convolution cache.
3. Run semantic encoder chunk by chunk with convolution cache.
4. Concatenate all acoustic means.
5. Sample acoustic tokens once from the concatenated acoustic distribution.
6. Concatenate all semantic means.
7. Project acoustic and semantic streams through their connectors.
8. Add them.
9. Splice the final combined sequence into the LM prefix.

The cache here is not the LM KV cache. It is an audio-tokenizer convolution/state cache used so segmented encoder computation behaves like a continuous long encode.

This is the subtle point: long audio is not rolled out as multiple LM turns. It is segmented before the LM, then reassembled into one audio-feature prefix.

## Why Chunk Boundaries Do Not Create Ordinary CNN Edge Artifacts

The tokenizer encoder is built from causal 1D convolutions, not symmetric non-causal convolutions over isolated chunks.

In `SConv1d`, streaming mode is only allowed for causal convolutions:

```python
assert self.causal, "Streaming mode is only supported for causal convolutions"
```

For each causal convolution layer, the module computes a left-context size:

```python
context_size = (kernel_size - 1) * dilation - (stride - 1)
```

During streaming encode, each chunk is not convolved as if it began from silence. Instead:

1. The layer retrieves cached trailing samples/states from the previous chunk.
2. It prepends that cache to the new chunk.
3. It runs the convolution over `cached_context + current_chunk`.
4. It stores the last `context_size` samples for the next chunk.

In code shape:

```python
cached_states = cache.get(layer_id, sample_indices)
input_with_context = torch.cat([cached_states, x], dim=2)
output = conv(input_with_context)
new_cache = input_with_context[:, :, -context_size:]
cache.set(layer_id, sample_indices, new_cache)
```

That is the main anti-edge-artifact mechanism. Every layer sees the left receptive-field context it needs at chunk boundaries, so the first outputs of chunk N are conditioned on the tail of chunk N-1 rather than on fresh padding.

There is no learned pooling layer whose job is to stitch chunk edges. The continuity comes from causal convolution plus explicit per-layer caches.

### What About Padding?

There is still padding, but it is not repeatedly injected at every chunk boundary.

For non-streaming causal conv, the whole input is padded like:

```python
x = pad1d(x, (padding_total, extra_padding), ...)
```

That means left padding at the beginning of the full signal, plus enough right padding to make the strided output length come out with ceil behavior.

For streaming causal conv:

- the first chunk starts with a zero cache, so only the true beginning of the whole recording has artificial left context
- middle chunks use cached real context from the previous chunk
- only the final chunk gets right-side `extra_padding` for stride/output-length alignment

The final-chunk logic is explicit:

```python
if is_final_chunk:
    extra_padding = get_extra_padding_for_conv1d(...)
    input_with_context = pad1d(input_with_context, (0, extra_padding), ...)
```

So the "padding fuckery" is controlled: begin-of-audio gets initial zero context, interior chunk edges get cached real context, end-of-audio gets right padding only once to match the expected token count.

### Does Streaming Exactly Equal Non-Streaming?

Not necessarily bit-for-bit for every possible length/configuration. In our fixture work, the important point was the opposite: the default model path for long audio uses the streaming encoder, so the fixture must use that same path to match `model.encode_speech(...)`.

The code is designed so streaming chunks behave as a continuous causal convolutional encode, avoiding isolated-window boundary artifacts. But the streaming path is the reference for long clips in this repo, not merely an optimization we can freely swap with the full non-streaming call.

## LM Prefill And KV Cache

After splicing, the LM receives one `inputs_embeds` tensor:

```text
[batch, prompt_length, hidden_size]
```

This tensor contains both:

- WTE embeddings for ordinary text tokens
- audio encoder embeddings at speech placeholder positions

The first LM forward pass is a normal prefill over that whole prefix. The resulting KV cache contains keys/values for every prefix position, including the positions where audio embeddings replaced placeholder-token embeddings.

After that, decoding is standard autoregressive generation:

```text
next token -> append to attention mask -> use past_key_values -> next token ...
```

So the assumption "the LM must get whole audio up front or the prefix/KV cache breaks" is basically right at the LM boundary. What changes is that "whole audio up front" means "the complete audio embedding sequence is ready before LM prefill", not "the raw waveform must be passed through the audio encoder as one giant tensor".

## Are There Turns?

There are no additional LM turns in the default ASR transcription path.

There are three different things that can sound like "streaming":

1. Audio encoder streaming: internal 60 second waveform chunks with tokenizer caches.
2. UI token streaming: `TextIteratorStreamer` yields generated text as it arrives.
3. LM KV cache decoding: normal autoregressive decode after prefill.

Only the first one chunks the audio, and only before the LM sees the prefix.

The Gradio "streaming output" is category 2. It does not mean the model is repeatedly called on successive audio windows as separate turns.

## Is Anything Truncated?

Not by the long-audio encoder path itself.

The practical limit is the model context length. Since audio is compressed to 7.5 frames/sec, the speech part of the prompt is:

```text
duration_seconds * 7.5
```

Approximate examples:

```text
60 seconds   ->    450 speech slots
20 minutes   ->   9000 speech slots
60 minutes   ->  27000 speech slots
```

Then add system/user text prompt tokens and generated output tokens. This is why a 64K-context ASR model can plausibly handle long recordings: the raw waveform is not represented at sample rate in the LM context.

If a caller explicitly asks the processor to truncate during batching, or if the prompt exceeds model/context limits, then truncation or failure can happen. That is separate from the default encoder streaming behavior.

## Why The NPZ Fixture Needed Streaming

The local fixture work exposed an important trap.

For `flight.wav`, a non-streaming standalone encode produced a tensor with the right shape:

```text
[1, 558, 3584]
```

but it did not match `model.encode_speech(...)`, because `flight.wav` is longer than 60 seconds and the real model used the streaming encoder path.

The corrected fixture had to match all of these:

1. ffmpeg native-rate load
2. `librosa` resample to 24 kHz
3. `AudioNormalizer`
4. acoustic and semantic tokenizer streaming caches for clips over 60 seconds
5. acoustic sampling with controlled seed
6. acoustic connector plus semantic connector

With those matched, the fixture compared exactly:

```text
encoder_compare=allclose:True max_abs_diff:0 mean_abs_diff:0
```

and manual cached decode from the spliced prefix produced the same Japanese transcription shape as the normal direct-audio path.

## One More Trap: `generate(inputs_embeds=...)`

A direct forward comparison showed that the normal audio path and the pre-spliced `inputs_embeds` path produced identical prefix logits.

However, calling Hugging Face `generate(...)` with both `input_ids` and `inputs_embeds` for this custom model produced bad repetitive text in this repo state. A manual cached greedy loop from the spliced prefix produced the expected Japanese output.

So for smoke testing this specific splice point, the robust checks are:

1. Compare `npz["output"]` against `model.encode_speech(...)`.
2. Compare direct-prefix logits against spliced-prefix logits.
3. If decoding from `inputs_embeds`, use a manual cached decode loop unless/until `generate(inputs_embeds=...)` is fixed for this model.
