# vLLM Prompt Embeddings Notes for VibeVoice Experiments

This note records the local code-reading result for using precomputed prompt
embeddings with vLLM. The motivating use case is serving VibeVoice 9B from the
official vLLM repository instead of a fork, while possibly moving the initial
audio encode to a macOS client using MLX to save server VRAM.

## Summary

vLLM currently has a user-facing `prompt_embeds` input path. Token IDs are not
the lowest-level supported prompt representation, provided the server is started
with `--enable-prompt-embeds`.

The practical shape is:

```text
(num_tokens, hidden_size)
```

For HTTP entrypoints, the tensor is sent as base64-encoded `torch.save` bytes.
For offline Python usage, it can be passed directly as a `torch.Tensor`.

This means the VibeVoice experiment can plausibly send model-ready hidden
vectors over the wire, as long as the client produces embeddings in exactly the
same space the model expects at its decoder input.

## Relevant vLLM Entry Points

### Offline `LLM.generate`

The offline API accepts an `EmbedsPrompt`:

```python
outputs = llm.generate({"prompt_embeds": prompt_embeds})
```

Relevant files:

- `vllm/inputs/llm.py`: defines `EmbedsPrompt`.
- `vllm/renderers/base.py`: validates and processes `prompt_embeds`.
- `examples/features/prompt_embed/prompt_embed_offline.py`: minimal example.

Server setup is not involved in this path, but the `LLM` object still needs:

```python
llm = LLM(model=model_name, enable_prompt_embeds=True)
```

### OpenAI-Compatible Completions

The Completions request schema includes:

```python
prompt_embeds: bytes | list[bytes] | None = None
```

Usage shape:

```python
from vllm.utils.serial_utils import tensor2base64

encoded = tensor2base64(prompt_embeds)

client.completions.create(
    model=model_name,
    prompt=None,
    max_tokens=...,
    extra_body={"prompt_embeds": encoded},
)
```

For Completions, vLLM does not apply a chat template. The caller must build the
full prompt representation before embedding it. If the model expects system
tokens, role markers, audio placeholders, generation markers, or other special
tokens, those need to be reflected in the embedding sequence sent to vLLM.

Relevant files:

- `vllm/entrypoints/openai/completion/protocol.py`
- `vllm/entrypoints/serve/render/serving.py`
- `examples/features/prompt_embed/prompt_embed_inference_with_openai_client.py`

### OpenAI-Compatible Chat Completions

Chat Completions supports content parts of type `prompt_embeds`:

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "prompt_embeds", "data": encoded_embeds},
        ],
    },
]
```

For Chat Completions, the example embeds only the content span. vLLM renders the
surrounding chat template and splices the embedding content into the rendered
token stream.

This is useful for mixed text-plus-embedding prompts, but it also means the
content-part embedding should not already include the full chat template unless
the goal is to bypass Chat Completions and use the Completions endpoint instead.

Relevant files:

- `vllm/entrypoints/chat_utils.py`
- `vllm/renderers/hf.py`
- `examples/features/prompt_embed/prompt_embed_inference_with_openai_client.py`

## Server Flag and Validation

The feature is off by default:

```bash
vllm serve <model> --enable-prompt-embeds
```

The config warns that this should only be enabled for trusted users because bad
embedding shapes can crash the engine.

Validation happens in `vllm/renderers/embed_utils.py`:

- the payload must decode with base64 validation;
- the decoded object must be a `torch.Tensor`;
- tensors with a singleton batch dimension may be squeezed;
- final rank must be 2;
- shape must be `(num_tokens, model_hidden_size)`;
- dtype must be floating point;
- dtype is cast to the server model dtype if needed.

The HTTP wire format currently uses `torch.save` plus base64:

```python
from vllm.utils.serial_utils import tensor2base64

encoded = tensor2base64(tensor)
```

For a non-Python client, the client must reproduce a payload that
`torch.load(..., weights_only=True, map_location="cpu")` can read, or a small
compatibility shim would need to be added server-side for another tensor format.

## How vLLM Uses the Embeddings

The V1 GPU runner has a path where `prompt_embeds` are copied into the runner's
`inputs_embeds` buffer and passed into the model instead of using `input_ids`
alone.

## Interleaving Text Tokens and Supplied Embeddings

Interleaving is supported. The native public prompt schema exposes it with
`prompt_token_ids`, `prompt_embeds`, and `prompt_is_token_ids`.

The contract is:

```python
{
    "prompt_token_ids": [...],       # length == total rendered sequence length
    "prompt_embeds": tensor,         # shape == (total rendered sequence length, hidden_size)
    "prompt_is_token_ids": [...],    # same length as prompt_token_ids
}
```

The mask meaning is:

```text
prompt_is_token_ids[i] == True
  use prompt_token_ids[i] and run the model's normal embedding layer / WTE

prompt_is_token_ids[i] == False
  use prompt_embeds[i] directly
```

So the embedding tensor is full-length, not just the audio span. For positions
where `prompt_is_token_ids` is `True`, the supplied row may be a placeholder
because vLLM overwrites that position with the server-side token embedding. For
positions where the mask is `False`, `prompt_token_ids` contains a placeholder
token ID and vLLM keeps the supplied embedding row.

Internally, `prompt_is_token_ids` is renamed to `is_token_ids` in
`vllm/inputs/engine.py`, but the user-facing `EmbedsPrompt` key is
`prompt_is_token_ids`.

OpenAI Chat Completions builds this machinery for `prompt_embeds` content parts:
the chat renderer inserts a reserved placeholder token, expands it to the
embedding span length, builds the full-length `prompt_embeds` tensor, and builds
the mask. Native/offline callers can provide the three fields directly.

The OpenAI-compatible Completions request exposes `prompt_embeds`, but not an
explicit `prompt_is_token_ids` mask. Treat Completions as the full-prompt
embedding path unless adding a custom request extension. For mixed text tokens
plus supplied embedding spans over HTTP, Chat Completions content parts are the
existing user-facing route.

Sketch for a native/offline mixed prompt:

```python
# Example logical layout:
#   text token, text token, audio embed row, audio embed row, text token

prompt_token_ids = [tok_a, tok_b, placeholder_id, placeholder_id, tok_c]

prompt_embeds = torch.empty(5, hidden_size)
prompt_embeds[2:4] = audio_embeds
# Rows 0, 1, and 4 are ignored because their mask entries are True.

prompt_is_token_ids = [True, True, False, False, True]

outputs = llm.generate({
    "prompt_token_ids": prompt_token_ids,
    "prompt_embeds": prompt_embeds,
    "prompt_is_token_ids": prompt_is_token_ids,
})
```

For mixed-mode input, vLLM keeps a mask:

```text
True  -> use token ID and run the model's embedding layer
False -> use the supplied prompt embedding row
```

This is the important bit for the VibeVoice idea: vLLM can splice externally
computed embeddings into a prompt and still embed normal text tokens itself.

Relevant files:

- `vllm/inputs/llm.py`: `prompt_token_ids` plus `prompt_is_token_ids`.
- `vllm/renderers/hf.py`: expands prompt-embed placeholders and builds the mask.
- `vllm/v1/worker/gpu_model_runner.py`: prepares `inputs_embeds`.

## Mapping to the VibeVoice Use Case

### Option A: Client Encodes Audio, Server Embeds Text

Desired flow:

```text
macOS client:
  audio -> MLX audio encoder/projector -> audio embedding rows

server:
  text tokens -> vLLM WTE
  audio rows -> supplied prompt_embeds
  merged sequence -> model prefill/decode
```

This is closest to the mixed-mode prompt-embeds path. It should be possible if
the model's prompt format can represent the audio span with a placeholder in
the rendered prompt, and if the supplied rows already match the language model
hidden size and expected normalization/projection.

This is likely the best first target because it avoids reimplementing text WTE
and tokenizer/template behavior on the client.

### Option B: Client Sends Full Embedding Sequence

Desired flow:

```text
macOS client:
  text -> tokenizer -> WTE
  audio -> MLX audio encoder/projector
  concatenate -> full prompt embeddings

server:
  prompt_embeds only
```

This is the fallback if mixed text-plus-embedding is awkward for VibeVoice's
prompt format. Use the Completions endpoint, not Chat Completions, and send the
full already-templated sequence as `prompt_embeds`.

The user-mentioned `[bsz, seqlen, hidden_dim]` shape is only partially aligned
with current validation. vLLM can squeeze a singleton batch dimension, so
`[1, seqlen, hidden_dim]` should become `[seqlen, hidden_dim]`. General
`bsz > 1` is not accepted as one tensor; use a list of per-request encoded
tensors instead.

### Option C: Custom VibeVoice Multimodal Input

If the official repo lacks a clean way to map VibeVoice audio features into the
generic `prompt_embeds` splice path, a model-specific multimodal adapter may
still be needed. Several vLLM model files already support modality-specific
`*_embeds` inputs, such as image embeddings for some vision-language models, but
those are model-specific contracts rather than a universal arbitrary tensor API.

This would be more invasive than using `prompt_embeds`, but it may be necessary
if VibeVoice requires additional modality metadata, special position handling,
or nonstandard interleaving rules.

## Known Limitations and Gotchas

- `prompt_embeds` requires `--enable-prompt-embeds`.
- Treat it as trusted-client-only. The config warning is explicit.
- Shape must match the server model hidden size.
- HTTP payloads currently expect PyTorch serialization, which is inconvenient
  for a pure MLX/macOS client.
- `echo` is unsupported with prompt embeds.
- `prompt_logprobs` is incompatible with prompt embeds.
- Streaming inputs reject `prompt_embeds`.
- The server does not know that supplied vectors are audio; it only sees hidden
  states. Any VibeVoice-specific audio semantics must already be encoded in the
  vectors and prompt layout.
- Prompt-template correctness matters. Chat Completions and Completions have
  different responsibilities for templating.
- If token IDs are absent for a pure-embeds request, output prompt token IDs may
  be placeholder zeros in some response paths.

## Suggested Experiment Order

1. Start with a normal text-only model and reproduce the existing
   `examples/features/prompt_embed` flow.
2. Try a full-prompt embedding request through Completions.
3. Try Chat Completions with a text prefix/suffix and a small embedded content
   span to confirm mixed-mode behavior.
4. For VibeVoice, first verify that official vLLM can load and run the model
   with ordinary supported inputs.
5. Reproduce the model's own text WTE output for a tiny prompt outside vLLM and
   compare it against the embedding table loaded by vLLM or Transformers.
6. Add the audio encoder/projector output only after the text-only embedding
   path is byte- or tolerance-close.
7. Decide whether the macOS client should emit PyTorch-compatible serialized
   tensors or whether the server should accept a simpler wire format such as
   safetensors, NumPy `.npy`, or raw dtype/shape/data.

## Open Questions

- Does VibeVoice's official model architecture expose a clean decoder
  `inputs_embeds` path in vLLM's model executor implementation?
- Does VibeVoice require modality-specific position IDs, masks, or metadata that
  cannot be represented as plain prompt embedding rows?
- Is the client expected to send only audio embeddings, or full prompt
  embeddings including text WTE?
- Can MLX reproduce the exact audio encoder/projector numerics closely enough
  for generation quality?
- Is a PyTorch serialization dependency acceptable on the macOS client, or
  should vLLM grow a non-PyTorch tensor payload format for this experiment?
