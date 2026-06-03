# VibeVoice ASR Data Flow Trace

## 1. Prompt Format (Raw Level)

**Chat template** (`vibevoice/modular/modular_vibevoice_text_tokenizer.py:262`):
```jinja2
{% for message in messages %}{{'▌<' + message['role'] + '\n' + message['content'] + '▌></s>\n'}}{% endfor %}{% if add_generation_prompt %}{{ '▌<assistant\n' }}{% endif %}
```

**System prompt** (`vibevoice/processor/vibevoice_asr_processor.py:27`):
```
You are a helpful assistant that transcribes audio input into text output in JSON format.
```

**User message content** (`vibevoice/processor/vibevoice_asr_processor.py:360-368`):

For a 5.0s audio at 24kHz, `vae_tok_len = ceil(5.0 * 24000 / 3200) = 38`:

```
<|speech_start|><|speech_pad|><|speech_pad|>...x36...<|speech_pad|><|speech_end|>
This is a 5.00 seconds audio, please transcribe it with these keys: Start time, End time, Speaker ID, Content
```

With `context_info="John Smith\nMachine Learning"` (hotwords injected inline):

```
<|speech_start|><|speech_pad|>...x36...<|speech_pad|><|speech_end|>
This is a 5.00 seconds audio, with extra info: John Smith
Machine Learning

Please transcribe it with these keys: Start time, End time, Speaker ID, Content
```

**Fully expanded raw prompt** (after chat template + `add_generation_prompt=True`):

```
▌<system
You are a helpful assistant that transcribes audio input into text output in JSON format.
▌></s>
▌<user
<|speech_start|><|speech_pad|><|speech_pad|>...<|speech_pad|><|speech_end|>
This is a 5.00 seconds audio, please transcribe it with these keys: Start time, End time, Speaker ID, Content
▌></s>
▌<assistant
```

The prompt ends with `▌<assistant\n` — no closing tag — so the LLM generates directly from there.

## 2. Control Tokens (Raw Level)

Three speech-specific special tokens registered in `VibeVoiceASRTextTokenizerFast` (`modular_vibevoice_text_tokenizer.py:264-283`), appended to the base Qwen2 vocab:

| Token string | Property name | Purpose |
|---|---|---|
| `<\|speech_start\|>` | `tokenizer.speech_start_id` | Marks beginning of audio embedding region |
| `<\|speech_pad\|>` | `tokenizer.speech_pad_id` | Placeholder for each encoded audio frame (N copies) |
| `<\|speech_end\|>` | `tokenizer.speech_end_id` | Marks end of audio embedding region |

The speech region in the token sequence is: `[speech_start_id, speech_pad_id × vae_tok_len, speech_end_id]`.

**Token count formula:**
```
vae_tok_len = ceil(audio_samples / speech_tok_compress_ratio)
            = ceil(duration_seconds * 24000 / 3200)
```

Where `speech_tok_compress_ratio = 3200` is the product of encoder stride ratios `[8, 5, 5, 4, 2, 2]` (`configuration_vibevoice.py:55`).

## 3. Audio Encoder Path

Two parallel VAE encoders process the raw waveform (`modeling_vibevoice_asr.py:208-339`):

### Acoustic Tokenizer (dim=64)
```
Raw audio [bsz, samples] @ 24kHz
  → unsqueeze(1) → [bsz, 1, samples]
  → acoustic_tokenizer.encode() → VAE latent mean [bsz, 64, vae_tok_len]
  → sample from Gaussian (std_dist_type='gaussian', fix_std=0.5)
  → audio_tokens [bsz, 64, vae_tok_len]
  → acoustic_connector: Linear(64→hidden) → RMSNorm(hidden) → Linear(hidden→hidden)
  → acoustic_features [bsz, hidden_dim, vae_tok_len]
```

### Semantic Tokenizer (dim=128)
```
Raw audio [bsz, samples] @ 24kHz
  → unsqueeze(1) → [bsz, 1, samples]
  → semantic_tokenizer.encode() → VAE latent mean [bsz, 128, vae_tok_len]
  → (no sampling, std_dist_type='none', fix_std=0)
  → semantic_tokens [bsz, 128, vae_tok_len]
  → semantic_connector: Linear(128→hidden) → RMSNorm(hidden) → Linear(hidden→hidden)
  → semantic_features [bsz, hidden_dim, vae_tok_len]
```

### SpeechConnector (`modeling_vibevoice.py:59-70`)
```python
class SpeechConnector(nn.Module):
    def __init__(self, input_dim, output_dim):
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.norm = LlamaRMSNorm(output_dim, eps=1e-6)
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, features):
        x = self.fc1(features)
        x = self.norm(x)
        x = self.fc2(x)
        return x
```

### Combined Features (`modeling_vibevoice_asr.py:333-337`)
```python
combined_features = acoustic_features[speech_masks] + semantic_features[speech_masks]
# Shape: [vae_tok_len, hidden_dim]  (flattened via boolean mask indexing)
```

### Long Audio Streaming (>60s)

For audio longer than the `streaming_segment_duration` (default 60s), segments are processed independently with a `VibeVoiceTokenizerStreamingCache` to maintain encoder state continuity, then concatenated before sampling (`modeling_vibevoice_asr.py:264-331`).

## 4. Embedding Fusion → `[bsz, input_seqlen, hidden_dim]`

### Step A — Text Tokenization (`vibevoice_asr_processor.py:346-386`)

The full prompt is built as a token list:
```
system_tokens + user_tokens (containing speech_start, speech_pad×N, speech_end, and text)
```

Left-padded in batch to `max_length` → `input_ids: [bsz, input_seqlen]`

An `acoustic_input_mask: [bsz, input_seqlen]` boolean mask is also built, where `True` only at positions where the token equals `speech_pad_id` (`vibevoice_asr_processor.py:379`).

### Step B — Word Token Embedding (WTE) (`modeling_vibevoice_asr.py:371`)

```python
inputs_embeds = self.get_input_embeddings()(input_ids)
# Qwen2 embed_tokens: [bsz, input_seqlen, vocab_size] → [bsz, input_seqlen, hidden_dim]
# e.g. [1, 4096, 896] for Qwen2.5-1.5B (hidden=896)
```

### Step C — Speech Embedding Scatter (`modeling_vibevoice_asr.py:374-383`)

```python
speech_features = self.encode_speech(speech_tensors, speech_masks)  # [vae_tok_len, hidden_dim]
inputs_embeds = inputs_embeds.clone()
inputs_embeds[acoustic_input_mask] = speech_features  # scatter assignment
```

The N `speech_pad` token positions have their learned WTE embeddings **replaced** by the corresponding audio VAE frame embeddings. The `speech_start` and `speech_end` tokens retain their original learned embeddings from the WTE lookup table.

### Final Tensor Into Transformer Trunk

```
inputs_embeds: [bsz, input_seqlen, hidden_dim]
example:       [1,   4096,       896]    (Qwen2.5-1.5B)
example:       [1,   4096,       3584]   (Qwen2.5-7B)
```

Internal layout of `input_seqlen` positions:
```
[pad...pad | ▌<system\n...▌></s>\n | ▌<user\n | <|speech_start|> | audio_emb_1 | audio_emb_2 | ... | audio_emb_N | <|speech_end|> | \nThis is a...\n▌</s>\n | ▌<assistant\n]
 ^left pad^    ^system message^        ^user message with audio region^                    ^suffix text^          ^generation prompt^
```

## 5. LLM Output Format

The model generates a JSON array of transcription segments. The system prompt instructs JSON output, and the user prompt specifies the keys: `Start time`, `End time`, `Speaker ID`, `Content`.

**Raw LLM output:**
```json
[
  {
    "Start time": 0.12,
    "End time": 3.45,
    "Speaker ID": "SPEAKER_00",
    "Content": "Hello, how are you?"
  },
  {
    "Start time": 3.80,
    "End time": 6.20,
    "Speaker ID": "SPEAKER_01",
    "Content": "I'm doing great, thanks!"
  }
]
```

**Post-processing** (`vibevoice_asr_processor.py:490-562`) normalizes keys:

| Raw key | Normalized key |
|---|---|
| `Start time` / `Start` | `start_time` |
| `End time` / `End` | `end_time` |
| `Speaker ID` / `Speaker` | `speaker_id` |
| `Content` | `text` |

Handles both bare JSON and markdown code blocks (```json ... ```).

## 6. Generation Flow (`prepare_inputs_for_generation`)

Speech tensors are only forwarded on the **first** generation pass (`cache_position[0] == 0`). Subsequent autoregressive steps receive `None` for all speech inputs, relying on `past_key_values` cache (`modeling_vibevoice_asr.py:491-508`). This follows the Qwen2-VL pattern.

## 7. Architecture Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT                                       │
│  Raw audio [bsz, samples] @ 24kHz + Prompt template + context_info │
└──────────────┬──────────────────────────────────────┬──────────────┘
               │                                      │
               ▼                                      ▼
┌─────────────────────────┐    ┌──────────────────────────────────┐
│   AUDIO ENCODERS        │    │   TEXT TOKENIZER (Qwen2)         │
│                         │    │                                  │
│  acoustic_tokenizer     │    │  Chat template → token IDs       │
│  (VAE, dim=64, sampled) │    │  [bsz, input_seqlen]             │
│  ↓ connector            │    │                                  │
│  semantic_tokenizer     │    │  WTE lookup                      │
│  (VAE, dim=128, mean)   │    │  → [bsz, input_seqlen, hidden]  │
│  ↓ connector            │    │                                  │
│  + (element-wise sum)   │    │  acoustic_input_mask             │
│                         │    │  (True at speech_pad positions)  │
│  → [N, hidden_dim]      │    └──────────────┬───────────────────┘
└──────────────┬──────────┘                   │
               │                              │
               ▼                              │
    scatter into WTE embeddings              │
               │                              │
               ▼                              │
    inputs_embeds [bsz, input_seqlen, hidden]│
               │                              │
               ▼                              │
┌──────────────────────────────────────────────┤
│   QWEN2 TRANSFORMER TRUNK                   │
│   (self-attention layers + FFN)             │
│   → last_hidden_state                       │
│   → lm_head → logits → JSON output          │
└─────────────────────────────────────────────┘
```
