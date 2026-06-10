#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "transformers @ git+https://github.com/huggingface/transformers.git@cbb65a4815d44f1d8b8ff7f51cca24ce491fc09e",
#   "huggingface-hub",
#   "librosa",
#   "numpy",
#   "requests",
#   "safetensors",
#   "scipy",
#   "torch",
# ]
# ///
"""Minimal VibeVoice ASR client for vLLM mixed prompt embeddings.

This script is the small, copyable runtime prototype for the split client-server
path:

1. Load only the HF-keyed audio encoder/projector bundle.
2. Load and normalize a mono audio file.
3. Run the HF VibeVoice acoustic/semantic encoders and multimodal projector.
4. Send the projected audio rows to vLLM Chat Completions as a
   ``prompt_embeds`` content part, while the ASR instruction remains normal
   text.

What this intentionally does *not* load:

* no Qwen decoder;
* no Qwen WTE / ``embed_tokens.weight``;
* no tokenizer or ASR processor files.

vLLM's mixed prompt-embedding renderer handles ordinary text server-side: it
tokenizes the Chat Completions text parts, runs the server-side WTE lookup for
those positions, and splices the supplied ``prompt_embeds`` rows into the prompt
at the content-part location.

Relevant upstream references:

* vLLM prompt embedding docs:
  https://docs.vllm.ai/en/latest/features/prompt_embeds.html
* vLLM prompt-embedding OpenAI client example:
  https://github.com/vllm-project/vllm/blob/main/examples/features/prompt_embed/prompt_embed_inference_with_openai_client.py
* vLLM Chat Completions prompt-embedding tests:
  https://github.com/vllm-project/vllm/blob/main/tests/entrypoints/openai/chat_completion/test_chat_completion_with_prompt_embeds.py
* HF VibeVoice acoustic feature extractor:
  https://github.com/huggingface/transformers/blob/main/src/transformers/models/vibevoice_acoustic_tokenizer/feature_extraction_vibevoice_acoustic_tokenizer.py

The code below favors clarity over framework architecture. Search for
``EXTENSION POINT`` comments when turning this into a reusable client module.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import requests
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from scipy.io import wavfile


TRANSFORMERS_GIT_REV = "cbb65a4815d44f1d8b8ff7f51cca24ce491fc09e"
DEFAULT_CHECKPOINT = "jkeisling/vibevoice-encoder-only"
AUDIO_ENCODER_FILE = "audio_encoder.safetensors"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that transcribes audio input into text output "
    "in JSON format."
)
DEFAULT_STOP = ["<|endoftext|>", "<|im_end|>"]


@dataclass(frozen=True)
class AudioRows:
    """Projected audio rows and the small amount of metadata needed for vLLM."""

    features: torch.Tensor
    duration_seconds: float
    sample_rate: int
    valid_samples: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Encode one audio file with a minimal HF-keyed VibeVoice audio "
            "bundle and POST it to vLLM as mixed Chat Completions prompt_embeds."
        )
    )
    parser.add_argument("audio", help="Audio file to transcribe.")
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help=(
            "Minimal HF-keyed audio bundle containing audio_encoder.safetensors "
            f"and config.json. Defaults to {DEFAULT_CHECKPOINT!r}; local paths "
            "and Hugging Face repo IDs are both accepted."
        ),
    )
    parser.add_argument("--url", default="http://localhost:8000", help="vLLM server URL.")
    parser.add_argument("--model", default="/models/vibevoice", help="Served model name.")
    parser.add_argument("--context-info", default=None, help="Optional hotwords/context text.")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--stop", action="append", default=None)
    parser.add_argument("--timeout", type=float, default=12000.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for the audio encoder/projector.",
    )
    parser.add_argument("--dtype", default="bfloat16", choices=("float32", "float16", "bfloat16"))
    parser.add_argument(
        "--save-audio-rows",
        default=None,
        help="Optional .pt path for saving the encoded audio rows before posting.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Encode audio rows and print request metadata without POSTing to vLLM.",
    )
    parser.add_argument(
        "--transformers-src",
        default=None,
        help="Optional local Transformers src checkout to prepend to sys.path.",
    )
    return parser.parse_args()


def import_hf_classes(transformers_src: str | None) -> dict[str, Any]:
    """Import only the HF classes needed for audio-side runtime work."""

    if transformers_src:
        sys.path.insert(0, str(Path(transformers_src).expanduser().resolve()))

    from transformers.models.vibevoice_acoustic_tokenizer import (
        VibeVoiceAcousticTokenizerEncoderModel,
        VibeVoiceAcousticTokenizerFeatureExtractor,
    )
    from transformers.models.vibevoice_asr import VibeVoiceAsrConfig
    from transformers.models.vibevoice_asr.modeling_vibevoice_asr import (
        VibeVoiceAsrMultiModalProjector,
    )

    import transformers

    print(f"transformers={transformers.__version__} from {transformers.__file__}")
    print(f"expected_git_rev={TRANSFORMERS_GIT_REV}")
    return {
        "VibeVoiceAcousticTokenizerEncoderModel": VibeVoiceAcousticTokenizerEncoderModel,
        "VibeVoiceAcousticTokenizerFeatureExtractor": VibeVoiceAcousticTokenizerFeatureExtractor,
        "VibeVoiceAsrConfig": VibeVoiceAsrConfig,
        "VibeVoiceAsrMultiModalProjector": VibeVoiceAsrMultiModalProjector,
    }


def load_audio_file(path: Path, target_sr: int) -> np.ndarray:
    """Load one audio file as mono float32 at the model sample rate.

    EXTENSION POINT: this is intentionally single-file and eager. A production
    client might accept file-like objects, stream chunks, resample lazily, or
    delegate decoding/resampling to platform-native audio APIs.
    """

    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        data = data.astype(np.float32) / max(abs(info.min), info.max)
    else:
        data = data.astype(np.float32)
    if sr != target_sr:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
    return data.astype(np.float32)


def resolve_audio_weights(checkpoint: str) -> Path:
    """Resolve ``audio_encoder.safetensors`` from a local bundle or HF Hub repo.

    ``transformers`` handles ``config.json`` idiomatically through
    ``VibeVoiceAsrConfig.from_pretrained(checkpoint)``. The split audio weights
    are a sidecar safetensors file, so for Hub IDs we resolve that file with
    ``huggingface_hub.hf_hub_download``.
    """

    local = Path(checkpoint).expanduser()
    if local.exists():
        return local / AUDIO_ENCODER_FILE
    return Path(hf_hub_download(repo_id=checkpoint, filename=AUDIO_ENCODER_FILE))


def load_prefixed_state(weights_path: Path, prefix: str) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key.startswith(prefix):
                state[key.removeprefix(prefix)] = handle.get_tensor(key)
    if not state:
        raise KeyError(f"{weights_path} has no tensors under prefix {prefix!r}")
    return state


def strict_load(module: torch.nn.Module, state: dict[str, torch.Tensor], label: str) -> None:
    expected = module.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    wrong_shape = sorted(
        key for key in set(expected) & set(state) if tuple(expected[key].shape) != tuple(state[key].shape)
    )
    if missing or unexpected or wrong_shape:
        raise RuntimeError(
            f"{label} state mismatch: "
            f"missing={missing[:8]} unexpected={unexpected[:8]} wrong_shape={wrong_shape[:8]}"
        )
    module.load_state_dict(state, strict=True)
    print(f"{label}: strict load ok ({len(state)} tensors)")


def build_audio_modules(
    classes: dict[str, Any],
    checkpoint: str,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Any, torch.nn.Module, torch.nn.Module, torch.nn.Module]:
    """Instantiate the HF audio encoder/projector modules and strict-load weights.

    EXTENSION POINT: module construction is eager for readability. A service
    client should wrap this in an object that loads once and reuses the modules
    across requests.
    """

    config = classes["VibeVoiceAsrConfig"].from_pretrained(checkpoint)
    acoustic = classes["VibeVoiceAcousticTokenizerEncoderModel"](
        config.acoustic_tokenizer_encoder_config
    ).to(device=device, dtype=dtype).eval()
    semantic = classes["VibeVoiceAcousticTokenizerEncoderModel"](
        config.semantic_tokenizer_encoder_config
    ).to(device=device, dtype=dtype).eval()
    projector = classes["VibeVoiceAsrMultiModalProjector"](config).to(device=device, dtype=dtype).eval()

    weights_path = resolve_audio_weights(checkpoint)
    acoustic_state = load_prefixed_state(weights_path, "model.acoustic_tokenizer_encoder.")
    semantic_state = load_prefixed_state(weights_path, "model.semantic_tokenizer_encoder.")
    projector_state = load_prefixed_state(weights_path, "model.multi_modal_projector.")
    strict_load(acoustic, acoustic_state, "acoustic_tokenizer_encoder")
    strict_load(semantic, semantic_state, "semantic_tokenizer_encoder")
    strict_load(projector, projector_state, "multi_modal_projector")

    return config, acoustic, semantic, projector


def preprocess_audio(
    feature_extractor_cls: Any,
    audio: np.ndarray,
    *,
    sample_rate: int,
    hop_length: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Normalize/pad audio with HF's acoustic-tokenizer feature extractor.

    This replaces ``AutoProcessor.apply_transcription_request`` for the mixed
    path. The audio-only client only needs ``input_values`` and ``padding_mask``;
    text token IDs and audio placeholder expansion are now server-side vLLM work.

    EXTENSION POINT: for bsz > 1, pass a list of arrays here and keep per-item
    row counts after projection. The current script intentionally supports one
    file so the wire format and correctness are easy to inspect.
    """

    extractor = feature_extractor_cls(sampling_rate=sample_rate)
    batch = extractor(
        audio,
        sampling_rate=sample_rate,
        pad_to_multiple_of=hop_length,
        return_attention_mask=True,
        return_tensors="pt",
    )
    valid_samples = int(batch["padding_mask"].sum(dim=-1)[0].item())
    return batch["input_values"], batch["padding_mask"], valid_samples


def encode_audio_features(
    *,
    acoustic: torch.nn.Module,
    semantic: torch.nn.Module,
    projector: torch.nn.Module,
    input_values: torch.Tensor,
    padding_mask: torch.Tensor,
    chunk_size: int,
    hop_length: int,
    vae_std: float,
    seed: int,
) -> torch.Tensor:
    """Mirror HF ``VibeVoiceAsrModel.get_audio_features`` without the decoder.

    EXTENSION POINT: this processes the whole audio prefix before decode. If you
    add VAD splitting or long-audio chunk orchestration, keep each segment's
    projected rows ordered exactly where its prompt text places the corresponding
    ``prompt_embeds`` content part.
    """

    torch.manual_seed(seed)

    acoustic_cache = None
    semantic_cache = None
    acoustic_latents = []
    semantic_latents = []
    for chunk in torch.split(input_values, chunk_size, dim=-1):
        acoustic_output = acoustic(chunk, padding_cache=acoustic_cache, use_cache=True)
        semantic_output = semantic(chunk, padding_cache=semantic_cache, use_cache=True)
        acoustic_latents.append(acoustic_output.latents)
        semantic_latents.append(semantic_output.latents)
        acoustic_cache = acoustic_output.padding_cache
        semantic_cache = semantic_output.padding_cache

    acoustic_latents = torch.cat(acoustic_latents, dim=1)
    semantic_latents = torch.cat(semantic_latents, dim=1)

    noise_std = vae_std * torch.randn(
        acoustic_latents.shape[0],
        device=acoustic_latents.device,
        dtype=acoustic_latents.dtype,
    )
    acoustic_latents = acoustic_latents + noise_std[:, None, None] * torch.randn_like(acoustic_latents)
    combined = projector(acoustic_latents, semantic_latents)

    num_audio_tokens = torch.ceil(padding_mask.sum(dim=-1) / hop_length).to(torch.int64)
    feature_mask = torch.arange(num_audio_tokens.max(), device=combined.device) < num_audio_tokens[:, None]
    return combined[feature_mask]


def encode_audio_rows(
    *,
    classes: dict[str, Any],
    checkpoint: str,
    audio_path: Path,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> AudioRows:
    """Load modules, preprocess one audio file, and return projected audio rows."""

    config, acoustic, semantic, projector = build_audio_modules(
        classes,
        checkpoint,
        dtype=dtype,
        device=device,
    )
    sample_rate = int(getattr(config, "sample_rate", 24000))
    hop_length = int(config.acoustic_tokenizer_encoder_config.hop_length)
    audio = load_audio_file(audio_path, sample_rate)
    input_values, padding_mask, valid_samples = preprocess_audio(
        classes["VibeVoiceAcousticTokenizerFeatureExtractor"],
        audio,
        sample_rate=sample_rate,
        hop_length=hop_length,
    )
    input_values = input_values.to(device=device, dtype=dtype)
    padding_mask = padding_mask.to(device=device)

    with torch.inference_mode():
        features = encode_audio_features(
            acoustic=acoustic,
            semantic=semantic,
            projector=projector,
            input_values=input_values,
            padding_mask=padding_mask,
            chunk_size=config.acoustic_tokenizer_chunk_size,
            hop_length=hop_length,
            vae_std=config.acoustic_tokenizer_encoder_config.vae_std,
            seed=seed,
        )

    expected_rows = int(np.ceil(valid_samples / hop_length))
    if int(features.shape[0]) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} audio rows, got {int(features.shape[0])}")

    return AudioRows(
        features=features.detach().cpu().float(),
        duration_seconds=valid_samples / sample_rate,
        sample_rate=sample_rate,
        valid_samples=valid_samples,
    )


def tensor_to_base64(tensor: torch.Tensor) -> str:
    """Serialize a 2D tensor in vLLM's current HTTP ``prompt_embeds`` format.

    EXTENSION POINT: a non-Python client may prefer a server shim accepting
    safetensors, npy, raw bytes, or another schema. vLLM's OpenAI-compatible
    server currently expects base64 of ``torch.save(tensor)`` for prompt embeds.
    """

    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def build_user_text(duration_seconds: float, context_info: str | None) -> str:
    requested_keys = "Start time, End time, Speaker ID, Content"
    duration = f"{duration_seconds:.2f}"
    if context_info:
        return (
            f"This is a {duration} seconds audio, with extra info: {context_info}\n\n"
            f"Please transcribe it with these keys: {requested_keys}"
        )
    return f"This is a {duration} seconds audio, please transcribe it with these keys: {requested_keys}"


def build_chat_payload(
    *,
    model: str,
    system_prompt: str,
    user_text: str,
    audio_rows: torch.Tensor,
    max_tokens: int,
    temperature: float,
    top_p: float,
    stop: list[str],
) -> dict[str, Any]:
    """Build the mixed Chat Completions request.

    EXTENSION POINT: for multiple audio segments, repeat or interleave
    ``prompt_embeds`` content parts with text parts in the order the prompt
    should see them. vLLM preserves the relative content-part positions.
    """

    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "prompt_embeds", "data": tensor_to_base64(audio_rows)},
                    {"type": "text", "text": "\n" + user_text},
                ],
            },
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stop": stop,
    }


def response_text(response_json: dict[str, Any]) -> str:
    choices = response_json.get("choices") or []
    if not choices:
        return ""
    return (choices[0].get("message") or {}).get("content") or ""


def post_to_vllm(url: str, payload: dict[str, Any], timeout: float) -> tuple[dict[str, Any], float]:
    started = time.time()
    response = requests.post(url.rstrip("/") + "/v1/chat/completions", json=payload, timeout=timeout)
    elapsed = time.time() - started
    if response.status_code != 200:
        print(f"request failed: HTTP {response.status_code}")
        print(response.text)
        raise SystemExit(1)
    return response.json(), elapsed


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    stop = args.stop if args.stop is not None else DEFAULT_STOP

    classes = import_hf_classes(args.transformers_src)
    rows = encode_audio_rows(
        classes=classes,
        checkpoint=checkpoint,
        audio_path=Path(args.audio).expanduser(),
        dtype=dtype,
        device=device,
        seed=args.seed,
    )
    user_text = build_user_text(rows.duration_seconds, args.context_info)

    print(
        "encoded_audio_rows "
        f"rows={tuple(rows.features.shape)} "
        f"duration_seconds={rows.duration_seconds:.2f} "
        f"valid_samples={rows.valid_samples}"
    )
    print(f"user_text={user_text!r}")

    if args.save_audio_rows:
        save_path = Path(args.save_audio_rows).expanduser()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "audio_features": rows.features,
                "duration_seconds": rows.duration_seconds,
                "sample_rate": rows.sample_rate,
                "valid_samples": rows.valid_samples,
                "user_text": user_text,
            },
            save_path,
        )
        print(f"saved_audio_rows={save_path}")

    payload = build_chat_payload(
        model=args.model,
        system_prompt=args.system_prompt,
        user_text=user_text,
        audio_rows=rows.features,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stop=stop,
    )

    if args.dry_run:
        preview = dict(payload)
        preview["messages"] = [
            payload["messages"][0],
            {
                "role": "user",
                "content": [
                    {"type": "prompt_embeds", "data": f"<base64 torch tensor: {tuple(rows.features.shape)}>"},
                    payload["messages"][1]["content"][1],
                ],
            },
        ]
        print(json.dumps(preview, indent=2, ensure_ascii=False))
        return 0

    data, elapsed = post_to_vllm(args.url, payload, args.timeout)
    text = response_text(data)
    print("\n--- output ---")
    print(text)
    print("--- end output ---")
    print(f"elapsed_seconds={elapsed:.2f}")
    usage = data.get("usage")
    if usage:
        print("usage=" + json.dumps(usage, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
