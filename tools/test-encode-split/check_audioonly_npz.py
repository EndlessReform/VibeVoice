#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "transformers @ git+https://github.com/huggingface/transformers.git@cbb65a4815d44f1d8b8ff7f51cca24ce491fc09e",
#   "librosa",
#   "numpy",
#   "requests",
#   "safetensors",
#   "scipy",
#   "torch",
# ]
# ///
"""Check split ASR audio weights against upstream Transformers.

This is intentionally self-contained: run it with `uv run --script` and it will
use the dependencies above instead of the repo's VibeVoice package or venv.

The main runtime contract tested here is the vLLM mixed prompt-embedding path:
the client sends normal Chat Completions strings for text and one
``prompt_embeds`` content part containing only projected audio rows. vLLM then
renders/tokenizes the text on the server, looks up text rows with the server-side
word embedding table, and splices the supplied audio rows into the prompt.

Useful upstream references:

* vLLM prompt embedding feature docs:
  https://docs.vllm.ai/en/latest/features/prompt_embeds.html
* vLLM example client for Completions vs Chat Completions prompt embeddings:
  https://github.com/vllm-project/vllm/blob/main/examples/features/prompt_embed/prompt_embed_inference_with_openai_client.py
* vLLM Chat Completions prompt-embedding tests:
  https://github.com/vllm-project/vllm/blob/main/tests/entrypoints/openai/chat_completion/test_chat_completion_with_prompt_embeds.py
* vLLM renderer helper that builds the internal ``prompt_is_token_ids`` mask:
  https://github.com/vllm-project/vllm/blob/main/vllm/renderers/hf.py
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import requests
import torch
from safetensors import safe_open
from scipy.io import wavfile


TRANSFORMERS_GIT_REV = "cbb65a4815d44f1d8b8ff7f51cca24ce491fc09e"
AUDIO_ENCODER_FILE = "audio_encoder.safetensors"
WTE_KEY = "model.language_model.embed_tokens.weight"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant that transcribes audio input into text output "
    "in JSON format."
)
DEFAULT_STOP = ["<|endoftext|>", "<|im_end|>"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load HF-compatible split audio weights with upstream Transformers, "
            "write an audio-row NPZ, and optionally validate the mixed "
            "prompt_embeds Chat Completions path against vLLM."
        )
    )
    parser.add_argument("--checkpoint", default="out/audioonly-hf-checkpoint")
    parser.add_argument(
        "--processor-path",
        default=None,
        help=(
            "Optional HF processor/tokenizer directory. Defaults to --checkpoint. "
            "Use this when validating a minimal audio-only checkpoint that does "
            "not include tokenizer/processor files."
        ),
    )
    parser.add_argument("--audio", default="flight.wav")
    parser.add_argument("--reference-npz", default="out/flight_prefix_fixture_check.npz")
    parser.add_argument("--output-npz", default="out/test-encode-split/audioonly_upstream_check.npz")
    parser.add_argument(
        "--transformers-src",
        default=None,
        help=(
            "Optional local Transformers src checkout. By default the uv-pinned "
            "dependency from this script metadata is used."
        ),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--context-info", default=None)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument(
        "--compare-key",
        action="append",
        dest="compare_keys",
        default=["input_ids", "attention_mask", "acoustic_input_mask", "speech_masks"],
        help=(
            "NPZ key to compare. Repeat to override defaults after passing "
            "`--no-default-compare-keys`."
        ),
    )
    parser.add_argument("--no-default-compare-keys", action="store_true")
    parser.add_argument("--compare-audio-features", action="store_true")
    parser.add_argument(
        "--build-full-inputs-embeds",
        action="store_true",
        help=(
            "Also load the decoder WTE tensor and build the legacy full "
            "inputs_embeds tensor. The mixed vLLM path does not need this."
        ),
    )
    parser.add_argument(
        "--post-mixed-vllm",
        action="store_true",
        help=(
            "POST the audio-only prompt_embeds rows to vLLM Chat Completions. "
            "Requires vLLM with --enable-prompt-embeds and Chat Completions "
            "prompt_embeds support, e.g. v0.20.2+ / v0.22.1."
        ),
    )
    parser.add_argument(
        "--post-full-vllm",
        action="store_true",
        help=(
            "POST the legacy full inputs_embeds tensor to vLLM Completions as "
            "a baseline. Implies --build-full-inputs-embeds."
        ),
    )
    parser.add_argument("--url", default="http://localhost:8000", help="vLLM server URL.")
    parser.add_argument(
        "--model",
        default="/models/vibevoice",
        help="Served model name for vLLM requests.",
    )
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--stop",
        action="append",
        default=None,
        help=(
            "Stop string for vLLM requests. Repeat for multiple stops. "
            "Defaults to <|endoftext|> and <|im_end|>."
        ),
    )
    parser.add_argument("--timeout", type=float, default=12000.0)
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt used for mixed Chat Completions vLLM validation.",
    )
    parser.add_argument(
        "--ignore-final-audio-token",
        action="store_true",
        help=(
            "When comparing audio_features to the old repo fixture, compare all "
            "but the final partial streamed token. HF main does not expose the "
            "original tokenizer's is_final_chunk right-padding path."
        ),
    )
    return parser.parse_args()


def import_upstream(transformers_src: str | None):
    if transformers_src:
        src = Path(transformers_src).expanduser().resolve()
    else:
        src = None
    if src is not None and src.exists():
        sys.path.insert(0, str(src))

    from transformers import AutoProcessor
    from transformers.models.vibevoice_acoustic_tokenizer import (
        VibeVoiceAcousticTokenizerEncoderModel,
    )
    from transformers.models.vibevoice_asr import VibeVoiceAsrConfig
    from transformers.models.vibevoice_asr.modeling_vibevoice_asr import VibeVoiceAsrMultiModalProjector

    import transformers

    print(f"transformers={transformers.__version__} from {transformers.__file__}")
    print(f"expected_git_rev={TRANSFORMERS_GIT_REV}")

    return {
        "AutoProcessor": AutoProcessor,
        "VibeVoiceAcousticTokenizerEncoderModel": VibeVoiceAcousticTokenizerEncoderModel,
        "VibeVoiceAsrConfig": VibeVoiceAsrConfig,
        "VibeVoiceAsrMultiModalProjector": VibeVoiceAsrMultiModalProjector,
    }


def load_audio(path: Path, target_sr: int = 24000) -> np.ndarray:
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


def load_prefixed_state(weights_path: Path, prefix: str) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key.startswith(prefix):
                state[key.removeprefix(prefix)] = handle.get_tensor(key)
    if not state:
        raise KeyError(f"{weights_path} has no tensors under prefix {prefix!r}")
    return state


def load_hf_audio_state(
    weights_path: Path,
    *,
    include_wte: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor | None]:
    """Load the HF audio split tensors.

    ``include_wte=False`` is the important default for the mixed-embedding
    validation path. vLLM's Chat Completions ``prompt_embeds`` content part lets
    a client send only non-token hidden states, while vLLM keeps ordinary text
    tokens as token IDs and performs the WTE lookup on the server. That is the
    behavior documented in the vLLM prompt-embedding feature page:
    https://docs.vllm.ai/en/latest/features/prompt_embeds.html

    Set ``include_wte=True`` only when comparing against the older full
    ``inputs_embeds`` path, where the client builds every text row locally.
    """

    wte: torch.Tensor | None = None
    if include_wte:
        with safe_open(weights_path, framework="pt", device="cpu") as handle:
            if WTE_KEY in handle.keys():
                wte = handle.get_tensor(WTE_KEY)
        if wte is None:
            raise KeyError(f"{weights_path} is missing {WTE_KEY}")

    acoustic = load_prefixed_state(weights_path, "model.acoustic_tokenizer_encoder.")
    semantic = load_prefixed_state(weights_path, "model.semantic_tokenizer_encoder.")
    projector = load_prefixed_state(weights_path, "model.multi_modal_projector.")
    wte_msg = f" wte_shape={tuple(wte.shape)}" if wte is not None else " wte=skipped"
    print(
        "loaded_hf_keys "
        f"acoustic={len(acoustic)} semantic={len(semantic)} "
        f"projector={len(projector)}{wte_msg}"
    )
    return acoustic, semantic, projector, wte


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


def build_modules(classes: dict[str, Any], checkpoint: Path, dtype: torch.dtype, device: torch.device):
    asr_config = classes["VibeVoiceAsrConfig"].from_pretrained(checkpoint)
    acoustic = classes["VibeVoiceAcousticTokenizerEncoderModel"](
        asr_config.acoustic_tokenizer_encoder_config
    ).to(device=device, dtype=dtype).eval()
    semantic = classes["VibeVoiceAcousticTokenizerEncoderModel"](
        asr_config.semantic_tokenizer_encoder_config
    ).to(device=device, dtype=dtype).eval()
    projector = classes["VibeVoiceAsrMultiModalProjector"](asr_config).to(device=device, dtype=dtype).eval()
    return acoustic, semantic, projector


def encode_audio_features(
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


def tensor_to_base64(tensor: torch.Tensor) -> str:
    """Serialize a 2D tensor in vLLM's HTTP ``prompt_embeds`` wire format.

    vLLM's OpenAI-compatible prompt-embedding endpoints currently expect a
    base64-encoded ``torch.save`` payload. The feature docs describe this as a
    base64 encoded torch tensor, and the official example uses vLLM's
    ``tensor2base64`` helper:
    https://github.com/vllm-project/vllm/blob/main/examples/features/prompt_embed/prompt_embed_inference_with_openai_client.py

    This local helper avoids importing vLLM into the split-audio validation
    script. It intentionally accepts a single sequence tensor of shape
    ``(num_tokens, hidden_size)`` because that is what Chat Completions content
    parts consume.
    """

    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def build_asr_user_text(duration_seconds: float, context_info: str | None) -> str:
    """Build the text content that accompanies audio rows in mixed vLLM mode.

    The runtime split we want is deliberately simple:

    * the client sends projected audio rows as a ``prompt_embeds`` content part;
    * the client sends this ordinary text as a Chat Completions text part;
    * vLLM tokenizes and embeds the text with the server-side Qwen WTE.

    This mirrors the ASR template wording without requiring the audio-only
    client to load tokenizer files or a 1 GiB embedding table.
    """

    duration = f"{duration_seconds:.2f}"
    requested_keys = "Start time, End time, Speaker ID, Content"
    if context_info:
        return (
            f"This is a {duration} seconds audio, with extra info: {context_info}\n\n"
            f"Please transcribe it with these keys: {requested_keys}"
        )
    return f"This is a {duration} seconds audio, please transcribe it with these keys: {requested_keys}"


def build_mixed_chat_payload(
    *,
    model: str,
    system_prompt: str,
    user_text: str,
    audio_features: torch.Tensor,
    max_tokens: int,
    temperature: float,
    top_p: float,
    stop: list[str],
) -> dict[str, Any]:
    """Build a vLLM Chat Completions request for audio-only prompt embeddings.

    vLLM v0.20.2+ can accept ``prompt_embeds`` as Chat Completions content
    parts. The renderer expands an internal placeholder span to the length of
    the supplied tensor, builds a full prompt-length embedding buffer, and marks
    normal text positions as token IDs so the server performs WTE lookup. The
    public tests exercise this exact shape here:
    https://github.com/vllm-project/vllm/blob/main/tests/entrypoints/openai/chat_completion/test_chat_completion_with_prompt_embeds.py

    For VibeVoice ASR, ``audio_features`` is not a full prompt. It is only the
    projected audio rows that replace the repeated audio placeholder positions.
    The surrounding ASR instruction remains text in the same Chat Completions
    message.
    """

    encoded_audio = tensor_to_base64(audio_features.detach().cpu().float())
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "prompt_embeds", "data": encoded_audio},
                    {"type": "text", "text": "\n" + user_text},
                ],
            },
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stop": stop,
    }


def build_full_completions_payload(
    *,
    model: str,
    inputs_embeds: torch.Tensor,
    max_tokens: int,
    temperature: float,
    top_p: float,
    stop: list[str],
) -> dict[str, Any]:
    """Build the legacy full-prompt vLLM Completions request.

    The Completions endpoint accepts ``prompt_embeds``, but it does not apply a
    chat template or perform mixed text-token/WTE handling for the caller. vLLM
    documents this distinction explicitly:
    https://docs.vllm.ai/en/latest/features/prompt_embeds.html#completions-api

    Use this only as a regression baseline. It requires ``inputs_embeds`` to
    already include both normal text WTE rows and projected audio rows.
    """

    return {
        "model": model,
        "prompt": None,
        "prompt_embeds": tensor_to_base64(inputs_embeds.detach().cpu().float()),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stop": stop,
    }


def response_text(response_json: dict[str, Any]) -> str:
    choices = response_json.get("choices") or []
    if not choices:
        return ""
    choice = choices[0]
    if "text" in choice:
        return choice.get("text") or ""
    return (choice.get("message") or {}).get("content") or ""


def post_json(url: str, payload: dict[str, Any], timeout: float) -> tuple[dict[str, Any], float]:
    started = time.time()
    response = requests.post(url, json=payload, timeout=timeout)
    elapsed = time.time() - started
    if response.status_code != 200:
        print(f"request failed: HTTP {response.status_code}")
        print(response.text)
        raise SystemExit(1)
    return response.json(), elapsed


def compare_arrays(key: str, got: np.ndarray, ref: np.ndarray, atol: float, rtol: float) -> bool:
    print(f"{key}: got_shape={got.shape} ref_shape={ref.shape} got_dtype={got.dtype} ref_dtype={ref.dtype}")
    if got.shape != ref.shape:
        print(f"{key}: shape_match=False")
        return False
    if got.dtype.kind in "biu" and ref.dtype.kind in "biu":
        ok = bool(np.array_equal(got, ref))
        print(f"{key}: exact={ok}")
        return ok
    diff = np.abs(got.astype(np.float32) - ref.astype(np.float32))
    ok = bool(np.allclose(got, ref, atol=atol, rtol=rtol))
    print(f"{key}: allclose={ok} max_abs_diff={float(diff.max()):.6g} mean_abs_diff={float(diff.mean()):.6g}")
    return ok


def compare_key(
    key: str,
    result: dict[str, np.ndarray],
    reference: Any,
    atol: float,
    rtol: float,
    ignore_final_audio_token: bool,
) -> bool:
    if key == "text_inputs_embeds":
        mask = ~result["acoustic_input_mask"].astype(np.bool_)
        ref_mask = ~reference["acoustic_input_mask"].astype(np.bool_)
        return compare_arrays(
            key,
            result["inputs_embeds"][mask],
            reference["inputs_embeds"][ref_mask],
            atol,
            rtol,
        )
    if key == "audio_features" and ignore_final_audio_token:
        got = result[key]
        ref = reference[key]
        if got.shape != ref.shape:
            return compare_arrays(key, got, ref, atol, rtol)
        if got.shape[0] < 2:
            raise ValueError("--ignore-final-audio-token requires at least two audio tokens")
        final_diff = np.abs(got[-1].astype(np.float32) - ref[-1].astype(np.float32))
        print(
            "audio_features_final_token: "
            f"max_abs_diff={float(final_diff.max()):.6g} "
            f"mean_abs_diff={float(final_diff.mean()):.6g} "
            "(ignored)"
        )
        return compare_arrays(
            "audio_features_without_final_token",
            got[:-1],
            ref[:-1],
            atol,
            rtol,
        )
    if key not in reference:
        print(f"{key}: missing from reference, skipped")
        return True
    return compare_arrays(key, result[key], reference[key], atol, rtol)


def main() -> int:
    args = parse_args()
    if args.no_default_compare_keys:
        args.compare_keys = []
    if args.compare_audio_features:
        args.compare_keys.append("audio_features")
    if args.post_full_vllm:
        args.build_full_inputs_embeds = True
    if "text_inputs_embeds" in args.compare_keys or "inputs_embeds" in args.compare_keys:
        args.build_full_inputs_embeds = True

    checkpoint = Path(args.checkpoint)
    weights_path = checkpoint / AUDIO_ENCODER_FILE
    classes = import_upstream(args.transformers_src)
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)

    processor_path = Path(args.processor_path).expanduser() if args.processor_path else checkpoint
    processor = classes["AutoProcessor"].from_pretrained(processor_path)

    audio = load_audio(Path(args.audio))
    inputs = processor.apply_transcription_request(audio=audio, prompt=args.context_info)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    acoustic_input_mask = input_ids == processor.audio_token_id
    padding_mask = inputs["padding_mask"].to(device)
    input_values = inputs["input_values"].to(device=device, dtype=dtype)

    acoustic_state, semantic_state, projector_state, wte = load_hf_audio_state(
        weights_path,
        include_wte=args.build_full_inputs_embeds,
    )
    asr_config = classes["VibeVoiceAsrConfig"].from_pretrained(checkpoint)
    acoustic, semantic, projector = build_modules(classes, checkpoint, dtype=dtype, device=device)
    strict_load(acoustic, acoustic_state, "acoustic_tokenizer_encoder")
    strict_load(semantic, semantic_state, "semantic_tokenizer_encoder")
    strict_load(projector, projector_state, "multi_modal_projector")

    inputs_embeds: torch.Tensor | None = None
    with torch.inference_mode():
        audio_features = encode_audio_features(
            acoustic=acoustic,
            semantic=semantic,
            projector=projector,
            input_values=input_values,
            padding_mask=padding_mask,
            chunk_size=asr_config.acoustic_tokenizer_chunk_size,
            hop_length=asr_config.acoustic_tokenizer_encoder_config.hop_length,
            vae_std=asr_config.acoustic_tokenizer_encoder_config.vae_std,
            seed=args.seed,
        )
        if int(acoustic_input_mask.sum()) != int(audio_features.shape[0]):
            raise ValueError(
                f"prompt has {int(acoustic_input_mask.sum())} audio slots, "
                f"but encoder produced {int(audio_features.shape[0])} features"
            )
        if args.build_full_inputs_embeds:
            if wte is None:
                raise RuntimeError("internal error: --build-full-inputs-embeds requested but WTE was not loaded")
            wte = wte.to(device=device, dtype=dtype)
            inputs_embeds = torch.nn.functional.embedding(input_ids, wte).clone()
            inputs_embeds[acoustic_input_mask] = audio_features.reshape(-1, audio_features.shape[-1])

    speech_token_count = int(acoustic_input_mask.sum().item())
    duration_seconds = int(input_values.shape[-1]) / 24000
    mixed_user_text = build_asr_user_text(duration_seconds, args.context_info)
    result = {
        "input_ids": input_ids[0].detach().cpu().numpy().astype(np.int64),
        "attention_mask": attention_mask[0].detach().cpu().numpy().astype(np.int64),
        "acoustic_input_mask": acoustic_input_mask[0].detach().cpu().numpy().astype(np.bool_),
        "speech_masks": np.ones((speech_token_count,), dtype=np.bool_),
        "audio_features": audio_features.detach().cpu().float().numpy(),
        "source_model_path": np.array(str(checkpoint)),
        "tokenizer_path": np.array(str(processor_path)),
        "audio_path": np.array(str(args.audio)),
        "sample_rate": np.array(24000),
        "num_audio_samples": np.array(int(input_values.shape[-1])),
        "audio_duration_seconds": np.array(duration_seconds, dtype=np.float64),
        "seed": np.array(args.seed),
        "dtype": np.array(args.dtype),
        "speech_token_count": np.array(speech_token_count),
        "hidden_size": np.array(int(audio_features.shape[-1])),
        "prompt_length": np.array(int(input_ids.shape[-1])),
        "mixed_vllm_system_prompt": np.array(args.system_prompt),
        "mixed_vllm_user_text": np.array(mixed_user_text),
        "mixed_vllm_endpoint": np.array("/v1/chat/completions"),
        "mixed_vllm_content_part_type": np.array("prompt_embeds"),
    }
    if inputs_embeds is not None:
        result["inputs_embeds"] = inputs_embeds[0].detach().cpu().float().numpy()

    output_path = Path(args.output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **result)
    print(f"wrote {output_path}")
    print(
        "mixed_vllm_ready "
        f"audio_rows={int(audio_features.shape[0])} "
        f"hidden_size={int(audio_features.shape[-1])} "
        f"user_text={mixed_user_text!r}"
    )

    stop = args.stop if args.stop is not None else DEFAULT_STOP
    if args.post_mixed_vllm:
        url = args.url.rstrip("/") + "/v1/chat/completions"
        payload = build_mixed_chat_payload(
            model=args.model,
            system_prompt=args.system_prompt,
            user_text=mixed_user_text,
            audio_features=audio_features,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            stop=stop,
        )
        print(
            "posting mixed vLLM Chat Completions request "
            f"url={url} model={args.model!r} "
            f"audio_rows={int(audio_features.shape[0])} "
            f"hidden_size={int(audio_features.shape[-1])}",
            flush=True,
        )
        data, elapsed = post_json(url, payload, args.timeout)
        text = response_text(data)
        print("\n--- mixed vLLM output ---")
        print(text)
        print("--- end mixed vLLM output ---")
        print(f"mixed_elapsed_seconds={elapsed:.2f}")
        usage = data.get("usage")
        if usage:
            print("mixed_usage=" + json.dumps(usage, sort_keys=True))

    if args.post_full_vllm:
        if inputs_embeds is None:
            raise RuntimeError("--post-full-vllm requires --build-full-inputs-embeds")
        url = args.url.rstrip("/") + "/v1/completions"
        payload = build_full_completions_payload(
            model=args.model,
            inputs_embeds=inputs_embeds[0],
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            stop=stop,
        )
        print(
            "posting full vLLM Completions baseline "
            f"url={url} model={args.model!r} "
            f"prompt_length={int(inputs_embeds.shape[1])} "
            f"hidden_size={int(inputs_embeds.shape[-1])}",
            flush=True,
        )
        data, elapsed = post_json(url, payload, args.timeout)
        text = response_text(data)
        print("\n--- full vLLM output ---")
        print(text)
        print("--- end full vLLM output ---")
        print(f"full_elapsed_seconds={elapsed:.2f}")
        usage = data.get("usage")
        if usage:
            print("full_usage=" + json.dumps(usage, sort_keys=True))

    if not args.reference_npz:
        return 0

    reference = np.load(args.reference_npz)
    ok = True
    for key in args.compare_keys:
        ok = compare_key(
            key,
            result,
            reference,
            args.atol,
            args.rtol,
            args.ignore_final_audio_token,
        ) and ok

    if ok:
        print("comparison ok")
        return 0
    print("comparison failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
