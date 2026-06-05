#!/usr/bin/env python
"""Build a VibeVoice ASR prefix and POST it to vLLM prompt_embeds Completions."""

import argparse
import base64
import io
import json
import time
from types import SimpleNamespace
from typing import Any

import requests
import torch

from export_asr_prefix import (
    DEFAULT_ASR_REVISION,
    DEFAULT_TEXTONLY_MODEL_PATH,
    build_prefix,
    default_model_path,
)


DEFAULT_STOP_TOKEN = "<|endoftext|>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the exact VibeVoice ASR LM prefix embeddings and send them to "
            "a vLLM OpenAI-compatible /v1/completions endpoint."
        )
    )
    parser.add_argument("audio", help="Audio file to transcribe.")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="vLLM server URL. Defaults to http://localhost:8000.",
    )
    parser.add_argument(
        "--model",
        default="/models/vibevoice",
        help=(
            "Served model name for the vLLM request. Defaults to /models/vibevoice, "
            "matching docs/usage.md."
        ),
    )
    parser.add_argument(
        "--context-info",
        default=None,
        help="Optional ASR context/hotword text passed to VibeVoiceASRProcessor.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32768,
        help="Maximum generated tokens. Defaults to 32768.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Defaults to 0.0.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p value. Defaults to 1.0.",
    )
    parser.add_argument(
        "--stop",
        default=DEFAULT_STOP_TOKEN,
        help=f"Stop string for the Completions request. Defaults to {DEFAULT_STOP_TOKEN!r}.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=12000.0,
        help="HTTP request timeout in seconds. Defaults to 12000.",
    )
    parser.add_argument(
        "--model-path",
        default=default_model_path(),
        help="HF repo id or local VibeVoice-ASR checkpoint directory.",
    )
    parser.add_argument(
        "--revision",
        default=DEFAULT_ASR_REVISION,
        help="HF hub revision (commit hash, branch, or tag) for --model-path.",
    )
    parser.add_argument(
        "--textonly-model-path",
        default=str(DEFAULT_TEXTONLY_MODEL_PATH),
        help="Local text-only LM checkpoint containing tokenizer files and embeddings.",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Optional tokenizer directory. Defaults to --textonly-model-path if usable.",
    )
    parser.add_argument(
        "--language-model",
        default="Qwen/Qwen2.5-7B",
        help=(
            "Fallback tokenizer path when no local tokenizer directory is available. "
            "For runtime prefix export this tokenizer must already include the ASR "
            "speech tokens."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for audio encoding and embedding lookup.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("float32", "float16", "bfloat16"),
        help="Torch dtype for the local prefix build. Defaults to bfloat16.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used before sampling acoustic latents.",
    )
    parser.add_argument(
        "--streaming-segment-duration",
        type=float,
        default=60.0,
        help="Segment duration used by the ASR encoder streaming path.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Resolve Hugging Face files from the local cache only.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Forward trust_remote_code=True when loading tokenizer/model references.",
    )
    return parser.parse_args()


def make_export_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        model_path=args.model_path,
        revision=args.revision,
        textonly_model_path=args.textonly_model_path,
        tokenizer_path=args.tokenizer_path,
        language_model=args.language_model,
        audio=args.audio,
        context_info=args.context_info,
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
        streaming_segment_duration=args.streaming_segment_duration,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        include_speech_tensors=False,
    )


def tensor_to_base64(tensor: torch.Tensor) -> str:
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def response_text(response_json: dict[str, Any]) -> str:
    choices = response_json.get("choices") or []
    if not choices:
        return ""

    choice = choices[0]
    if "text" in choice:
        return choice["text"] or ""

    message = choice.get("message") or {}
    return message.get("content") or ""


def main() -> None:
    args = parse_args()

    print("building prompt embeddings...", flush=True)
    prefix = build_prefix(make_export_args(args))
    prompt_embeds = torch.from_numpy(prefix["inputs_embeds"]).float()
    encoded = tensor_to_base64(prompt_embeds)

    url = args.url.rstrip("/") + "/v1/completions"
    payload = {
        "model": args.model,
        "prompt": None,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "stop": [args.stop],
        "prompt_embeds": encoded,
    }

    print(
        "posting to "
        f"{url} model={args.model!r} "
        f"prompt_length={int(prefix['prompt_length'].item())} "
        f"speech_tokens={int(prefix['speech_token_count'].item())} "
        f"hidden_size={int(prefix['hidden_size'].item())}",
        flush=True,
    )

    started = time.time()
    response = requests.post(url, json=payload, timeout=args.timeout)
    elapsed = time.time() - started

    if response.status_code != 200:
        print(f"request failed: HTTP {response.status_code}")
        print(response.text)
        raise SystemExit(1)

    data = response.json()
    text = response_text(data)

    print("\n--- output ---")
    print(text)
    print("--- end output ---")
    print(f"elapsed_seconds={elapsed:.2f}")

    finish_reason = (data.get("choices") or [{}])[0].get("finish_reason")
    if finish_reason is not None:
        print(f"finish_reason={finish_reason}")

    usage = data.get("usage")
    if usage:
        print("usage=" + json.dumps(usage, sort_keys=True))


if __name__ == "__main__":
    main()
