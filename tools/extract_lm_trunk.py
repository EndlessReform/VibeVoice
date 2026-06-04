#!/usr/bin/env python
"""Extract the LM trunk and audio encoder from VibeVoice ASR checkpoint into separate safetensors files.

Produces two standalone checkpoints:
  - lm_trunk.safetensors: Qwen2ForCausalLM weights for text-only downstream use
  - audio_encoder.safetensors: acoustic/semantic tokenizers + connectors, for audio feature extraction
"""

import argparse
import json
from pathlib import Path

from safetensors.torch import save_file


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Split a VibeVoice ASR checkpoint into LM trunk and audio encoder "
            "safetensors files."
        ),
    )
    parser.add_argument(
        "--model-path",
        default="microsoft/VibeVoice-ASR",
        help=(
            "HF repo id or local checkpoint directory. "
            "Defaults to microsoft/VibeVoice-ASR."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="out",
        help="Directory to write output files.",
    )
    return parser.parse_args()


def resolve_checkpoint_dir(model_path: str) -> Path:
    path = Path(model_path).expanduser()
    if path.exists():
        return path

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=model_path,
            allow_patterns=[
                "config.json",
                "model.safetensors.index.json",
                "model-*.safetensors",
            ],
        )
    )


def load_tensors(checkpoint_dir: Path, keys_to_load: dict[str, str]) -> dict[str, "torch.Tensor"]:
    """Load selected tensors from sharded safetensors, returning a plain state dict."""
    import torch
    from safetensors import safe_open

    state_dict = {}
    for shard_name in sorted(set(keys_to_load.values())):
        shard_path = checkpoint_dir / shard_name
        keys = [k for k, s in keys_to_load.items() if s == shard_name]
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in keys:
                state_dict[key] = f.get_tensor(key)
    return state_dict


def save_checkpoint(state_dict: dict, out_path: Path, metadata: dict | None = None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state_dict, str(out_path), metadata=metadata)

    import torch

    param_count = sum(t.numel() for t in state_dict.values())
    total_bytes = sum(t.element_size() * t.numel() for t in state_dict.values())
    print(f"Wrote {out_path}")
    print(f"  Parameters: {param_count:,}")
    print(f"  Size: {total_bytes / 1024 / 1024:.1f} MB")
    print(f"  Keys: {len(state_dict)}")
    top_keys = sorted(set(k.split(".")[0] for k in state_dict.keys()))
    print(f"  Top-level keys: {top_keys}")


def rename_lm_key(key: str) -> str:
    """Map VibeVoice language-model keys to standalone Qwen2ForCausalLM keys."""
    if key == "lm_head.weight":
        return key

    suffix = key.removeprefix("model.language_model.")

    if suffix.startswith("model."):
        return suffix

    qwen2_model_keys = (
        "embed_tokens.weight",
        "layers.",
        "norm.weight",
    )
    if suffix == "embed_tokens.weight" or suffix == "norm.weight" or suffix.startswith("layers."):
        return f"model.{suffix}"

    return suffix


def validate_lm_keys(state_dict: dict):
    required_keys = {
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    }
    missing_keys = sorted(required_keys - set(state_dict))
    if missing_keys:
        raise RuntimeError(f"LM checkpoint is missing required Qwen2 keys: {missing_keys}")


def main():
    args = parse_args()
    checkpoint_dir = resolve_checkpoint_dir(args.model_path)
    out_dir = Path(args.output_dir).expanduser()

    # Load weight index
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing checkpoint index: {index_path}")

    with index_path.open("r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    # ---- Split keys ----
    lm_keys = {
        k: s
        for k, s in weight_map.items()
        if k.startswith("model.language_model.") or k == "lm_head.weight"
    }

    audio_keys = {
        k: s
        for k, s in weight_map.items()
        if not (k.startswith("model.language_model.") or k == "lm_head.weight")
    }

    print(f"Total keys: {len(weight_map)}")
    print(f"  LM trunk:     {len(lm_keys)} keys")
    print(f"  Audio encoder: {len(audio_keys)} keys")
    print()

    # ---- LM trunk ----
    print("=== LM TRUNK ===")
    lm_state = load_tensors(checkpoint_dir, lm_keys)
    lm_renamed = {rename_lm_key(k): v for k, v in lm_state.items()}
    validate_lm_keys(lm_renamed)
    save_checkpoint(
        lm_renamed,
        out_dir / "lm_trunk.safetensors",
        metadata={
            "extracted_from": "VibeVoice-ASR",
            "description": "LM trunk (WTE + transformer layers + lm_head)",
        },
    )

    # ---- Audio encoder ----
    print()
    print("=== AUDIO ENCODER ===")
    audio_state = load_tensors(checkpoint_dir, audio_keys)
    # Strip model. prefix for standalone format
    audio_renamed = {}
    for k, v in audio_state.items():
        new_k = k.removeprefix("model.")
        audio_renamed[new_k] = v
    save_checkpoint(
        audio_renamed,
        out_dir / "audio_encoder.safetensors",
        metadata={
            "extracted_from": "VibeVoice-ASR",
            "description": (
                "Audio encoder (acoustic_tokenizer + semantic_tokenizer + "
                "acoustic_connector + semantic_connector)"
            ),
        },
    )


if __name__ == "__main__":
    main()
