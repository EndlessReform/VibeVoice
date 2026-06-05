#!/usr/bin/env python3
"""
Export the merged VibeVoice-ASR text tokenizer.

This is intentionally one-off development tooling. Runtime split checkpoints
should vendor the exported tokenizer files and load them with vanilla
AutoTokenizer/tokenizers; they should not import the custom tokenizer class.

This is a one-off helper for split deployments where the audio encoder runs
outside the LM process. It follows the same tokenizer path as
demo/vibevoice_asr_gradio_demo.py:

    VibeVoiceASRProcessor.from_pretrained(model_path)
      -> VibeVoiceASRTextTokenizerFast.from_pretrained(language_model)

The important side effect is VibeVoiceASRTextTokenizerFast.__init__(), which
adds the ASR speech tokens and chat template from
vibevoice/modular/modular_vibevoice_text_tokenizer.py before save_pretrained().
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_LANGUAGE_MODEL = "Qwen/Qwen2.5-7B"
PREPROCESSOR_CONFIG = "preprocessor_config.json"
MODEL_CONFIG = "config.json"
REQUIRED_SPEECH_TOKENS = (
    "<|object_ref_start|>",
    "<|object_ref_end|>",
    "<|box_start|>",
)


def _load_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_preprocessor_config(model_path: str, local_files_only: bool = False) -> dict[str, Any]:
    return _load_hf_json(model_path, PREPROCESSOR_CONFIG, local_files_only=local_files_only)


def _load_model_config(model_path: str, local_files_only: bool = False) -> dict[str, Any]:
    return _load_hf_json(model_path, MODEL_CONFIG, local_files_only=local_files_only)


def _load_hf_json(
    model_path: str,
    filename: str,
    local_files_only: bool = False,
) -> dict[str, Any]:
    local_path = Path(model_path) / filename
    if local_path.exists():
        return _load_json(local_path)

    try:
        from transformers.utils import cached_file

        cached_path = cached_file(
            model_path,
            filename,
            local_files_only=local_files_only,
        )
    except Exception:
        return {}

    if cached_path is None:
        return {}
    return _load_json(cached_path)


def _infer_language_model_from_asr_config(config: dict[str, Any]) -> str | None:
    decoder_config = config.get("decoder_config") or {}
    if decoder_config.get("model_type") != "qwen2":
        return None

    qwen2_5_7b_shape = {
        "vocab_size": 152064,
        "hidden_size": 3584,
        "intermediate_size": 18944,
        "num_hidden_layers": 28,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
    }
    if all(decoder_config.get(key) == value for key, value in qwen2_5_7b_shape.items()):
        return "Qwen/Qwen2.5-7B"

    qwen2_5_1_5b_shape = {
        "vocab_size": 151936,
        "hidden_size": 1536,
        "intermediate_size": 8960,
        "num_hidden_layers": 28,
        "num_attention_heads": 12,
        "num_key_value_heads": 2,
    }
    if all(decoder_config.get(key) == value for key, value in qwen2_5_1_5b_shape.items()):
        return "Qwen/Qwen2.5-1.5B"

    return None


def _resolve_language_model(
    model_path: str,
    override: str | None,
    local_files_only: bool = False,
) -> str:
    if override:
        return override

    config = _load_preprocessor_config(model_path, local_files_only=local_files_only)
    if config.get("language_model_pretrained_name"):
        return config["language_model_pretrained_name"]

    model_config = _load_model_config(model_path, local_files_only=local_files_only)
    inferred_language_model = _infer_language_model_from_asr_config(model_config)
    return inferred_language_model or DEFAULT_LANGUAGE_MODEL


def _resolve_cached_tokenizer_dir(model_id_or_path: str, local_files_only: bool) -> str:
    path = Path(model_id_or_path)
    if path.exists():
        return str(path)

    if not local_files_only:
        return model_id_or_path

    from transformers.utils import cached_file

    tokenizer_path = cached_file(
        model_id_or_path,
        "tokenizer.json",
        local_files_only=True,
    )
    if tokenizer_path is None:
        return model_id_or_path
    return str(Path(tokenizer_path).parent)


def _verify_tokenizer(tokenizer: Any) -> dict[str, int]:
    ids = {token: tokenizer.convert_tokens_to_ids(token) for token in REQUIRED_SPEECH_TOKENS}
    missing = [token for token, token_id in ids.items() if token_id is None or token_id < 0]
    if missing:
        raise RuntimeError(f"Tokenizer is missing required ASR speech tokens: {missing}")

    if tokenizer.speech_start_id != ids["<|object_ref_start|>"]:
        raise RuntimeError("speech_start_id does not match <|object_ref_start|>")
    if tokenizer.speech_end_id != ids["<|object_ref_end|>"]:
        raise RuntimeError("speech_end_id does not match <|object_ref_end|>")
    if tokenizer.speech_pad_id != ids["<|box_start|>"]:
        raise RuntimeError("speech_pad_id does not match <|box_start|>")

    return ids


def _write_vanilla_tokenizer_config(output_path: Path) -> None:
    config_path = output_path / "tokenizer_config.json"
    if not config_path.exists():
        return

    config = _load_json(config_path)
    config["tokenizer_class"] = "Qwen2TokenizerFast"
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def export_tokenizer(
    model_path: str,
    output_dir: str,
    language_model: str | None = None,
    trust_remote_code: bool = False,
    local_files_only: bool = False,
) -> None:
    from vibevoice.modular.modular_vibevoice_text_tokenizer import (
        VibeVoiceASRTextTokenizerFast,
    )

    resolved_language_model = _resolve_language_model(
        model_path,
        language_model,
        local_files_only=local_files_only,
    )
    tokenizer_source = _resolve_cached_tokenizer_dir(
        resolved_language_model,
        local_files_only=local_files_only,
    )

    tokenizer = VibeVoiceASRTextTokenizerFast.from_pretrained(
        tokenizer_source,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    token_ids = _verify_tokenizer(tokenizer)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_path)
    _write_vanilla_tokenizer_config(output_path)

    manifest = {
        "source_model_path": model_path,
        "language_model_pretrained_name": resolved_language_model,
        "tokenizer_source": tokenizer_source,
        "tokenizer_class": "Qwen2TokenizerFast",
        "created_with_tokenizer_class": "VibeVoiceASRTextTokenizerFast",
        "speech_token_ids": token_ids,
    }
    manifest_path = output_path / "vibevoice_asr_tokenizer_export.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    print(f"Exported merged ASR text tokenizer to: {output_path}")
    print(f"Resolved language-model tokenizer: {resolved_language_model}")
    print(f"Loaded tokenizer source: {tokenizer_source}")
    for token, token_id in token_ids.items():
        print(f"  {token}: {token_id}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export tokenizer files after applying VibeVoice-ASR speech-token merge."
    )
    parser.add_argument(
        "--model-path",
        default="microsoft/VibeVoice-ASR",
        help=(
            "VibeVoice-ASR model path or Hugging Face repo used to resolve "
            "preprocessor_config.json."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where merged tokenizer files will be written.",
    )
    parser.add_argument(
        "--language-model",
        default=None,
        help=(
            "Override the base language-model tokenizer. By default this is read from "
            "preprocessor_config.json, then falls back to Qwen/Qwen2.5-7B."
        ),
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Forward trust_remote_code=True when loading the base tokenizer.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Resolve tokenizer files from the local Hugging Face cache only.",
    )
    args = parser.parse_args()

    export_tokenizer(
        model_path=args.model_path,
        output_dir=args.output_dir,
        language_model=args.language_model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )


if __name__ == "__main__":
    main()
