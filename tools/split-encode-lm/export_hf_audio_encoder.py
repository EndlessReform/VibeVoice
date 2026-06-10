#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "transformers @ git+https://github.com/huggingface/transformers.git@cbb65a4815d44f1d8b8ff7f51cca24ce491fc09e",
#   "safetensors",
#   "torch",
# ]
# ///
"""Convert the transitional ASR audio split into an HF-main keyed bundle.

By default this writes the minimal runtime artifact for the mixed vLLM
``prompt_embeds`` path: audio encoders, multimodal projector, ``config.json``,
and ``bundle_config.json``. It deliberately does not copy the Qwen word
embedding table (WTE) or tokenizer/processor files unless explicitly requested.

The WTE and tokenizer are useful for the older full-``inputs_embeds`` validation
path, but they are not needed when vLLM Chat Completions receives normal text
parts plus an audio-only ``prompt_embeds`` content part.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


TRANSFORMERS_GIT_REV = "cbb65a4815d44f1d8b8ff7f51cca24ce491fc09e"
AUDIO_ENCODER_FILE = "audio_encoder.safetensors"
WTE_KEY = "model.language_model.embed_tokens.weight"
ASR_CHAT_TEMPLATE = """{%- set system_prompt = system_prompt | default("You are a helpful assistant that transcribes audio input into text output in JSON format.") -%}
<|im_start|>system
{{ system_prompt }}<|im_end|>
{%- set audio_token = audio_token | default("<|box_start|>") -%}
{%- set audio_start_token = "<|object_ref_start|>" -%}
{%- set audio_end_token = "<|object_ref_end|>" -%}
{%- for message in messages -%}
    {%- if message['role'] == 'user' -%}
{{ '\n' }}<|im_start|>user{{ '\n' }}{%- set text_items = message['content'] | selectattr('type', 'equalto', 'text') | list -%}
        {%- set context_text = text_items[0]['text'] if text_items else none -%}
        {%- for item in message['content'] -%}
            {%- if item['type'] == 'audio' -%}
{{ audio_start_token }}{{ audio_token }}{{ audio_end_token }}{{ "\n" }}{%- if context_text -%}
This is a <|AUDIO_DURATION|> seconds audio, with extra info: {{ context_text }}

Please transcribe it with these keys: Start time, End time, Speaker ID, Content{%- else -%}
This is a <|AUDIO_DURATION|> seconds audio, please transcribe it with these keys: Start time, End time, Speaker ID, Content{%- endif -%}
            {%- endif -%}
        {%- endfor -%}
<|im_end|>{{ '\n' }}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
{{ '<|im_start|>assistant\n' }}
{%- endif -%}"""
PROCESSOR_OUTPUT_FILES = {
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vibevoice_asr_tokenizer_export.json",
    "vocab.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read a repo-fork-keyed audio_encoder.safetensors split and write a "
            "Transformers-main-compatible audio encoder bundle."
        )
    )
    parser.add_argument("--input-dir", default="out/audioonly-checkpoint")
    parser.add_argument("--output-dir", default="out/audioonly-hf-checkpoint")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument(
        "--include-wte",
        action="store_true",
        help=(
            "Copy model.language_model.embed_tokens.weight into the exported "
            "audio_encoder.safetensors. Off by default because mixed vLLM "
            "Chat Completions embeds normal text on the server."
        ),
    )
    parser.add_argument(
        "--include-processor-files",
        action="store_true",
        help=(
            "Write tokenizer/processor files into the output bundle. Off by "
            "default for the audio-only mixed-vLLM runtime artifact."
        ),
    )
    parser.add_argument(
        "--acoustic-vae-std",
        type=float,
        default=0.625,
        help=(
            "HF acoustic encoder sampling std. This corresponds to the original "
            "repo's gaussian std scale fix_std / 0.8; the VibeVoice-ASR default "
            "is 0.5 / 0.8 = 0.625."
        ),
    )
    return parser.parse_args()


def hf_encoder_key(prefix: str, old_key: str, hf_prefix: str) -> str | None:
    if not old_key.startswith(prefix):
        return None

    key = old_key.removeprefix(prefix)
    if key.startswith("decoder."):
        return None

    if key.startswith("encoder.downsample_layers.0.0."):
        mapped = "stem.conv." + key.removeprefix("encoder.downsample_layers.0.0.")
    elif key.startswith("encoder.stages.0."):
        mapped = "stem.stage." + key.removeprefix("encoder.stages.0.")
    elif key.startswith("encoder.head."):
        mapped = "head." + key.removeprefix("encoder.head.")
    else:
        mapped = None
        for idx in range(1, 7):
            if key.startswith(f"encoder.downsample_layers.{idx}.0."):
                mapped = (
                    f"conv_layers.{idx - 1}.conv."
                    + key.removeprefix(f"encoder.downsample_layers.{idx}.0.")
                )
                break
            if key.startswith(f"encoder.stages.{idx}."):
                mapped = (
                    f"conv_layers.{idx - 1}.stage."
                    + key.removeprefix(f"encoder.stages.{idx}.")
                )
                break

    if mapped is None:
        return None

    mapped = mapped.replace(".mixer.conv.conv.conv.", ".mixer.conv.")
    mapped = mapped.replace("conv.conv.", "conv.")
    return hf_prefix + mapped


def hf_projector_key(old_key: str) -> str | None:
    mappings = {
        "model.acoustic_connector.fc1.": "model.multi_modal_projector.acoustic_linear_1.",
        "model.acoustic_connector.norm.": "model.multi_modal_projector.acoustic_norm.",
        "model.acoustic_connector.fc2.": "model.multi_modal_projector.acoustic_linear_2.",
        "model.semantic_connector.fc1.": "model.multi_modal_projector.semantic_linear_1.",
        "model.semantic_connector.norm.": "model.multi_modal_projector.semantic_norm.",
        "model.semantic_connector.fc2.": "model.multi_modal_projector.semantic_linear_2.",
    }
    for prefix, replacement in mappings.items():
        if old_key.startswith(prefix):
            return replacement + old_key.removeprefix(prefix)
    return None


def convert_state(input_path: Path, *, include_wte: bool) -> tuple[dict, dict[str, int], tuple[int, int]]:
    converted = {}
    wte_shape: tuple[int, int] | None = None
    counts = {
        "acoustic_encoder": 0,
        "semantic_encoder": 0,
        "projector": 0,
        "wte": 0,
        "omitted_decoder_or_unknown": 0,
    }

    with safe_open(input_path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key == WTE_KEY:
                tensor = handle.get_tensor(key)
                wte_shape = (int(tensor.shape[0]), int(tensor.shape[1]))
                if include_wte:
                    converted[key] = tensor
                    counts["wte"] += 1
                else:
                    counts["omitted_decoder_or_unknown"] += 1
                continue

            mapped = hf_encoder_key(
                "model.acoustic_tokenizer.",
                key,
                "model.acoustic_tokenizer_encoder.",
            )
            if mapped is not None:
                converted[mapped] = handle.get_tensor(key)
                counts["acoustic_encoder"] += 1
                continue

            mapped = hf_encoder_key(
                "model.semantic_tokenizer.",
                key,
                "model.semantic_tokenizer_encoder.",
            )
            if mapped is not None:
                converted[mapped] = handle.get_tensor(key)
                counts["semantic_encoder"] += 1
                continue

            mapped = hf_projector_key(key)
            if mapped is not None:
                converted[mapped] = handle.get_tensor(key)
                counts["projector"] += 1
                continue

            counts["omitted_decoder_or_unknown"] += 1

    if wte_shape is None:
        raise RuntimeError(f"Expected one WTE tensor at {WTE_KEY}, got none")
    return converted, counts, wte_shape


def write_config(output_dir: Path, vocab_size: int, hidden_size: int, acoustic_vae_std: float) -> None:
    from transformers import Qwen2Config
    from transformers.models.vibevoice_acoustic_tokenizer import VibeVoiceAcousticTokenizerEncoderConfig
    from transformers.models.vibevoice_asr import VibeVoiceAsrConfig

    config = VibeVoiceAsrConfig(
        acoustic_tokenizer_encoder_config=VibeVoiceAcousticTokenizerEncoderConfig(
            hidden_size=64,
            vae_std=acoustic_vae_std,
        ),
        semantic_tokenizer_encoder_config=VibeVoiceAcousticTokenizerEncoderConfig(hidden_size=128),
        text_config=Qwen2Config(vocab_size=vocab_size, hidden_size=hidden_size),
    )
    config.architectures = ["VibeVoiceAsrForConditionalGeneration"]
    config.transformers_git_revision = TRANSFORMERS_GIT_REV
    config.to_json_file(output_dir / "config.json")


def write_processor_files(input_dir: Path, output_dir: Path, sample_rate: int) -> None:
    from transformers import AutoTokenizer
    from transformers.models.vibevoice_acoustic_tokenizer import VibeVoiceAcousticTokenizerFeatureExtractor
    from transformers.models.vibevoice_asr import VibeVoiceAsrProcessor

    tokenizer = AutoTokenizer.from_pretrained(input_dir)
    processor = VibeVoiceAsrProcessor(
        feature_extractor=VibeVoiceAcousticTokenizerFeatureExtractor(sampling_rate=sample_rate),
        tokenizer=tokenizer,
        chat_template=ASR_CHAT_TEMPLATE,
    )
    processor.save_pretrained(output_dir)

    manifest_path = input_dir / "vibevoice_asr_tokenizer_export.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        with (output_dir / "vibevoice_asr_tokenizer_export.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
            handle.write("\n")


def remove_processor_files(output_dir: Path) -> None:
    """Remove stale tokenizer/processor files from a minimal audio-only export.

    The exporter may be rerun into an existing directory. Without this cleanup,
    a previous full bundle could leave tokenizer files behind even though the
    current invocation did not opt into them. The default artifact should be
    honest: audio weights plus config only.
    """

    for name in PROCESSOR_OUTPUT_FILES:
        path = output_dir / name
        if path.exists():
            path.unlink()


def write_bundle_config(
    output_dir: Path,
    counts: dict[str, int],
    vocab_size: int,
    hidden_size: int,
    sample_rate: int,
    acoustic_vae_std: float,
    include_wte: bool,
    include_processor_files: bool,
) -> None:
    bundle_config = {
        "format": "vibevoice-asr-hf-audio-encoder-v1",
        "transformers_git_revision": TRANSFORMERS_GIT_REV,
        "audio_encoder_weight_format": "hf-vibevoice-asr-audio-v1",
        "audio_encoder_file": AUDIO_ENCODER_FILE,
        "includes_wte": include_wte,
        "includes_processor_files": include_processor_files,
        "wte_key": WTE_KEY if include_wte else None,
        "text_hidden_size": hidden_size,
        "text_vocab_size": vocab_size,
        "sample_rate": sample_rate,
        "acoustic_vae_std": acoustic_vae_std,
        "speech_token_compress_ratio": 3200,
        "key_prefixes": {
            "acoustic_encoder": "model.acoustic_tokenizer_encoder.",
            "semantic_encoder": "model.semantic_tokenizer_encoder.",
            "projector": "model.multi_modal_projector.",
            "wte": WTE_KEY if include_wte else None,
        },
        "tensor_counts": counts,
    }
    with (output_dir / "bundle_config.json").open("w", encoding="utf-8") as handle:
        json.dump(bundle_config, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    input_weights = input_dir / AUDIO_ENCODER_FILE
    output_weights = output_dir / AUDIO_ENCODER_FILE

    if not input_weights.exists():
        raise FileNotFoundError(input_weights)

    output_dir.mkdir(parents=True, exist_ok=True)
    state, counts, (vocab_size, hidden_size) = convert_state(
        input_weights,
        include_wte=args.include_wte,
    )

    save_file(
        state,
        output_weights,
        metadata={
            "format": "hf-vibevoice-asr-audio-v1",
            "transformers_git_revision": TRANSFORMERS_GIT_REV,
        },
    )
    write_config(
        output_dir,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        acoustic_vae_std=args.acoustic_vae_std,
    )
    if args.include_processor_files:
        write_processor_files(input_dir, output_dir, sample_rate=args.sample_rate)
    else:
        remove_processor_files(output_dir)
    write_bundle_config(
        output_dir,
        counts=counts,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        sample_rate=args.sample_rate,
        acoustic_vae_std=args.acoustic_vae_std,
        include_wte=args.include_wte,
        include_processor_files=args.include_processor_files,
    )

    total_params = sum(t.numel() for t in state.values())
    total_bytes = sum(t.numel() * t.element_size() for t in state.values())
    print(f"Wrote {output_weights}")
    print(f"  keys: {len(state)}")
    print(f"  params: {total_params:,}")
    print(f"  size: {total_bytes / 1024 / 1024:.1f} MiB")
    print(f"  include_wte: {args.include_wte}")
    print(f"  include_processor_files: {args.include_processor_files}")
    for key, value in counts.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
