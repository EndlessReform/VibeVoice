#!/usr/bin/env python
"""Export the exact VibeVoice ASR LM prefix as an NPZ artifact."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from vibevoice.modular.configuration_vibevoice import VibeVoiceASRConfig
from vibevoice.modular.modeling_vibevoice import SpeechConnector
from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.modular.modular_vibevoice_tokenizer import (
    VibeVoiceAcousticTokenizerModel,
    VibeVoiceSemanticTokenizerModel,
    VibeVoiceTokenizerEncoderOutput,
    VibeVoiceTokenizerStreamingCache,
)
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor


DEFAULT_ASR_REPO = "microsoft/VibeVoice-ASR"
DEFAULT_ASR_REVISION = "d0c9efdb8d614685062c04425d91e01b6f37d944"
DEFAULT_TEXTONLY_MODEL_PATH = Path("out/textonly-checkpoint")
ACOUSTIC_TOKENIZER_PREFIX = "model.acoustic_tokenizer."
ACOUSTIC_CONNECTOR_PREFIX = "model.acoustic_connector."
SEMANTIC_TOKENIZER_PREFIX = "model.semantic_tokenizer."
SEMANTIC_CONNECTOR_PREFIX = "model.semantic_connector."


def default_model_path() -> str:
    return DEFAULT_ASR_REPO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the normal VibeVoice ASR prompt, encode the audio, splice the "
            "audio features into LM token embeddings, and write the resulting "
            "prefix tensor to an NPZ file."
        )
    )
    parser.add_argument(
        "--model-path",
        default=default_model_path(),
        help="HF repo id or local VibeVoice-ASR checkpoint directory.",
    )
    parser.add_argument(
        "--revision",
        default=DEFAULT_ASR_REVISION,
        help="HF hub revision (commit hash, branch, or tag).",
    )
    parser.add_argument(
        "--textonly-model-path",
        default=str(DEFAULT_TEXTONLY_MODEL_PATH),
        help=(
            "Local text-only LM checkpoint containing tokenizer files and "
            "model.embed_tokens.weight. Defaults to out/textonly-checkpoint."
        ),
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help=(
            "Optional tokenizer directory. Defaults to --textonly-model-path when "
            "that directory has tokenizer files, otherwise --language-model."
        ),
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
        "--audio",
        required=True,
        help="Audio file to encode with the same processor path as the ASR demo.",
    )
    parser.add_argument(
        "--context-info",
        default=None,
        help="Optional ASR context/hotword text passed to VibeVoiceASRProcessor.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination .npz path.",
    )
    parser.add_argument(
        "--format",
        default="npz",
        choices=("npz",),
        help="Output format. Only npz is currently supported.",
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
        help=(
            "Torch dtype for the audio modules and embedding lookup. The saved "
            "float tensors are converted to float32."
        ),
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
    parser.add_argument(
        "--include-speech-tensors",
        action="store_true",
        help="Also write the normalized 24 kHz waveform used by the processor.",
    )
    parser.add_argument(
        "--compare-npz",
        default=None,
        help="Optional existing prefix NPZ to compare against after export.",
    )
    parser.add_argument(
        "--compare-demo-path",
        action="store_true",
        help=(
            "Also load the full ASR model and compare against the exact demo "
            "model.get_input_embeddings()+model.encode_speech path."
        ),
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-3,
        help="Absolute tolerance for float comparisons.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance for float comparisons.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        help="Attention implementation used only by --compare-demo-path.",
    )
    return parser.parse_args()


def resolve_checkpoint_dir(model_path: str, local_files_only: bool, revision: str = None) -> Path:
    path = Path(model_path).expanduser()
    if path.exists():
        return path

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=model_path,
            revision=revision,
            allow_patterns=[
                "config.json",
                "model.safetensors.index.json",
                "model-*.safetensors",
            ],
            local_files_only=local_files_only,
        )
    )


def load_prefixed_weights(module: torch.nn.Module, checkpoint_dir: Path, prefix: str) -> None:
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing checkpoint index: {index_path}")

    with index_path.open("r", encoding="utf-8") as handle:
        weight_map = json.load(handle)["weight_map"]

    selected = {key: shard for key, shard in weight_map.items() if key.startswith(prefix)}
    if not selected:
        raise ValueError(f"No checkpoint weights found for prefix {prefix!r}")

    state_dict = {}
    for shard_name in sorted(set(selected.values())):
        shard_path = checkpoint_dir / shard_name
        keys = [key for key, shard in selected.items() if shard == shard_name]
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            for key in keys:
                state_dict[key.removeprefix(prefix)] = shard.get_tensor(key)

    module.load_state_dict(state_dict, strict=True)


def load_audio_encoder(
    checkpoint_dir: Path,
    config: VibeVoiceASRConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.nn.Module]:
    acoustic_tokenizer = VibeVoiceAcousticTokenizerModel(
        config.acoustic_tokenizer_config
    ).to(device=device, dtype=dtype)
    semantic_tokenizer = VibeVoiceSemanticTokenizerModel(
        config.semantic_tokenizer_config
    ).to(device=device, dtype=dtype)
    acoustic_connector = SpeechConnector(
        config.acoustic_vae_dim,
        config.decoder_config.hidden_size,
    ).to(device=device, dtype=dtype)
    semantic_connector = SpeechConnector(
        config.semantic_vae_dim,
        config.decoder_config.hidden_size,
    ).to(device=device, dtype=dtype)

    load_prefixed_weights(acoustic_tokenizer, checkpoint_dir, ACOUSTIC_TOKENIZER_PREFIX)
    load_prefixed_weights(semantic_tokenizer, checkpoint_dir, SEMANTIC_TOKENIZER_PREFIX)
    load_prefixed_weights(acoustic_connector, checkpoint_dir, ACOUSTIC_CONNECTOR_PREFIX)
    load_prefixed_weights(semantic_connector, checkpoint_dir, SEMANTIC_CONNECTOR_PREFIX)

    modules = {
        "acoustic_tokenizer": acoustic_tokenizer,
        "semantic_tokenizer": semantic_tokenizer,
        "acoustic_connector": acoustic_connector,
        "semantic_connector": semantic_connector,
    }
    for module in modules.values():
        module.eval()
    return modules


def iter_segments(total_length: int, segment_length: int):
    if segment_length <= 0:
        raise ValueError("segment_length must be positive")
    for start in range(0, total_length, segment_length):
        end = min(start + segment_length, total_length)
        if end > start:
            yield start, end


def encode_speech_with_modules(
    modules: dict[str, torch.nn.Module],
    speech_tensors: torch.Tensor,
    speech_masks: torch.Tensor,
    streaming_segment_duration: float,
) -> torch.Tensor:
    dtype = next(modules["acoustic_tokenizer"].parameters()).dtype
    speech_tensors = speech_tensors.to(dtype)
    if speech_tensors.ndim == 1:
        speech_tensors = speech_tensors.unsqueeze(0)

    batch_size, total_samples = speech_tensors.shape
    sample_rate = 24000
    segment_samples = int(streaming_segment_duration * sample_rate)
    use_streaming = total_samples > segment_samples

    acoustic_tokenizer = modules["acoustic_tokenizer"]
    semantic_tokenizer = modules["semantic_tokenizer"]
    acoustic_connector = modules["acoustic_connector"]
    semantic_connector = modules["semantic_connector"]

    if not use_streaming:
        encoder_output = acoustic_tokenizer.encode(speech_tensors.unsqueeze(1))
        audio_tokens = encoder_output.sample(dist_type=acoustic_tokenizer.std_dist_type)[0]
        acoustic_features = acoustic_connector(audio_tokens)
        semantic_tokens = semantic_tokenizer.encode(speech_tensors.unsqueeze(1)).mean
        semantic_features = semantic_connector(semantic_tokens)
    else:
        acoustic_cache = VibeVoiceTokenizerStreamingCache()
        semantic_cache = VibeVoiceTokenizerStreamingCache()
        acoustic_mean_segments = []
        semantic_mean_segments = []
        sample_indices = torch.arange(batch_size, device=speech_tensors.device)
        segments = list(iter_segments(total_samples, segment_samples))

        for segment_index, (start, end) in enumerate(segments):
            chunk = speech_tensors[:, start:end].contiguous()
            is_final = segment_index == len(segments) - 1

            acoustic_output = acoustic_tokenizer.encode(
                chunk.unsqueeze(1),
                cache=acoustic_cache,
                sample_indices=sample_indices,
                use_cache=True,
                is_final_chunk=is_final,
            )
            acoustic_mean_segments.append(acoustic_output.mean)

            semantic_output = semantic_tokenizer.encode(
                chunk.unsqueeze(1),
                cache=semantic_cache,
                sample_indices=sample_indices,
                use_cache=True,
                is_final_chunk=is_final,
            )
            semantic_mean_segments.append(semantic_output.mean)

        acoustic_mean = torch.cat(acoustic_mean_segments, dim=1).contiguous()
        encoder_output = VibeVoiceTokenizerEncoderOutput(
            mean=acoustic_mean,
            std=acoustic_tokenizer.fix_std,
        )
        audio_tokens = encoder_output.sample(dist_type=acoustic_tokenizer.std_dist_type)[0]
        acoustic_features = acoustic_connector(audio_tokens)
        semantic_tokens = torch.cat(semantic_mean_segments, dim=1).contiguous()
        semantic_features = semantic_connector(semantic_tokens)

    return acoustic_features[speech_masks] + semantic_features[speech_masks]


def tokenizer_source(args: argparse.Namespace) -> str:
    if args.tokenizer_path:
        return args.tokenizer_path

    textonly_path = Path(args.textonly_model_path).expanduser()
    if (textonly_path / "tokenizer.json").exists() or (textonly_path / "tokenizer_config.json").exists():
        return str(textonly_path)

    return args.language_model


def load_processor(args: argparse.Namespace) -> VibeVoiceASRProcessor:
    source = tokenizer_source(args)
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    return VibeVoiceASRProcessor(tokenizer=tokenizer)


def load_text_embedding_weight(
    textonly_model_path: str,
    checkpoint_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    textonly_path = Path(textonly_model_path).expanduser()
    if (textonly_path / "model.safetensors").exists():
        with safe_open(textonly_path / "model.safetensors", framework="pt", device="cpu") as handle:
            return handle.get_tensor("model.embed_tokens.weight").to(device=device, dtype=dtype)

    index_path = textonly_path / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as handle:
            weight_map = json.load(handle)["weight_map"]
        key = "model.embed_tokens.weight"
        shard_path = textonly_path / weight_map[key]
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            return handle.get_tensor(key).to(device=device, dtype=dtype)

    asr_index_path = checkpoint_dir / "model.safetensors.index.json"
    with asr_index_path.open("r", encoding="utf-8") as handle:
        weight_map = json.load(handle)["weight_map"]
    key = "model.language_model.embed_tokens.weight"
    shard_path = checkpoint_dir / weight_map[key]
    with safe_open(shard_path, framework="pt", device="cpu") as handle:
        return handle.get_tensor(key).to(device=device, dtype=dtype)


def set_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def build_prefix(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    checkpoint_dir = resolve_checkpoint_dir(
        args.model_path, args.local_files_only, revision=args.revision
    )
    config = VibeVoiceASRConfig.from_pretrained(checkpoint_dir)
    processor = load_processor(args)

    inputs = processor(
        audio=args.audio,
        sampling_rate=None,
        return_tensors="pt",
        padding=True,
        add_generation_prompt=True,
        context_info=args.context_info,
    )
    input_ids = inputs["input_ids"].to(device)
    acoustic_input_mask = inputs["acoustic_input_mask"].to(device)
    speech_tensors = inputs["speech_tensors"].to(device)
    speech_masks = inputs["speech_masks"].to(device)

    modules = load_audio_encoder(checkpoint_dir, config, device=device, dtype=dtype)
    embedding_weight = load_text_embedding_weight(
        args.textonly_model_path,
        checkpoint_dir,
        device=device,
        dtype=dtype,
    )

    with torch.inference_mode():
        set_seed(args.seed, device)
        audio_features = encode_speech_with_modules(
            modules=modules,
            speech_tensors=speech_tensors,
            speech_masks=speech_masks,
            streaming_segment_duration=args.streaming_segment_duration,
        )
        inputs_embeds = torch.nn.functional.embedding(input_ids, embedding_weight).clone()

        expected_slots = int(acoustic_input_mask.sum().item())
        if audio_features.shape[0] != expected_slots:
            raise ValueError(
                f"Prompt has {expected_slots} speech slots, but encoder produced "
                f"{audio_features.shape[0]} frames"
            )
        if audio_features.shape[-1] != inputs_embeds.shape[-1]:
            raise ValueError(
                f"Audio hidden size {audio_features.shape[-1]} does not match "
                f"text embedding size {inputs_embeds.shape[-1]}"
            )

        inputs_embeds[acoustic_input_mask] = audio_features.reshape(
            -1,
            audio_features.shape[-1],
        )

    result = {
        "input_ids": inputs["input_ids"][0].detach().cpu().numpy().astype(np.int64),
        "attention_mask": inputs["attention_mask"][0].detach().cpu().numpy().astype(np.int64),
        "acoustic_input_mask": inputs["acoustic_input_mask"][0].detach().cpu().numpy().astype(np.bool_),
        "speech_masks": inputs["speech_masks"][0].detach().cpu().numpy().astype(np.bool_),
        "audio_features": audio_features.detach().cpu().float().numpy(),
        "inputs_embeds": inputs_embeds[0].detach().cpu().float().numpy(),
    }
    if args.include_speech_tensors:
        result["speech_tensors"] = inputs["speech_tensors"][0].detach().cpu().numpy().astype(np.float32)

    metadata = {
        "source_model_path": str(args.model_path),
        "textonly_model_path": str(args.textonly_model_path),
        "tokenizer_path": tokenizer_source(args),
        "audio_path": str(args.audio),
        "context_info": args.context_info or "",
        "sample_rate": 24000,
        "num_audio_samples": int(inputs["speech_tensors"].shape[-1]),
        "seed": int(args.seed),
        "dtype": args.dtype,
        "speech_token_count": int(inputs["acoustic_input_mask"].sum().item()),
        "hidden_size": int(result["inputs_embeds"].shape[-1]),
        "prompt_length": int(result["input_ids"].shape[0]),
        "streaming_segment_duration": float(args.streaming_segment_duration),
    }
    result.update({key: np.array(value) for key, value in metadata.items()})
    return result


def write_npz(path: str, arrays: dict[str, Any]) -> None:
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **arrays)


def compare_arrays(
    label: str,
    left: np.ndarray,
    right: np.ndarray,
    atol: float,
    rtol: float,
    exact: bool = False,
) -> bool:
    print(f"{label}: exported_shape={left.shape} reference_shape={right.shape}")
    print(f"{label}: exported_dtype={left.dtype} reference_dtype={right.dtype}")
    if left.shape != right.shape:
        print(f"{label}: shape_match=False")
        return False

    if exact:
        ok = np.array_equal(left, right)
        print(f"{label}: exact={ok}")
        if not ok:
            mismatch = np.flatnonzero(left.reshape(-1) != right.reshape(-1))
            if mismatch.size:
                idx = int(mismatch[0])
                print(f"{label}: first_mismatch_flat_index={idx}")
        return bool(ok)

    diff = np.abs(left.astype(np.float32) - right.astype(np.float32))
    allclose = np.allclose(left, right, atol=atol, rtol=rtol)
    print(f"{label}: allclose={allclose}")
    print(f"{label}: max_abs_diff={float(diff.max()) if diff.size else 0:.6g}")
    print(f"{label}: mean_abs_diff={float(diff.mean()) if diff.size else 0:.6g}")
    return bool(allclose)


def compare_npz(exported: dict[str, Any], fixture_path: str, atol: float, rtol: float) -> bool:
    reference = np.load(Path(fixture_path).expanduser())
    checks = [
        ("input_ids", True),
        ("attention_mask", True),
        ("acoustic_input_mask", True),
        ("speech_masks", True),
        ("audio_features", False),
        ("inputs_embeds", False),
    ]
    ok = True
    for key, exact in checks:
        if key not in reference:
            print(f"{key}: missing from reference, skipped")
            continue
        ok = compare_arrays(key, exported[key], reference[key], atol, rtol, exact=exact) and ok
    return ok


def build_demo_reference(args: argparse.Namespace) -> dict[str, np.ndarray]:
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    processor = load_processor(args)
    inputs = processor(
        audio=args.audio,
        sampling_rate=None,
        return_tensors="pt",
        padding=True,
        add_generation_prompt=True,
        context_info=args.context_info,
    )
    inputs = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}

    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
        args.model_path,
        revision=args.revision,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    ).to(device)
    model.eval()

    with torch.inference_mode():
        text_embeds = model.get_input_embeddings()(inputs["input_ids"]).clone()
        set_seed(args.seed, device)
        audio_features = model.encode_speech(
            speech_tensors=inputs["speech_tensors"],
            speech_masks=inputs["speech_masks"],
            streaming_segment_duration=args.streaming_segment_duration,
        )
        text_embeds[inputs["acoustic_input_mask"]] = audio_features.reshape(
            -1,
            audio_features.shape[-1],
        )

    return {
        "input_ids": inputs["input_ids"][0].detach().cpu().numpy().astype(np.int64),
        "attention_mask": inputs["attention_mask"][0].detach().cpu().numpy().astype(np.int64),
        "acoustic_input_mask": inputs["acoustic_input_mask"][0].detach().cpu().numpy().astype(np.bool_),
        "speech_masks": inputs["speech_masks"][0].detach().cpu().numpy().astype(np.bool_),
        "audio_features": audio_features.detach().cpu().float().numpy(),
        "inputs_embeds": text_embeds[0].detach().cpu().float().numpy(),
    }


def main() -> None:
    args = parse_args()
    exported = build_prefix(args)
    write_npz(args.output, exported)

    print(
        f"wrote {Path(args.output).expanduser()} "
        f"prompt_length={exported['prompt_length'].item()} "
        f"speech_tokens={exported['speech_token_count'].item()} "
        f"hidden_size={exported['hidden_size'].item()}"
    )

    ok = True
    if args.compare_npz:
        print(f"--- comparing against NPZ: {args.compare_npz} ---")
        ok = compare_npz(exported, args.compare_npz, args.atol, args.rtol) and ok

    if args.compare_demo_path:
        print("--- comparing against full demo path ---")
        reference = build_demo_reference(args)
        for key in ("input_ids", "attention_mask", "acoustic_input_mask", "speech_masks"):
            ok = compare_arrays(key, exported[key], reference[key], args.atol, args.rtol, exact=True) and ok
        for key in ("audio_features", "inputs_embeds"):
            ok = compare_arrays(key, exported[key], reference[key], args.atol, args.rtol) and ok

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
