#!/usr/bin/env python
"""Export VibeVoice ASR audio embeddings to an npz fixture."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from vibevoice.modular.configuration_vibevoice import VibeVoiceASRConfig
from vibevoice.modular.modeling_vibevoice import SpeechConnector
from vibevoice.modular.modular_vibevoice_tokenizer import (
    VibeVoiceTokenizerEncoderOutput,
    VibeVoiceTokenizerStreamingCache,
    VibeVoiceAcousticTokenizerModel,
    VibeVoiceSemanticTokenizerModel,
)
from vibevoice.processor.audio_utils import AudioNormalizer, load_audio_use_ffmpeg


ACOUSTIC_TOKENIZER_PREFIX = "model.acoustic_tokenizer."
ACOUSTIC_CONNECTOR_PREFIX = "model.acoustic_connector."
SEMANTIC_TOKENIZER_PREFIX = "model.semantic_tokenizer."
SEMANTIC_CONNECTOR_PREFIX = "model.semantic_connector."


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Load only the VibeVoice ASR audio encoder path and export the "
            "combined acoustic+semantic tensor that can be spliced into speech "
            "prompt slots."
        )
    )
    parser.add_argument(
        "--model-path",
        default="microsoft/VibeVoice-ASR",
        help="HF repo id or local checkpoint directory. Defaults to microsoft/VibeVoice-ASR.",
    )
    parser.add_argument(
        "--audio",
        default="./flight.wav",
        help="Audio file to encode. The file is decoded as mono 24 kHz with ffmpeg.",
    )
    parser.add_argument(
        "--output",
        default="./flight_audio_features.npz",
        help="Destination .npz path.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for encoding.",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=("float32", "float16", "bfloat16"),
        help="Dtype for the acoustic/semantic encoders and connectors.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip the repo's AudioNormalizer step.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Seed used before sampling acoustic latents. The ASR path samples "
            "acoustic latents, so set this to match later allclose fixtures."
        ),
    )
    parser.add_argument(
        "--streaming-segment-duration",
        type=float,
        default=60.0,
        help="Segment duration used by the ASR encoder streaming path.",
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


def load_prefixed_weights(module: torch.nn.Module, checkpoint_dir: Path, prefix: str):
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing checkpoint index: {index_path}")

    with index_path.open("r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    selected = {
        key: shard for key, shard in weight_map.items() if key.startswith(prefix)
    }
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


def load_audio_tensor(audio_path: str, device: torch.device, dtype: torch.dtype, normalize: bool):
    audio, original_sample_rate = load_audio_use_ffmpeg(
        str(Path(audio_path).expanduser()),
        resample=False,
    )
    audio = audio.astype(np.float32)
    sample_rate = 24000
    if original_sample_rate != sample_rate:
        import librosa

        audio = librosa.resample(
            audio,
            orig_sr=original_sample_rate,
            target_sr=sample_rate,
        ).astype(np.float32)
    if normalize:
        audio = AudioNormalizer()(audio)

    tensor = torch.from_numpy(audio).to(device=device, dtype=dtype).unsqueeze(0)
    return tensor, sample_rate, original_sample_rate


def iter_segments(total_length: int, segment_length: int):
    if segment_length <= 0:
        raise ValueError("segment_length must be positive")
    for start in range(0, total_length, segment_length):
        end = min(start + segment_length, total_length)
        if end > start:
            yield start, end


def main():
    args = parse_args()
    checkpoint_dir = resolve_checkpoint_dir(args.model_path)
    config = VibeVoiceASRConfig.from_pretrained(checkpoint_dir)

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

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
    acoustic_tokenizer.eval()
    semantic_tokenizer.eval()
    acoustic_connector.eval()
    semantic_connector.eval()

    audio, sample_rate, original_sample_rate = load_audio_tensor(
        args.audio,
        device=device,
        dtype=dtype,
        normalize=not args.no_normalize,
    )

    with torch.inference_mode():
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)

        segment_samples = int(args.streaming_segment_duration * sample_rate)
        use_streaming = audio.shape[-1] > segment_samples

        if not use_streaming:
            encoder_input = audio.unsqueeze(1)
            acoustic_out = acoustic_tokenizer.encode(encoder_input)
            acoustic_tokens = acoustic_out.sample(
                dist_type=acoustic_tokenizer.std_dist_type
            )[0]
            acoustic_features = acoustic_connector(acoustic_tokens)

            semantic_tokens = semantic_tokenizer.encode(encoder_input).mean
            semantic_features = semantic_connector(semantic_tokens)
        else:
            acoustic_cache = VibeVoiceTokenizerStreamingCache()
            semantic_cache = VibeVoiceTokenizerStreamingCache()
            acoustic_mean_segments = []
            semantic_mean_segments = []
            sample_indices = torch.arange(audio.shape[0], device=device)
            segments = list(iter_segments(audio.shape[-1], segment_samples))

            for segment_index, (start, end) in enumerate(segments):
                chunk = audio[:, start:end].contiguous()
                is_final = segment_index == len(segments) - 1

                acoustic_out = acoustic_tokenizer.encode(
                    chunk.unsqueeze(1),
                    cache=acoustic_cache,
                    sample_indices=sample_indices,
                    use_cache=True,
                    is_final_chunk=is_final,
                )
                acoustic_mean_segments.append(acoustic_out.mean)

                semantic_out = semantic_tokenizer.encode(
                    chunk.unsqueeze(1),
                    cache=semantic_cache,
                    sample_indices=sample_indices,
                    use_cache=True,
                    is_final_chunk=is_final,
                )
                semantic_mean_segments.append(semantic_out.mean)

            acoustic_mean = torch.cat(acoustic_mean_segments, dim=1).contiguous()
            acoustic_out = VibeVoiceTokenizerEncoderOutput(
                mean=acoustic_mean,
                std=acoustic_tokenizer.fix_std,
            )
            acoustic_tokens = acoustic_out.sample(
                dist_type=acoustic_tokenizer.std_dist_type
            )[0]
            acoustic_features = acoustic_connector(acoustic_tokens)

            semantic_tokens = torch.cat(semantic_mean_segments, dim=1).contiguous()
            semantic_features = semantic_connector(semantic_tokens)

        output = acoustic_features + semantic_features

    output_np = output.detach().cpu().float().numpy()
    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        output=output_np,
        sample_rate=np.array(sample_rate, dtype=np.int32),
        original_sample_rate=np.array(original_sample_rate, dtype=np.int32),
        num_samples=np.array(audio.shape[-1], dtype=np.int64),
        seed=np.array(args.seed, dtype=np.int64),
        streaming=np.array(use_streaming, dtype=np.bool_),
        streaming_segment_duration=np.array(
            args.streaming_segment_duration,
            dtype=np.float32,
        ),
    )

    print(
        f"wrote {out_path} with output shape {output_np.shape} "
        f"and dtype {output_np.dtype}"
    )


if __name__ == "__main__":
    main()
