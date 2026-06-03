#!/usr/bin/env python
"""Smoke test ASR generation from precomputed spliced audio embeddings."""

import argparse
import re
from pathlib import Path

import numpy as np
import torch

from vibevoice.modular.modular_vibevoice_text_tokenizer import VibeVoiceASRTextTokenizerFast
from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor


DEFAULT_MODEL_SNAPSHOT = Path(
    "~/.cache/huggingface/hub/models--microsoft--VibeVoice-ASR/"
    "snapshots/d0c9efdb8d614685062c04425d91e01b6f37d944"
).expanduser()
DEFAULT_TOKENIZER_DIR = Path("./vibevoice-asr-text-tokenizer")


def default_model_path() -> str:
    if DEFAULT_MODEL_SNAPSHOT.exists():
        return str(DEFAULT_MODEL_SNAPSHOT)
    return "microsoft/VibeVoice-ASR"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build the normal VibeVoice ASR text prompt, splice npz audio "
            "features into the speech slots after WTE, and run a short "
            "generation smoke test."
        )
    )
    parser.add_argument(
        "--model-path",
        default=default_model_path(),
        help="HF repo id or local checkpoint directory. Defaults to microsoft/VibeVoice-ASR.",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=str(DEFAULT_TOKENIZER_DIR),
        help="Local VibeVoice ASR text tokenizer directory.",
    )
    parser.add_argument(
        "--audio",
        default="./flight.wav",
        help="Audio path used only to build the same ASR prompt and speech mask.",
    )
    parser.add_argument(
        "--npz",
        default="./flight_audio_features.npz",
        help="NPZ produced by export_audio_features_npz.py.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for the full ASR model.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("float32", "float16", "bfloat16"),
        help="Dtype for loading the full ASR model.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        help="Attention implementation passed to from_pretrained.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Number of generated tokens to inspect.",
    )
    parser.add_argument(
        "--compare-encoder",
        action="store_true",
        help="Compare npz output against full-model encode_speech before generation.",
    )
    return parser.parse_args()


def looks_nongibberish(text: str) -> tuple[bool, list[str]]:
    reasons = []
    stripped = text.strip()
    if len(stripped) < 20:
        reasons.append("decoded text is very short")

    if "\ufffd" in stripped:
        reasons.append("decoded text contains replacement characters")

    printable = sum(ch.isprintable() or ch.isspace() for ch in stripped)
    if stripped and printable / len(stripped) < 0.95:
        reasons.append("decoded text has too many non-printable characters")

    alpha = sum(ch.isalpha() for ch in stripped)
    if stripped and alpha / len(stripped) < 0.20:
        reasons.append("decoded text has too little alphabetic content")

    if re.search(r"(.)\1{24,}", stripped):
        reasons.append("decoded text has a long repeated-character run")

    repeated_words = re.findall(r"\b([\w'-]{2,})\b(?:\s+\1\b){8,}", stripped, re.I)
    if repeated_words:
        reasons.append(f"decoded text repeats word loop(s): {sorted(set(repeated_words))}")

    lower = stripped.lower()
    expected_terms = ("start time", "end time", "speaker", "content")
    if not any(term in lower for term in expected_terms):
        reasons.append("decoded text does not contain expected transcription keys")

    japanese_chars = re.findall(r"[\u3040-\u30ff\u3400-\u9fff]", stripped)
    if len(japanese_chars) < 5:
        reasons.append("decoded text does not contain enough Japanese characters")

    return not reasons, reasons


def greedy_decode_from_embeds(
    model: VibeVoiceASRForConditionalGeneration,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
) -> torch.Tensor:
    generated = []
    prefix_len = inputs_embeds.shape[1]

    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )
    past_key_values = outputs.past_key_values
    next_token = outputs.logits[:, -1].argmax(dim=-1)

    for step in range(max_new_tokens):
        generated.append(next_token)
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (attention_mask.shape[0], 1),
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                ),
            ],
            dim=1,
        )
        position_ids = attention_mask.long().cumsum(-1)[:, -1:] - 1
        cache_position = torch.tensor(
            [prefix_len + step],
            device=attention_mask.device,
        )
        outputs = model(
            input_ids=next_token[:, None],
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1].argmax(dim=-1)

    return torch.stack(generated, dim=1)


def main():
    args = parse_args()
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)

    print(f"Loading ASR tokenizer from {args.tokenizer_path}")
    tokenizer = VibeVoiceASRTextTokenizerFast.from_pretrained(
        args.tokenizer_path,
        local_files_only=True,
    )
    processor = VibeVoiceASRProcessor(
        tokenizer=tokenizer,
    )

    print(f"Loading ASR model from {args.model_path}")
    model = VibeVoiceASRForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device)
    model.eval()

    inputs = processor(
        audio=args.audio,
        sampling_rate=None,
        return_tensors="pt",
        padding=True,
        add_generation_prompt=True,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    acoustic_input_mask = inputs["acoustic_input_mask"].to(device)

    audio_features = np.load(Path(args.npz).expanduser())["output"]
    if audio_features.ndim != 3 or audio_features.shape[0] != input_ids.shape[0]:
        raise ValueError(
            f"Expected npz output shape [batch, frames, hidden], got {audio_features.shape}"
        )

    with torch.inference_mode():
        inputs_embeds = model.get_input_embeddings()(input_ids).clone()
        features = torch.from_numpy(audio_features).to(
            device=device,
            dtype=inputs_embeds.dtype,
        )

        if args.compare_encoder:
            seed = int(np.load(Path(args.npz).expanduser()).get("seed", np.array(0)))
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)

            direct_features = model.encode_speech(
                speech_tensors=inputs["speech_tensors"].to(device),
                speech_masks=inputs["speech_masks"].to(device),
            )
            direct_features = direct_features.reshape_as(features)
            max_abs_diff = (features.float() - direct_features.float()).abs().max().item()
            mean_abs_diff = (features.float() - direct_features.float()).abs().mean().item()
            allclose = torch.allclose(
                features.float(),
                direct_features.float(),
                rtol=1e-3,
                atol=1e-3,
            )
            print(
                "encoder_compare="
                f"allclose:{allclose} max_abs_diff:{max_abs_diff:.6g} "
                f"mean_abs_diff:{mean_abs_diff:.6g}"
            )

        expected_slots = int(acoustic_input_mask.sum().item())
        feature_slots = features.shape[0] * features.shape[1]
        if expected_slots != feature_slots:
            raise ValueError(
                f"Prompt has {expected_slots} speech slots, but npz has {feature_slots} frames"
            )
        if inputs_embeds.shape[-1] != features.shape[-1]:
            raise ValueError(
                f"Prompt hidden size {inputs_embeds.shape[-1]} does not match npz {features.shape[-1]}"
            )

        inputs_embeds[acoustic_input_mask] = features.reshape(-1, features.shape[-1])

        generated_ids = greedy_decode_from_embeds(
            model=model,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
        )

    generated_ids = generated_ids[0]
    text = processor.decode(generated_ids, skip_special_tokens=True)
    ok, reasons = looks_nongibberish(text)

    print(f"prompt_shape={tuple(input_ids.shape)}")
    print(f"spliced_features_shape={tuple(features.shape)}")
    print(f"generated_tokens={generated_ids.numel()}")
    print("--- decoded generated text ---")
    print(text)
    print("--- smoke result ---")
    if ok:
        print("PASS: first generated tokens look non-gibberish")
    else:
        print("FAIL: " + "; ".join(reasons))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
