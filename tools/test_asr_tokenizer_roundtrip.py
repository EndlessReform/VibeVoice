#!/usr/bin/env python3
"""
Round-trip the exported VibeVoice-ASR tokenizer against the existing processor.

This is intentionally one-off development tooling. It verifies the custom
tokenizer path used to create vendored split-checkpoint tokenizer files, while
runtime code loads those exported files with vanilla tokenizer APIs.

Reference side:
    VibeVoiceASRProcessor + VibeVoiceASRTextTokenizerFast from transformers.

Standalone side:
    tokenizers.Tokenizer loaded directly from the exported tokenizer.json.

The test verifies the exact input IDs and the decoded UTF-8 prompt bytes for
several dummy-audio prefixes. It also checks the audio placeholder span count,
but does not attempt to compare model-side encoded audio embeddings.
"""

import argparse
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from tokenizers import Tokenizer

from export_asr_text_tokenizer import export_tokenizer
from vibevoice.modular.modular_vibevoice_text_tokenizer import (
    VibeVoiceASRTextTokenizerFast,
)
from vibevoice.processor.vibevoice_asr_processor import (
    SYSTEM_PROMPT,
    VibeVoiceASRProcessor,
)


DEFAULT_LANGUAGE_MODEL = "Qwen/Qwen2.5-7B"
DEFAULT_MODEL_PATH = "microsoft/VibeVoice-ASR"
SAMPLE_RATE = 24000
SPEECH_TOK_COMPRESS_RATIO = 3200
SHOW_KEYS = ["Start time", "End time", "Speaker ID", "Content"]


@dataclass(frozen=True)
class PrefixCase:
    name: str
    samples: int
    context_info: str | None = None


CASES = (
    PrefixCase("half_second_no_context", SAMPLE_RATE // 2),
    PrefixCase(
        "one_second_hotwords",
        SAMPLE_RATE,
        "Hotwords: VibeVoice, Qwen2.5, CUDA graphs.",
    ),
    PrefixCase(
        "unicode_context",
        (2 * SAMPLE_RATE) + 123,
        "Names: 小林さん, 佐藤さん; terms: diarization, tcpWER, cpWER.",
    ),
    PrefixCase(
        "newline_context",
        (5 * SAMPLE_RATE) + 17,
        "Meeting notes:\n- speaker Alice\n- speaker Bob\nReturn JSON exactly.",
    ),
    PrefixCase(
        "long_audio_prefix",
        (61 * SAMPLE_RATE) + 1,
        "Long-form prefix check with punctuation: []{}:,.;!? and quotes \"like this\".",
    ),
)


def render_asr_chat(messages: Iterable[dict[str, str]], add_generation_prompt: bool = False) -> str:
    rendered = "".join(
        f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
        for message in messages
    )
    if add_generation_prompt:
        rendered += "<|im_start|>assistant\n"
    return rendered


def build_user_suffix(samples: int, context_info: str | None) -> str:
    audio_duration = samples / SAMPLE_RATE
    if context_info and context_info.strip():
        return (
            f"This is a {audio_duration:.2f} seconds audio, with extra info: "
            f"{context_info.strip()}\n\nPlease transcribe it with these keys: "
            + ", ".join(SHOW_KEYS)
        )
    return (
        f"This is a {audio_duration:.2f} seconds audio, please transcribe it with these keys: "
        + ", ".join(SHOW_KEYS)
    )


def build_rendered_prompt(
    tokenizer: VibeVoiceASRTextTokenizerFast,
    case: PrefixCase,
) -> tuple[str, str, int]:
    vae_tok_len = math.ceil(case.samples / SPEECH_TOK_COMPRESS_RATIO)

    speech_start = tokenizer.convert_ids_to_tokens(tokenizer.speech_start_id)
    speech_pad = tokenizer.convert_ids_to_tokens(tokenizer.speech_pad_id)
    speech_end = tokenizer.convert_ids_to_tokens(tokenizer.speech_end_id)

    system_text = render_asr_chat([{"role": "system", "content": SYSTEM_PROMPT}])
    user_input = (
        speech_start
        + (speech_pad * vae_tok_len)
        + speech_end
        + "\n"
        + build_user_suffix(case.samples, case.context_info)
    )
    user_text = render_asr_chat(
        [{"role": "user", "content": user_input}],
        add_generation_prompt=True,
    )

    return system_text, user_text, vae_tok_len


def first_mismatch(left: list[int], right: list[int]) -> str:
    for index, (left_id, right_id) in enumerate(zip(left, right)):
        if left_id != right_id:
            return f"index {index}: reference={left_id}, standalone={right_id}"
    return f"length mismatch: reference={len(left)}, standalone={len(right)}"


def assert_same_bytes(label: str, left: str, right: str) -> None:
    left_bytes = left.encode("utf-8")
    right_bytes = right.encode("utf-8")
    if left_bytes != right_bytes:
        mismatch = next(
            (
                idx
                for idx, (left_byte, right_byte) in enumerate(zip(left_bytes, right_bytes))
                if left_byte != right_byte
            ),
            min(len(left_bytes), len(right_bytes)),
        )
        raise AssertionError(
            f"{label} byte mismatch at byte {mismatch}: "
            f"reference_len={len(left_bytes)}, standalone_len={len(right_bytes)}"
        )


def reference_ids(
    tokenizer: VibeVoiceASRTextTokenizerFast,
    processor: VibeVoiceASRProcessor,
    case: PrefixCase,
) -> tuple[list[int], list[int]]:
    audio = np.zeros(case.samples, dtype=np.float32)
    encoded = processor(
        audio=audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors=None,
        padding=True,
        use_streaming=False,
        context_info=case.context_info,
    )
    return list(encoded["input_ids"]), list(encoded["acoustic_input_mask"])


def run_case(
    case: PrefixCase,
    reference_tokenizer: VibeVoiceASRTextTokenizerFast,
    processor: VibeVoiceASRProcessor,
    bare_tokenizer: Tokenizer,
) -> None:
    ref_ids, ref_audio_mask = reference_ids(reference_tokenizer, processor, case)
    system_text, user_text, expected_audio_tokens = build_rendered_prompt(
        reference_tokenizer,
        case,
    )

    standalone_ids = (
        bare_tokenizer.encode(system_text, add_special_tokens=True).ids
        + bare_tokenizer.encode(user_text, add_special_tokens=True).ids
    )
    standalone_full_ids = bare_tokenizer.encode(
        system_text + user_text,
        add_special_tokens=True,
    ).ids

    if ref_ids != standalone_ids:
        raise AssertionError(
            f"{case.name}: segmented ID mismatch at {first_mismatch(ref_ids, standalone_ids)}"
        )
    if ref_ids != standalone_full_ids:
        raise AssertionError(
            f"{case.name}: full-prompt ID mismatch at {first_mismatch(ref_ids, standalone_full_ids)}"
        )

    expected_text = system_text + user_text
    reference_decoded = reference_tokenizer.decode(
        ref_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    standalone_decoded = bare_tokenizer.decode(standalone_ids, skip_special_tokens=False)

    assert_same_bytes(f"{case.name}: rendered vs reference decode", expected_text, reference_decoded)
    assert_same_bytes(f"{case.name}: reference vs standalone decode", reference_decoded, standalone_decoded)

    speech_pad_count = ref_ids.count(reference_tokenizer.speech_pad_id)
    audio_mask_count = sum(1 for value in ref_audio_mask if value)
    if speech_pad_count != expected_audio_tokens:
        raise AssertionError(
            f"{case.name}: expected {expected_audio_tokens} speech pad tokens, got {speech_pad_count}"
        )
    if audio_mask_count != expected_audio_tokens:
        raise AssertionError(
            f"{case.name}: expected {expected_audio_tokens} audio-mask entries, got {audio_mask_count}"
        )

    print(
        f"PASS {case.name}: ids={len(ref_ids)} audio_tokens={expected_audio_tokens} "
        f"bytes={len(reference_decoded.encode('utf-8'))}"
    )


def run_roundtrip(args: argparse.Namespace) -> None:
    export_dir_context = (
        tempfile.TemporaryDirectory(prefix="vibevoice_asr_tokenizer_")
        if args.export_dir is None
        else None
    )
    export_dir = Path(args.export_dir or export_dir_context.name)

    try:
        export_tokenizer(
            model_path=args.model_path,
            output_dir=str(export_dir),
            language_model=args.language_model,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
        )

        tokenizer_json = export_dir / "tokenizer.json"
        if not tokenizer_json.exists():
            raise FileNotFoundError(f"exported tokenizer.json not found: {tokenizer_json}")

        reference_tokenizer = VibeVoiceASRTextTokenizerFast.from_pretrained(
            args.language_model,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
        )
        processor = VibeVoiceASRProcessor(
            tokenizer=reference_tokenizer,
            speech_tok_compress_ratio=SPEECH_TOK_COMPRESS_RATIO,
            target_sample_rate=SAMPLE_RATE,
            normalize_audio=False,
        )
        bare_tokenizer = Tokenizer.from_file(str(tokenizer_json))

        for case in CASES:
            run_case(case, reference_tokenizer, processor, bare_tokenizer)

        print("All ASR tokenizer round-trip checks passed byte-for-byte.")
    finally:
        if export_dir_context is not None:
            export_dir_context.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare VibeVoice-ASR transformer tokenization with bare tokenizers output."
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="VibeVoice-ASR model path used only to resolve preprocessor_config.json if present.",
    )
    parser.add_argument(
        "--language-model",
        default=DEFAULT_LANGUAGE_MODEL,
        help="Base LM tokenizer repo/path.",
    )
    parser.add_argument(
        "--export-dir",
        default=None,
        help="Optional directory for the exported standalone tokenizer. Defaults to a temp dir.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Forward trust_remote_code=True when loading the tokenizer.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only local Hugging Face cache files when resolving repo IDs.",
    )
    args = parser.parse_args()
    run_roundtrip(args)


if __name__ == "__main__":
    main()
