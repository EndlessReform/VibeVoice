# VibeVoice-ASR

[![Hugging Face](https://img.shields.io/badge/HuggingFace-Collection-orange?logo=huggingface)](https://huggingface.co/microsoft/VibeVoice-ASR)
[![Live Playground](https://img.shields.io/badge/Live-Playground-green?logo=gradio)](https://aka.ms/vibevoice-asr)

**VibeVoice-ASR** is a unified speech-to-text model designed to handle **60-minute long-form audio** in a single pass, generating structured transcriptions containing **Who (Speaker), When (Timestamps), and What (Content)**, with support for **Customized Hotwords** and over **50 languages**.

**Model:** [VibeVoice-ASR-7B](https://huggingface.co/microsoft/VibeVoice-ASR)<br>
**Demo:** [VibeVoice-ASR-Demo](https://aka.ms/vibevoice-asr)<br>
**Report:** [VibeVoice-ASR-Report](https://arxiv.org/pdf/2601.18184)<br>
**Finetuning:** [finetune-guide](../finetuning-asr/README.md)<br>
**vLLM:** [vLLM-asr](./vibevoice-vllm-asr.md)<br>
**Transformers:** [VibeVoice-ASR-HF](https://huggingface.co/microsoft/VibeVoice-ASR-HF)<br>


## 🔥 Key Features

- **🕒 60-minute Single-Pass Processing**:
  Unlike conventional ASR models that slice audio into short chunks (often losing global context), VibeVoice ASR accepts up to **60 minutes** of continuous audio input within 64K token length. This ensures consistent speaker tracking and semantic coherence across the entire hour.

- **👤 Customized Hotwords**:
  Users can provide customized hotwords (e.g., specific names, technical terms, or background info) to guide the recognition process, significantly improving accuracy on domain-specific content.

- **📝 Rich Transcription (Who, When, What)**:
  The model jointly performs ASR, diarization, and timestamping, producing a structured output that indicates *who* said *what* and *when*.
  
- **🌍 Multilingual & Code-Switching Support**:
  It supports over 50 languages, requires no explicit language setting, and natively handles code-switching within and across utterances. See the [Language distribution](#language-distribution).


## 🏗️ Model Architecture

<p align="center">
  <img src="../Figures/VibeVoice_ASR_archi.png" alt="VibeVoice ASR Architecture" width="80%">
</p>

# Demo

<div align="center" id="vibevoice-asr">

https://github.com/user-attachments/assets/acde5602-dc17-4314-9e3b-c630bc84aefa

</div>

## Evaluation
<p align="center">
  <img src="../Figures/DER.jpg" alt="DER" width="50%"><br>
  <img src="../Figures/cpWER.jpg" alt="cpWER" width="50%"><br>
  <img src="../Figures/tcpWER.jpg" alt="tcpWER" width="50%">
</p>



## Installation
We recommend using NVIDIA Deep Learning Container to manage the CUDA environment. 

1. Launch docker
```bash
# NVIDIA PyTorch Container 24.07 ~ 25.12 verified. 
# Previous versions are also compatible.
sudo docker run --privileged --net=host --ipc=host --ulimit memlock=-1:-1 --ulimit stack=-1:-1 --gpus all --rm -it  nvcr.io/nvidia/pytorch:25.12-py3

## If flash attention is not included in your docker environment, you need to install it manually
## Refer to https://github.com/Dao-AILab/flash-attention for installation instructions
# pip install flash-attn --no-build-isolation
```

2. Install from github 
```bash
git clone https://github.com/microsoft/VibeVoice.git
cd VibeVoice

pip install -e .
```

## Usages

### Usage 1: Launch Gradio demo
```bash
apt update && apt install ffmpeg -y # for demo

python demo/vibevoice_asr_gradio_demo.py --model_path microsoft/VibeVoice-ASR --share
```

### Usage 2: Inference from files directly
```bash
python demo/vibevoice_asr_inference_from_file.py --model_path microsoft/VibeVoice-ASR --audio_files [add an audio path here] 
```

### Usage 3: Export the ASR text tokenizer for split deployments

For deployments that run the audio encoder separately from the LM-specific
parts of the model, export the merged ASR text tokenizer once and load the
saved tokenizer files in the LM process. This follows the same path used by
`demo/vibevoice_asr_gradio_demo.py`: `VibeVoiceASRProcessor.from_pretrained(...)`
loads `VibeVoiceASRTextTokenizerFast`, which applies the ASR speech-token merge
implemented in `vibevoice/modular/modular_vibevoice_text_tokenizer.py`.

```bash
python tools/export_asr_text_tokenizer.py \
  --model-path microsoft/VibeVoice-ASR \
  --output-dir ./vibevoice-asr-text-tokenizer
```

If you already know the base LM tokenizer, pass it explicitly:

```bash
python tools/export_asr_text_tokenizer.py \
  --model-path microsoft/VibeVoice-ASR \
  --language-model Qwen/Qwen2.5-7B \
  --output-dir ./vibevoice-asr-text-tokenizer
```

The script only loads tokenizer assets; it does not load the ASR model weights.
The exported tokenizer includes the ASR speech boundary and pad tokens used for
audio placeholders: `<|object_ref_start|>`, `<|box_start|>`, and
`<|object_ref_end|>`.

By default, the script resolves the language-model tokenizer from
`preprocessor_config.json` when that file is present. If the released
`microsoft/VibeVoice-ASR` checkpoint does not include that file, the script
inspects `config.json` and recognizes the released ASR decoder shape as
`Qwen/Qwen2.5-7B`. The `--language-model` flag is only an override for custom or
experimental checkpoints.

#### Building the exact ASR prefix without `transformers`

If your LM process uses the exported `tokenizer.json` directly with the
low-level Hugging Face `tokenizers` library, the tokenizer will know all token
IDs, but it will not run `transformers` conveniences such as
`apply_chat_template`. In that setup the client must render the chat text
itself before calling `Tokenizer.encode(...)`.

There is no hidden chat-template magic beyond the literal strings below. With
hotwords or other context, a full valid ASR prefix is:

```text
<|im_start|>system
You are a helpful assistant that transcribes audio input into text output in JSON format.<|im_end|>
<|im_start|>user
<|object_ref_start|><|box_start|>...<|box_start|><|object_ref_end|>
This is a {duration_seconds:.2f} seconds audio, with extra info: {hotwords_or_context}

Please transcribe it with these keys: Start time, End time, Speaker ID, Content<|im_end|>
<|im_start|>assistant
```

Without hotwords or context, use this user suffix instead:

```text
This is a {duration_seconds:.2f} seconds audio, please transcribe it with these keys: Start time, End time, Speaker ID, Content
```

Important details:

- The prefix uses two independent chat blocks: one `system` block followed by
  one `user` block, followed by the open assistant generation marker.
- With `add_generation_prompt=True`, the ASR processor appends
  `<|im_start|>assistant\n`; generation starts after that marker.
- The audio span is represented in text as one `<|object_ref_start|>`, followed
  by `ceil(num_audio_samples / 3200)` copies of `<|box_start|>`, followed by
  one `<|object_ref_end|>`.
- `num_audio_samples` is the length of the 24 kHz mono audio seen by the model.
  For example, one second of audio is `24000` samples, so it uses
  `ceil(24000 / 3200) = 8` `<|box_start|>` placeholders.
- The newline after `<|object_ref_end|>` is required.
- Hotwords, speaker names, titles, or other hints go in the `extra info` text.
  If there is no extra info, use the shorter no-context sentence shown below.
- The `acoustic_input_mask` is `true` only at positions whose token ID equals
  the `<|box_start|>` / `speech_pad_id` token ID.

The following builder mirrors `VibeVoiceASRProcessor` and the round-trip test:

```python
import math
from tokenizers import Tokenizer

SYSTEM_PROMPT = (
    "You are a helpful assistant that transcribes audio input into text output "
    "in JSON format."
)
SHOW_KEYS = ["Start time", "End time", "Speaker ID", "Content"]
SAMPLE_RATE = 24000
SPEECH_TOK_COMPRESS_RATIO = 3200

SPEECH_START = "<|object_ref_start|>"
SPEECH_PAD = "<|box_start|>"
SPEECH_END = "<|object_ref_end|>"


def render_asr_chat(role: str, content: str) -> str:
    return f"<|im_start|>{role}\n{content}<|im_end|>\n"


def build_asr_prefix(num_audio_samples: int, extra_info: str | None = None) -> str:
    duration_seconds = num_audio_samples / SAMPLE_RATE
    num_audio_tokens = math.ceil(num_audio_samples / SPEECH_TOK_COMPRESS_RATIO)

    system_text = render_asr_chat("system", SYSTEM_PROMPT)
    audio_text = SPEECH_START + (SPEECH_PAD * num_audio_tokens) + SPEECH_END

    if extra_info and extra_info.strip():
        user_suffix = (
            f"This is a {duration_seconds:.2f} seconds audio, with extra info: "
            f"{extra_info.strip()}\n\nPlease transcribe it with these keys: "
            + ", ".join(SHOW_KEYS)
        )
    else:
        user_suffix = (
            f"This is a {duration_seconds:.2f} seconds audio, "
            "please transcribe it with these keys: "
            + ", ".join(SHOW_KEYS)
        )

    user_text = render_asr_chat("user", audio_text + "\n" + user_suffix)
    return system_text + user_text + "<|im_start|>assistant\n"


tokenizer = Tokenizer.from_file("./vibevoice-asr-text-tokenizer/tokenizer.json")
prefix = build_asr_prefix(
    num_audio_samples=24000,
    extra_info="Hotwords: VibeVoice, Qwen2.5, CUDA graphs.",
)
input_ids = tokenizer.encode(prefix, add_special_tokens=True).ids
speech_pad_id = tokenizer.token_to_id(SPEECH_PAD)
acoustic_input_mask = [token_id == speech_pad_id for token_id in input_ids]
```

For a one-second audio file with those hotwords, the exact rendered prefix starts
like this:

```text
<|im_start|>system
You are a helpful assistant that transcribes audio input into text output in JSON format.<|im_end|>
<|im_start|>user
<|object_ref_start|><|box_start|><|box_start|><|box_start|><|box_start|><|box_start|><|box_start|><|box_start|><|box_start|><|object_ref_end|>
This is a 1.00 seconds audio, with extra info: Hotwords: VibeVoice, Qwen2.5, CUDA graphs.

Please transcribe it with these keys: Start time, End time, Speaker ID, Content<|im_end|>
<|im_start|>assistant
```

When using the exported tokenizer with bare `tokenizers`, that is the only
client-side formatting required: render the literal prefix, tokenize it, replace
or merge the embeddings at the `<|box_start|>` positions with the encoder output,
and pass the resulting LM prefix to the model.

To verify the exported tokenizer against the existing `VibeVoiceASRProcessor`
path, run the round-trip check. It compares the processor-built input IDs with
IDs produced by loading the exported `tokenizer.json` directly through the
low-level Hugging Face `tokenizers` library, then checks the decoded UTF-8 prompt
bytes match exactly.

```bash
python tools/test_asr_tokenizer_roundtrip.py \
  --model-path microsoft/VibeVoice-ASR \
  --language-model Qwen/Qwen2.5-7B
```


## Finetuning
LoRA (Low-Rank Adaptation) fine-tuning is supported. See [Finetuning](../finetuning-asr/README.md) for detailed guide.



## Results

### Multilingual
| Dataset        | Language  | DER  | cpWER | tcpWER | WER  |
|----------------|-----------|------|-------|--------|------|
| MLC-Challenge  | English   | 4.28 | 11.48 | 13.02  | 7.99  |
| MLC-Challenge  | French    | 3.80 | 18.80 | 19.64  | 15.21 |
| MLC-Challenge  | German    | 1.04 | 17.10 | 17.26  | 16.30 |
| MLC-Challenge  | Italian   | 2.08 | 15.76 | 15.91  | 13.91 |
| MLC-Challenge  | Japanese  | 0.82 | 15.33 | 15.41  | 14.69 |
| MLC-Challenge  | Korean    | 4.52 | 15.35 | 16.07  | 9.65  |
| MLC-Challenge  | Portuguese| 7.98 | 29.91 | 31.65  | 21.54 |
| MLC-Challenge  | Russian   | 0.90 | 12.94 | 12.98  | 12.40 |
| MLC-Challenge  | Spanish   | 2.67 | 10.51 | 11.71  | 8.04  |
| MLC-Challenge  | Thai      | 4.09 | 14.91 | 15.57  | 13.61 |
| MLC-Challenge  | Vietnamese| 0.16 | 14.57 | 14.57  | 14.43 |

---

| Dataset        | Language  | DER  | cpWER | tcpWER | WER  |
|----------------|-----------|------|-------|--------|------|
| AISHELL-4      | Chinese   | 6.77 | 24.99 | 25.35  | 21.40 |
| AMI-IHM        | English   | 11.92| 20.41 | 20.82  | 18.81 |
| AMI-SDM        | English   | 13.43| 28.82 | 29.80  | 24.65 |
| AliMeeting     | Chinese   | 10.92| 29.33 | 29.51  | 27.40 |
| MLC-Challenge  | Average   | 3.42 | 14.81 | 15.66  | 12.07|


## Language Distribution
<p align="center">
  <img src="../Figures/language_distribution_horizontal.png" alt="Language Distribution" width="80%">
</p>

## 📄 License

This project is licensed under the [MIT License](../LICENSE).
