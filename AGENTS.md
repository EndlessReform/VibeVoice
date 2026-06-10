## About the repo

This is my personal fork of Microsoft's VibeVoice ASR model, geared for split client-server:
- Audio encoder on client hardware, encoding audio + text token input through WTE to the projected LM input
- LM backbone as a generic Qwen 2.5 decoder-only transformer, run in stock vLLM

## Tooling

Fork (unlike upstream) uses Astral UV (do _not_ use system Python!)

## Upstream libraries

- `docs/upstream/transformers`: Local shallow-cloned master branch of HF Transformers
