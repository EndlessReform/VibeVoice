# Usage

For the current split ASR workflow, see [tools/README.md](../tools/README.md).

The preferred path is now the mixed vLLM Chat Completions flow:

```text
client:
  minimal audio encoder bundle
  projected audio rows
  Chat Completions text + prompt_embeds content part

server:
  vLLM with --enable-prompt-embeds
  server-side tokenizer and WTE
  transcript
```

The older full-prefix `inputs_embeds` tools still exist for parity/debugging,
but they are no longer the default runtime contract.
