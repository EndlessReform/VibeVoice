import torch
from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRConfig
from vibevoice.modular.modular_vibevoice_tokenizer import VibeVoiceSemanticTokenizerModel

try:
    print("Loading config...")
    config = VibeVoiceASRConfig.from_pretrained("microsoft/VibeVoice-ASR", trust_remote_code=True)
    semantic_config = config.semantic_tokenizer_config
    print("Initializing semantic tokenizer model...")
    model = VibeVoiceSemanticTokenizerModel(semantic_config)
    params = sum(p.numel() for p in model.parameters())
    print(f"Semantic encoder parameters: {params:,}")
except Exception as e:
    import traceback
    traceback.print_exc()
