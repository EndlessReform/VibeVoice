import torch

from vibevoice.modular.configuration_vibevoice import VibeVoiceASRConfig, VibeVoiceConfig


def test_vibevoice_config_serializes_dtype_kwargs():
    for config_class in (VibeVoiceASRConfig, VibeVoiceConfig):
        config = config_class(dtype=torch.bfloat16)

        assert config.to_dict()["dtype"] == "bfloat16"
        assert '"dtype": "bfloat16"' in config.to_json_string()


def test_vibevoice_config_serializes_nested_dtype_values():
    config = VibeVoiceASRConfig(
        acoustic_tokenizer_config={"dtype": torch.float32},
        semantic_tokenizer_config={"dtype": torch.bfloat16},
    )
    config_dict = config.to_dict()

    assert config_dict["acoustic_tokenizer_config"]["dtype"] == "float32"
    assert config_dict["semantic_tokenizer_config"]["dtype"] == "bfloat16"
    assert '"dtype": "float32"' in config.to_json_string()
