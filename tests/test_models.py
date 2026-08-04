from ai_runtime import ModelConfig, build_model_client
from ai_runtime.providers.deepseek import DEEPSEEK_BASE_URL, DeepSeekModelClient


def _deepseek_config(base_url: str | None = None) -> ModelConfig:
    return ModelConfig(
        provider="deepseek",
        model="deepseek-chat",
        api_key="test-key",
        base_url=base_url,
    )


def test_builds_deepseek_client_with_default_base_url():
    client = build_model_client(_deepseek_config())

    assert isinstance(client, DeepSeekModelClient)
    assert client._config.base_url == DEEPSEEK_BASE_URL


def test_deepseek_client_preserves_custom_base_url():
    client = build_model_client(_deepseek_config("https://deepseek.example/v1"))

    assert client._config.base_url == "https://deepseek.example/v1"
