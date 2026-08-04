from dataclasses import replace

from ai_runtime.models import ModelConfig
from ai_runtime.providers.openai_compatible import OpenAICompatibleModelClient


DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class DeepSeekModelClient(OpenAICompatibleModelClient):
    """DeepSeek adapter backed by its OpenAI-compatible API."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(
            replace(
                config,
                base_url=config.base_url or DEEPSEEK_BASE_URL,
            )
        )
