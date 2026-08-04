from dataclasses import dataclass
from typing import Literal, Protocol

from ai_runtime.errors import ModelConfigurationError


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class GenerationResult:
    text: str
    model: str
    usage: TokenUsage = TokenUsage()


class ModelClient(Protocol):
    @property
    def model_id(self) -> str:
        ...

    def generate(
        self,
        prompt: str,
        system: str | None = None,
    ) -> GenerationResult:
        ...


@dataclass(frozen=True)
class ModelConfig:
    provider: Literal["gemini", "openai-compatible", "deepseek"]
    model: str
    api_key: str | None
    base_url: str | None = None
    temperature: float = 0.2
    timeout_seconds: float = 60
    max_retries: int = 2


def build_model_client(config: ModelConfig) -> ModelClient:
    if config.provider == "gemini":
        from ai_runtime.providers.gemini import GeminiModelClient

        return GeminiModelClient(config)
    if config.provider == "openai-compatible":
        from ai_runtime.providers.openai_compatible import OpenAICompatibleModelClient

        return OpenAICompatibleModelClient(config)
    if config.provider == "deepseek":
        from ai_runtime.providers.deepseek import DeepSeekModelClient

        return DeepSeekModelClient(config)
    raise ModelConfigurationError(f"Unsupported model provider: {config.provider!r}")
