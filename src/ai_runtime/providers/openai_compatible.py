from functools import cached_property

from ai_runtime.errors import ModelConfigurationError, ModelGenerationError
from ai_runtime.models import GenerationResult, ModelConfig, TokenUsage


class OpenAICompatibleModelClient:
    """Works with OpenAI and compatible endpoints such as LiteLLM/DeepSeek/Qwen."""

    def __init__(self, config: ModelConfig) -> None:
        self._config = config

    @property
    def model_id(self) -> str:
        return self._config.model

    @cached_property
    def _client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ModelConfigurationError(
                "Install ai-runtime-sdk[openai] to use an OpenAI-compatible provider"
            ) from exc
        if not self._config.api_key:
            raise ModelConfigurationError("OpenAI-compatible API key is not configured")
        return OpenAI(
            api_key=self._config.api_key,
            base_url=self._config.base_url,
            timeout=self._config.timeout_seconds,
            max_retries=self._config.max_retries,
        )

    def generate(
        self,
        prompt: str,
        system: str | None = None,
    ) -> GenerationResult:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            response = self._client.chat.completions.create(
                model=self._config.model,
                messages=messages,
                temperature=self._config.temperature,
            )
        except Exception as exc:
            raise ModelGenerationError(
                f"OpenAI-compatible generation failed for {self._config.model}"
            ) from exc

        text = response.choices[0].message.content if response.choices else None
        if not text or not text.strip():
            raise ModelGenerationError("OpenAI-compatible endpoint returned an empty response")
        usage = getattr(response, "usage", None)
        return GenerationResult(
            text=text.strip(),
            model=self._config.model,
            usage=TokenUsage(
                input_tokens=getattr(usage, "prompt_tokens", None),
                output_tokens=getattr(usage, "completion_tokens", None),
                total_tokens=getattr(usage, "total_tokens", None),
            ),
        )

