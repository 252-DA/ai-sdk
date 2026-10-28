from collections.abc import Iterator
from functools import cached_property

from ai_runtime.errors import ModelConfigurationError, ModelGenerationError
from ai_runtime.models import GenerationResult, ModelConfig, StreamItem, TextDelta, TokenUsage


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

    @staticmethod
    def _messages(prompt: str, system: str | None) -> list[dict[str, str]]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    @staticmethod
    def _usage(usage) -> TokenUsage:
        return TokenUsage(
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
        )

    def generate(
        self,
        prompt: str,
        system: str | None = None,
    ) -> GenerationResult:
        try:
            response = self._client.chat.completions.create(
                model=self._config.model,
                messages=self._messages(prompt, system),
                temperature=self._config.temperature,
            )
        except Exception as exc:
            raise ModelGenerationError(
                f"OpenAI-compatible generation failed for {self._config.model}"
            ) from exc

        text = response.choices[0].message.content if response.choices else None
        if not text or not text.strip():
            raise ModelGenerationError("OpenAI-compatible endpoint returned an empty response")
        return GenerationResult(
            text=text.strip(),
            model=self._config.model,
            usage=self._usage(getattr(response, "usage", None)),
        )

    def stream(
        self,
        prompt: str,
        system: str | None = None,
    ) -> Iterator[StreamItem]:
        try:
            chunks = self._client.chat.completions.create(
                model=self._config.model,
                messages=self._messages(prompt, system),
                temperature=self._config.temperature,
                stream=True,
                # Gói cuối mang số token (choices rỗng).
                stream_options={"include_usage": True},
            )
        except Exception as exc:
            raise ModelGenerationError(
                f"OpenAI-compatible generation failed for {self._config.model}"
            ) from exc

        parts: list[str] = []
        usage = None
        try:
            for chunk in chunks:
                if getattr(chunk, "usage", None) is not None:
                    usage = chunk.usage
                if not chunk.choices:
                    continue
                # Mô hình suy luận gửi reasoning_content riêng; chỉ lấy phần trả lời.
                piece = chunk.choices[0].delta.content
                if piece:
                    parts.append(piece)
                    yield TextDelta(piece)
        except Exception as exc:
            raise ModelGenerationError(
                f"OpenAI-compatible streaming failed for {self._config.model}"
            ) from exc
        finally:
            # Nơi gọi bỏ dở (người học đóng kết nối) thì đóng luôn kết nối tới provider.
            close = getattr(chunks, "close", None)
            if close is not None:
                close()

        text = "".join(parts).strip()
        if not text:
            raise ModelGenerationError("OpenAI-compatible endpoint returned an empty response")
        yield GenerationResult(text=text, model=self._config.model, usage=self._usage(usage))

