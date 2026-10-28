from collections.abc import Iterator
from functools import cached_property

from ai_runtime.errors import ModelConfigurationError, ModelGenerationError
from ai_runtime.models import GenerationResult, ModelConfig, StreamItem, TextDelta, TokenUsage


class GeminiModelClient:
    def __init__(self, config: ModelConfig) -> None:
        self._config = config

    @property
    def model_id(self) -> str:
        return self._config.model

    @cached_property
    def _sdk(self):
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise ModelConfigurationError(
                "Install ai-runtime-sdk[gemini] to use the Gemini provider"
            ) from exc
        return genai, types

    @cached_property
    def _client(self):
        if not self._config.api_key:
            raise ModelConfigurationError("Gemini API key is not configured")
        genai, types = self._sdk
        return genai.Client(
            api_key=self._config.api_key,
            http_options=types.HttpOptions(
                timeout=int(self._config.timeout_seconds * 1000)
            ),
        )

    @staticmethod
    def _usage(usage) -> TokenUsage:
        return TokenUsage(
            input_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None),
            total_tokens=getattr(usage, "total_token_count", None),
        )

    def _config_for(self, system: str | None):
        _, types = self._sdk
        return types.GenerateContentConfig(
            system_instruction=system,
            temperature=self._config.temperature,
        )

    def generate(
        self,
        prompt: str,
        system: str | None = None,
    ) -> GenerationResult:
        try:
            response = self._client.models.generate_content(
                model=self._config.model,
                contents=prompt,
                config=self._config_for(system),
            )
        except Exception as exc:
            raise ModelGenerationError(
                f"Gemini generation failed for {self._config.model}"
            ) from exc

        text = getattr(response, "text", None)
        if not text or not text.strip():
            raise ModelGenerationError("Gemini returned an empty response")
        return GenerationResult(
            text=text.strip(),
            model=self._config.model,
            usage=self._usage(getattr(response, "usage_metadata", None)),
        )

    def stream(
        self,
        prompt: str,
        system: str | None = None,
    ) -> Iterator[StreamItem]:
        try:
            chunks = self._client.models.generate_content_stream(
                model=self._config.model,
                contents=prompt,
                config=self._config_for(system),
            )
        except Exception as exc:
            raise ModelGenerationError(
                f"Gemini generation failed for {self._config.model}"
            ) from exc

        parts: list[str] = []
        usage = None
        try:
            for chunk in chunks:
                # Số token cộng dồn: gói sau ghi đè gói trước.
                if getattr(chunk, "usage_metadata", None) is not None:
                    usage = chunk.usage_metadata
                piece = getattr(chunk, "text", None)
                if piece:
                    parts.append(piece)
                    yield TextDelta(piece)
        except Exception as exc:
            raise ModelGenerationError(
                f"Gemini streaming failed for {self._config.model}"
            ) from exc
        finally:
            close = getattr(chunks, "close", None)
            if close is not None:
                close()

        text = "".join(parts).strip()
        if not text:
            raise ModelGenerationError("Gemini returned an empty response")
        yield GenerationResult(text=text, model=self._config.model, usage=self._usage(usage))

