from types import SimpleNamespace

import pytest

from ai_runtime import AIRuntime, GenerationResult, ModelConfig, TextDelta
from ai_runtime.errors import ModelGenerationError
from ai_runtime.providers.openai_compatible import OpenAICompatibleModelClient


class _PlainModel:
    model_id = "plain"

    def generate(self, prompt: str, system: str | None = None) -> GenerationResult:
        return GenerationResult(text="whole answer", model=self.model_id)


def test_stream_text_falls_back_to_one_piece_when_provider_cannot_stream():
    items = list(AIRuntime(_PlainModel()).stream_text("q"))

    assert items[0] == TextDelta("whole answer")
    assert isinstance(items[-1], GenerationResult)
    assert items[-1].text == "whole answer"


class _FakeStream:
    def __init__(self, chunks, fail_after: int | None = None) -> None:
        self._chunks = chunks
        self._fail_after = fail_after
        self.closed = False

    def __iter__(self):
        for i, chunk in enumerate(self._chunks):
            if self._fail_after is not None and i == self._fail_after:
                raise ConnectionError("dropped")
            yield chunk

    def close(self) -> None:
        self.closed = True


def _chunk(content=None, usage=None):
    choices = [] if content is None else [SimpleNamespace(delta=SimpleNamespace(content=content))]
    return SimpleNamespace(choices=choices, usage=usage)


def _client_with(stream: _FakeStream) -> tuple[OpenAICompatibleModelClient, dict]:
    client = OpenAICompatibleModelClient(ModelConfig(provider="openai-compatible", model="m", api_key="k"))
    seen: dict = {}

    def create(**kwargs):
        seen.update(kwargs)
        return stream

    client.__dict__["_client"] = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return client, seen


def test_openai_stream_yields_deltas_then_result_with_usage():
    stream = _FakeStream([
        _chunk("Xin "),
        _chunk(None),  # gói chỉ có reasoning/role
        _chunk("chào"),
        _chunk(None, usage=SimpleNamespace(prompt_tokens=7, completion_tokens=2, total_tokens=9)),
    ])
    client, seen = _client_with(stream)

    items = list(client.stream("q", system="s"))

    assert seen["stream"] is True
    assert [i.text for i in items[:-1]] == ["Xin ", "chào"]
    result = items[-1]
    assert isinstance(result, GenerationResult)
    assert result.text == "Xin chào"
    assert result.usage.input_tokens == 7 and result.usage.output_tokens == 2
    assert stream.closed


def test_openai_stream_failure_mid_way_raises_generation_error_and_closes():
    stream = _FakeStream([_chunk("a"), _chunk("b")], fail_after=1)
    client, _ = _client_with(stream)

    with pytest.raises(ModelGenerationError):
        list(client.stream("q"))
    assert stream.closed


def test_openai_stream_closes_provider_when_caller_stops_early():
    stream = _FakeStream([_chunk("a"), _chunk("b"), _chunk("c")])
    client, _ = _client_with(stream)

    items = client.stream("q")
    next(items)
    items.close()

    assert stream.closed

