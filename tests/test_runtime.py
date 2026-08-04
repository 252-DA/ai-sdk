from pydantic import BaseModel

from ai_runtime import AIRuntime, GenerationResult


class _Answer(BaseModel):
    answer: str


class _FakeModel:
    model_id = "fake-model"

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = 0

    def generate(self, prompt: str, system: str | None = None) -> GenerationResult:
        response = self.responses[self.calls]
        self.calls += 1
        return GenerationResult(text=response, model=self.model_id)


def test_generate_structured_validates_first_response():
    model = _FakeModel(['{"answer":"ok"}'])
    result = AIRuntime(model).generate_structured("answer", _Answer)

    assert result.value.answer == "ok"
    assert result.repaired is False
    assert model.calls == 1


def test_generate_structured_repairs_invalid_response_once():
    model = _FakeModel(["not-json", '{"answer":"repaired"}'])
    result = AIRuntime(model).generate_structured("answer", _Answer)

    assert result.value.answer == "repaired"
    assert result.repaired is True
    assert model.calls == 2

