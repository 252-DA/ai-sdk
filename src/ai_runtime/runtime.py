import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Generic, TypeVar

from pydantic import BaseModel, ValidationError

from ai_runtime.errors import StructuredOutputError
from ai_runtime.models import GenerationResult, ModelClient, StreamItem, TextDelta

SchemaT = TypeVar("SchemaT", bound=BaseModel)

_REPAIR_SYSTEM = "Repair malformed model output into strict JSON. Return JSON only."


@dataclass(frozen=True)
class StructuredGenerationResult(Generic[SchemaT]):
    value: SchemaT
    generation: GenerationResult
    repaired: bool = False


class AIRuntime:
    def __init__(self, model: ModelClient) -> None:
        self._model = model

    @property
    def model_id(self) -> str:
        return self._model.model_id

    def generate_text(
        self,
        prompt: str,
        system: str | None = None,
    ) -> GenerationResult:
        return self._model.generate(prompt=prompt, system=system)

    def stream_text(
        self,
        prompt: str,
        system: str | None = None,
    ) -> Iterator[StreamItem]:
        """
        Sinh theo luồng: các TextDelta rồi đúng một GenerationResult ở cuối.

        Provider chưa hỗ trợ luồng thì sinh một lần và trả toàn văn thành một mẩu duy nhất,
        để nơi gọi chỉ cần một cách đọc.
        """
        stream = getattr(self._model, "stream", None)
        if stream is None:
            result = self.generate_text(prompt=prompt, system=system)
            yield TextDelta(result.text)
            yield result
            return
        yield from stream(prompt=prompt, system=system)

    def generate_structured(
        self,
        prompt: str,
        schema: type[SchemaT],
        system: str | None = None,
        repair: bool = True,
    ) -> StructuredGenerationResult[SchemaT]:
        generation = self.generate_text(prompt=prompt, system=system)
        parsed = self._parse(generation.text, schema)
        if parsed is not None:
            return StructuredGenerationResult(value=parsed, generation=generation)
        if not repair:
            raise StructuredOutputError(
                f"{self.model_id} returned output that does not match {schema.__name__}"
            )

        repair_prompt = (
            "Rewrite the content below to match this JSON Schema exactly.\n"
            f"{json.dumps(schema.model_json_schema(), ensure_ascii=False)}\n\n"
            "Invalid content:\n"
            f"{generation.text}"
        )
        repaired_generation = self.generate_text(
            prompt=repair_prompt,
            system=_REPAIR_SYSTEM,
        )
        parsed = self._parse(repaired_generation.text, schema)
        if parsed is None:
            raise StructuredOutputError(
                f"{self.model_id} returned invalid {schema.__name__} after one repair"
            )
        return StructuredGenerationResult(
            value=parsed,
            generation=repaired_generation,
            repaired=True,
        )

    @staticmethod
    def _parse(raw_text: str, schema: type[SchemaT]) -> SchemaT | None:
        text = raw_text.strip()
        fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
        if fenced:
            text = fenced.group(1).strip()
        try:
            return schema.model_validate(json.loads(text))
        except (json.JSONDecodeError, ValidationError):
            return None

