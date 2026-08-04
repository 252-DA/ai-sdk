from typing import Any

from pydantic import BaseModel, Field

from ai_runtime.mcp.client import MCPToolClient


class QuizContextChunk(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    heading_path: list[str] = Field(default_factory=list)
    page_number: int | None = None
    rank: int
    score: float | None = None


class QuizContext(BaseModel):
    course_id: str
    lo_id: str
    lo_code: str
    lo_statement: str
    bloom_level: str
    assessment_style: str
    query: str
    chunks: list[QuizContextChunk]
    total_chars: int


class LearningContextClient:
    def __init__(self, mcp: MCPToolClient) -> None:
        self._mcp = mcp

    def retrieve_quiz_context(
        self,
        course_id: str,
        lo_code: str,
        query: str | None = None,
        bloom_level: str | None = None,
        assessment_style: str = "quiz",
        top_k: int = 5,
    ) -> QuizContext:
        arguments: dict[str, Any] = {
            "course_id": course_id,
            "lo_code": lo_code,
            "assessment_style": assessment_style,
            "top_k": top_k,
        }
        if query:
            arguments["query"] = query
        if bloom_level:
            arguments["bloom_level"] = bloom_level
        payload = self._mcp.call_tool("retrieve_quiz_context", arguments)
        return QuizContext.model_validate(payload)

