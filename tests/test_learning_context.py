from ai_runtime.mcp import LearningContextClient


class _FakeMCP:
    def __init__(self) -> None:
        self.call = None

    def call_tool(self, name, arguments):
        self.call = (name, arguments)
        return {
            "course_id": "CO3115",
            "lo_id": "CO3115:L.O.3.1",
            "lo_code": "L.O.3.1",
            "lo_statement": "Phân tích yêu cầu",
            "bloom_level": "analyze",
            "assessment_style": "quiz",
            "query": "Phân tích yêu cầu",
            "chunks": [
                {
                    "chunk_id": "chunk-1",
                    "document_id": "doc-1",
                    "content": "Context",
                    "heading_path": ["Chương 3"],
                    "page_number": 4,
                    "rank": 1,
                    "score": 0.9,
                }
            ],
            "total_chars": 7,
        }


def test_learning_context_client_calls_expected_mcp_tool():
    mcp = _FakeMCP()
    client = LearningContextClient(mcp)

    result = client.retrieve_quiz_context(
        course_id="CO3115",
        lo_code="L.O.3.1",
        bloom_level="analyze",
    )

    assert result.chunks[0].chunk_id == "chunk-1"
    assert mcp.call == (
        "retrieve_quiz_context",
        {
            "course_id": "CO3115",
            "lo_code": "L.O.3.1",
            "assessment_style": "quiz",
            "top_k": 5,
            "bloom_level": "analyze",
        },
    )
