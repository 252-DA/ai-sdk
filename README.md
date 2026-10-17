# ai-runtime-sdk

Thin internal SDK shared by Python workers and AI services.

It keeps three concerns in one reusable package:

- provider-neutral text generation (`gemini` and OpenAI-compatible endpoints);
- validated structured output with one repair attempt;
- MCP Streamable HTTP tool calls.

It intentionally contains no LMS or quiz persistence rules. Those remain in
the calling application's use cases.

## Install

```bash
uv add --editable "../ai-sdk[gemini]"
```

The default Gemini profile is `gemini-flash-lite`, which calls
`gemini-3.1-flash-lite` for enrichment, quiz verification and tutoring.
Select a profile with `LLM_PROFILE__<TASK>` or `LLM_PROFILE`; `gemini-flash`
remains available for Gemini 3 Flash Preview.

## Generate structured output

```python
from pydantic import BaseModel

from ai_runtime import AIRuntime, ModelConfig, build_model_client


class Answer(BaseModel):
    value: str


runtime = AIRuntime(
    build_model_client(
        ModelConfig(
            provider="gemini",
            model="gemini-3.1-flash-lite",
            api_key="...",
        )
    )
)
result = runtime.generate_structured("Return one answer.", Answer)
print(result.value)
```

## Retrieve quiz context through MCP

```python
from ai_runtime.mcp import LearningContextClient, MCPToolClient

context = LearningContextClient(
    MCPToolClient("http://mcp-server:8001/mcp")
).retrieve_quiz_context(
    course_id="CO3115",
    lo_code="L.O.3.1",
    bloom_level="analyze",
)
```

For an OpenAI-compatible model endpoint, select
`provider="openai-compatible"` and provide `base_url`. The SDK keeps provider
selection, timeout/retry policy, token usage, structured-output validation,
and MCP transport out of application use cases.

For DeepSeek, select `provider="deepseek"`. The SDK uses
`https://api.deepseek.com` by default, so only `model` and `api_key` are
required; `base_url` can still override the endpoint.
