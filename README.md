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
            model="gemini-3-flash-preview",
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
