class AIRuntimeError(RuntimeError):
    """Base SDK error."""


class ModelConfigurationError(AIRuntimeError):
    """Model provider configuration is invalid or incomplete."""


class ModelGenerationError(AIRuntimeError):
    """A model request failed or returned no usable text."""


class StructuredOutputError(AIRuntimeError):
    """The model response could not be validated as the requested schema."""


class MCPToolError(AIRuntimeError):
    """An MCP tool call failed or returned an invalid response."""

