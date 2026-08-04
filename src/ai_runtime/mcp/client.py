import asyncio
import json
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

from mcp import ClientSession, types
from mcp.client.streamable_http import streamable_http_client

from ai_runtime.errors import MCPToolError

ResultT = TypeVar("ResultT")


class MCPToolClient:
    def __init__(self, url: str, timeout_seconds: float = 30) -> None:
        self._url = url
        self._timeout_seconds = timeout_seconds

    @property
    def url(self) -> str:
        return self._url

    async def call_tool_async(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            return await asyncio.wait_for(
                self._call_tool(name=name, arguments=arguments),
                timeout=self._timeout_seconds,
            )
        except MCPToolError:
            raise
        except Exception as exc:
            raise MCPToolError(
                f"MCP tool {name!r} failed at {self._url}"
            ) from exc

    def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        return _run_sync(lambda: self.call_tool_async(name, arguments))

    async def _call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        async with streamable_http_client(self._url) as (
            read_stream,
            write_stream,
            _,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments=arguments)

        if result.isError:
            message = "MCP tool returned an error"
            for block in result.content:
                if isinstance(block, types.TextContent):
                    message = block.text
                    break
            raise MCPToolError(message)

        structured = getattr(result, "structuredContent", None)
        if isinstance(structured, dict):
            return structured

        for block in result.content:
            if not isinstance(block, types.TextContent):
                continue
            try:
                parsed = json.loads(block.text)
            except json.JSONDecodeError:
                return {"text": block.text}
            if isinstance(parsed, dict):
                return parsed
            return {"value": parsed}
        raise MCPToolError(f"MCP tool {name!r} returned no structured or text content")


def _run_sync(factory: Callable[[], Awaitable[ResultT]]) -> ResultT:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(factory())

    # Worker callbacks may already run inside an asyncio loop. Run the short
    # MCP session on a dedicated thread instead of nesting event loops.
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="mcp-client") as executor:
        return executor.submit(lambda: asyncio.run(factory())).result()

