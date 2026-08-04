import multiprocessing
import socket
import time

from mcp.server.fastmcp import FastMCP

from ai_runtime.mcp import MCPToolClient


def _serve(port: int) -> None:
    server = FastMCP(
        "roundtrip-test",
        host="127.0.0.1",
        port=port,
        stateless_http=True,
        json_response=True,
    )

    @server.tool()
    def echo(value: str) -> dict:
        return {"value": value}

    server.run(transport="streamable-http")


def _wait_for_port(port: int) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.05)
    raise TimeoutError(f"MCP test server did not start on port {port}")


def test_streamable_http_roundtrip(unused_tcp_port):
    process = multiprocessing.Process(target=_serve, args=(unused_tcp_port,))
    process.start()
    try:
        _wait_for_port(unused_tcp_port)
        result = MCPToolClient(
            f"http://127.0.0.1:{unused_tcp_port}/mcp",
            timeout_seconds=5,
        ).call_tool("echo", {"value": "ok"})
        assert result == {"value": "ok"}
    finally:
        process.terminate()
        process.join(timeout=5)
