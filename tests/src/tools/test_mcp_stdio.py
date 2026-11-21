"""Exercise the installed FastMCP/SDK pair through EvoAgentX's public wrapper."""

import sys
from importlib import import_module


def test_mcp_client_discovers_and_calls_local_stdio_tool(tmp_path):
    # A compatible install must also expose the public workflow entry point.
    import_module("evoagentx.workflow")
    from evoagentx.tools.mcp import MCPClient

    server = tmp_path / "mcp_server.py"
    server.write_text(
        "from fastmcp import FastMCP\n"
        "server = FastMCP('offline-arithmetic')\n"
        "@server.tool\n"
        "def add(a: int, b: int) -> int:\n"
        "    return a + b\n"
        "if __name__ == '__main__':\n"
        "    server.run()\n",
        encoding="utf-8",
    )
    config = {
        "mcpServers": {"arithmetic": {"command": sys.executable, "args": [str(server)]}}
    }

    with MCPClient(config, connect_timeout=10) as toolkits:
        assert len(toolkits) == 1
        toolkit = toolkits[0]
        assert toolkit.get_tool_names() == ["add"]
        assert toolkit.get_tool("add")(a=2, b=3) == ["5"]
