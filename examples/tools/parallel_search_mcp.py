"""Search and fetch public web sources through the free Parallel Search MCP."""

from uuid import uuid4

from evoagentx import __version__
from evoagentx.tools import MCPToolkit


def main():
    """Run keyless web research using EvoAgentX's existing MCP toolkit."""
    config = {
        "mcpServers": {
            "parallel-search": {
                "url": "https://search.parallel.ai/mcp",
                "transport": "http",
                # Identify the project for aggregate free MCP usage measurement.
                # Keep this project-wide; do not add user or installation IDs.
                "headers": {"User-Agent": f"EvoAgentX/{__version__}"},
            }
        }
    }
    toolkit = MCPToolkit(config=config)
    try:
        session_id = uuid4().hex
        tools = {
            tool.name: tool
            for server in toolkit.get_toolkits()
            for tool in server.get_tools()
        }
        print(tools["web_search"](
            objective="Find the official Parallel Search MCP setup instructions",
            search_queries=["Parallel Search MCP setup"],
            session_id=session_id,
        ))
        print(tools["web_fetch"](
            urls=["https://docs.parallel.ai/integrations/mcp/search-mcp"],
            objective="Explain anonymous access and the available tools",
            session_id=session_id,
        ))
    finally:
        toolkit.disconnect()


if __name__ == "__main__":
    main()
