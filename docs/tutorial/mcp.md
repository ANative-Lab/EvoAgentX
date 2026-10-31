# Working with MCP Tools in EvoAgentX

This tutorial walks you through using the Model Context Protocol (MCP) integration in EvoAgentX. MCP allows agents to interact with various tools and services through a standardized protocol. The implementation uses FastMCP 2.0 for enhanced performance and reliability. We'll cover:

1. **Understanding MCP**: Learn about the Model Context Protocol and its integration in EvoAgentX
2. **Setting Up MCP Tools**: Configure and initialize MCP tools using different methods
3. **Using MCP Tools**: Access and use MCP tools within your agent workflows

By the end of this tutorial, you'll understand how to leverage MCP tools in your own EvoAgentX applications.

---

## 1. Understanding MCP

The Model Context Protocol (MCP) is a standardized way for language models to communicate with external tools and services. EvoAgentX provides integration with MCP through the `MCPToolkit` class, powered by FastMCP 2.0.

```python
from evoagentx.tools import MCPToolkit
```

MCP integration in EvoAgentX allows you to:

- Connect to MCP-compatible servers (both stdio and HTTP-based)
- Discover available tools from these servers
- Use these tools through a consistent interface
- Benefit from FastMCP 2.0's enhanced performance and reliability

### Key Concepts

- **MCP Servers**: External processes or services that provide tools through the MCP protocol
- **Tool Discovery**: Automatically discover and expose tools provided by MCP servers
- **Standardized Interface**: All MCP tools follow the same interface pattern, making them easy to use

---

## 2. Setting Up MCP Tools

EvoAgentX provides two main ways to set up MCP tools:

1. **From a Configuration File**: Load server configurations from a JSON file
2. **From a Configuration Dictionary**: Configure servers directly in your code

### 2.1 From a Configuration File

You can set up MCP tools by providing a configuration file path:

```python
from evoagentx.tools import MCPToolkit

# Initialize with a configuration file
toolkit = MCPToolkit(config_path="examples/mcp.config")

# Get all available tools
tools = toolkit.get_toolkits()
```

#### 2.1.1 Configuration File Format

The configuration file should be a JSON file that follows FastMCP 2.0's configuration format. Here's an example:

```json
{
    "mcpServers": {
        "github": {
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-github"
            ],
            "env": {
                "GITHUB_PERSONAL_ACCESS_TOKEN": "your_github_token_here"
            },
            "timeout": 30
        },
        "arxiv": {
            "command": "uv",
            "args": [
                "tool",
                "run",
                "arxiv-mcp-server",
                "--storage-path", "./data/"
            ],
            "timeout": 45
        }
    }
}
```

Each server configuration can include:

- `command`: The command to start a local stdio MCP server (required for local servers)
- `args`: Command-line arguments for the server (optional)
- `env`: Environment variables to set when running the server (optional)
- `timeout`: Connection timeout in seconds (optional, defaults to 120.0)

### 2.2 From a Configuration Dictionary

Alternatively, you can configure MCP tools directly in your code:

```python
import os
from evoagentx.tools import MCPToolkit

# Configuration dictionary
config = {
    "mcpServers": {
        "arxiv": {
            "command": "uv",
            "args": [
                "tool", 
                "run", 
                "arxiv-mcp-server",
                "--storage-path", "./data/"
            ],
            "timeout": 45
        },
    }
}

# Initialize with the configuration dictionary
toolkit = MCPToolkit(config=config)

# Get all available tools
tools = toolkit.get_toolkits()
```

### 2.3 Remote HTTP Servers: Keyless Web Search

Remote servers use a `url` instead of a local `command`. Streamable HTTP
configuration requires FastMCP 2.4 or later, within the supported 2.x series.
Install the tools extra from the repository root:

```bash
pip install -e '.[tools]'
```

[Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp)
provides `web_search` and `web_fetch` without a Parallel account or API key.
Anonymous searches use Fast mode and free access is rate limited.

```python
from evoagentx import __version__
from evoagentx.tools import MCPToolkit

config = {
    "mcpServers": {
        "parallel-search": {
            "url": "https://search.parallel.ai/mcp",
            "transport": "http",
            "headers": {"User-Agent": f"EvoAgentX/{__version__}"},
        }
    }
}

toolkit = MCPToolkit(config=config)
try:
    toolkits = toolkit.get_toolkits()
    # Pass these toolkits to your agent, or call their tools directly.
    for server in toolkits:
        print(server.name, [tool.name for tool in server.get_tools()])
finally:
    toolkit.disconnect()
```

The `headers` field identifies EvoAgentX through its package version so Parallel
can measure aggregate free MCP usage. It contains no user or installation ID.
For a JSON configuration file, use a literal value such as `EvoAgentX/0.1.4`;
Python expressions are not evaluated by the JSON loader.

This connection is enabled only when your application loads the configuration.
It does not select a default search provider or change your other toolkits.
Once you give these tools to an agent, it may call them during its work. Search
queries, requested URLs, and any supplied objectives or context go to Parallel.
Use public information in this example. No Authorization header is needed for
the anonymous endpoint.

For a complete example that discovers the tools, runs a search, fetches a public
page, and disconnects, run:

```bash
python examples/tools/parallel_search_mcp.py
```

---

## 3. Using MCP Tools

Once your MCP toolkit is initialized, you can access and use the available tools.

### 3.1 Getting Available Tools

The `get_toolkits()` method returns a list of `Toolkit` objects from all connected MCP servers. Each `Toolkit` contains multiple tools:

```python
# Initialize the MCP toolkit
toolkit = MCPToolkit(config_path="examples/mcp.config")

# Get all available toolkits (each toolkit contains multiple tools)
toolkits = toolkit.get_toolkits()
```

### 3.2 Accessing Tool Information

Each toolkit contains multiple tools that you can access through the `get_tools()` method:

```python
# Loop through available toolkits
for toolkit in toolkits:
    print(f"Toolkit: {toolkit.name}")
    
    # Get tools from this toolkit
    tools = toolkit.get_tools()
    
    for tool in tools:
        # Get tool name
        name = tool.name
        print(f"  Tool: {name}")
        
        # Get tool description
        description = tool.description
        print(f"  Description: {description}")
        
        # Get tool inputs schema
        inputs = tool.inputs
        print(f"  Inputs: {inputs}")
        
        # Get required parameters
        required = tool.required
        print(f"  Required: {required}")
```

**Sample Output:**

```
Toolkit: arxiv-mcp-server
  Tool: arxiv_search
  Description: Search for research papers on arXiv by providing keywords and filters
  Inputs: {
      "query": {
          "type": "string",
          "description": "Search query for arXiv papers"
      },
      "max_results": {
          "type": "integer",
          "description": "Maximum number of results to return",
          "default": 10
      },
      "sort_by": {
          "type": "string",
          "description": "Sort results by relevance, lastUpdatedDate, or submittedDate",
          "default": "relevance"
      },
      "sort_order": {
          "type": "string",
          "description": "Sort order: ascending or descending",
          "default": "descending"
      }
  }
  Required: ["query"]
...
```

### 3.3 Using MCP Tools in Practice

Here's a complete example of initializing and using MCP tools:

```python
from evoagentx.tools import MCPToolkit

# Initialize the MCP toolkit
toolkit = MCPToolkit(config_path="examples/mcp.config")

# Get all available toolkits
toolkits = toolkit.get_toolkits()

# Find a specific tool by searching through toolkits
arxiv_tool = None
for toolkit in toolkits:
    tools = toolkit.get_tools()
    for tool in tools:
        if "arxiv" in tool.name.lower() or "search" in tool.name.lower():
            arxiv_tool = tool
            break
    if arxiv_tool:
        break

if arxiv_tool:
    # Call the tool with appropriate parameters (showing arXiv search schema)
    result = arxiv_tool(
        query="artificial intelligence machine learning",
        max_results=5,
        sort_by="relevance",
        sort_order="descending"
    )
    print(f"arXiv search results: {result}")

# Clean up when done
toolkit.disconnect()
```

### 3.4 Handling MCP Server Lifecycle

The MCP toolkit handles server connections automatically, but you should properly disconnect when you're done:

```python
# Initialize the toolkit
toolkit = MCPToolkit(config_path="examples/mcp.config")

try:
    # Use MCP tools
    tools = toolkit.get_toolkits()
    # ... work with tools ...
finally:
    # Disconnect from servers
    toolkit.disconnect()
```

You can also use the toolkit as a context manager:

```python
from evoagentx.tools.mcp import MCPClient

# Setup server configurations
server_configs = {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-github"],
    "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "your_token_here"}
}

with MCPClient(server_configs) as mcp_toolkits:
    # Use MCP tools here
    # Automatic disconnection happens when exiting the context
    for toolkit in mcp_toolkits:
        print(f"Available toolkit: {toolkit.name}")
        tools = toolkit.get_tools()
        for tool in tools:
            print(f"  - Tool: {tool.name}")
```

### 3.5 Using MCP in Practical Applications

Once you've initialized your MCP toolkit and obtained the toolkits, you can use them directly:

```python
# Initialize MCP toolkit
toolkit = MCPToolkit(config_path="mcp_config.json")
mcp_toolkits = toolkit.get_toolkits()

try:
    # Find a specific tool by searching through toolkits
    arxiv_search = None
    for toolkit in mcp_toolkits:
        tools = toolkit.get_tools()
        for tool in tools:
            if "arxiv" in tool.name.lower() or "search" in tool.name.lower():
                arxiv_search = tool
                break
        if arxiv_search:
            break
    
    if arxiv_search:
        # Call the tool with appropriate parameters (demonstrating arXiv search options)
        result = arxiv_search(
            query="quantum computing algorithms",
            max_results=10,
            sort_by="submittedDate",
            sort_order="descending"
        )
        print(f"arXiv search results: {result}")
finally:
    # Always disconnect when done
    toolkit.disconnect()
```

---

## 4. Using MCP with Agents

MCP tools integrate seamlessly with EvoAgentX agents through tool calling actions. Here's how to use MCP tools with agents.

### 4.1 Creating a Customized Agent with Tool Calling Action

The most flexible way to use MCP tools is to create a customized agent with tool mapping:

```python
import os 
from dotenv import load_dotenv
from evoagentx.models import OpenAILLMConfig
from evoagentx.agents import CustomizeAgent
from evoagentx.prompts import StringTemplate 
from evoagentx.tools import MCPToolkit

# Load environment variables and setup OpenAI config
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY, stream=True, output_response=True)

# Initialize MCP toolkit and get toolkits
toolkit = MCPToolkit(config_path="examples/mcp.config")
toolkits = toolkit.get_toolkits()

# Create a customized agent with the toolkits
mcp_agent = CustomizeAgent(
    name="MCPAgent",
    description="A MCP agent that can use the tools provided by the MCP server",
    prompt_template=StringTemplate(
        instruction="Do some operations based on the user's instruction."
    ), 
    llm_config=openai_config,
    inputs=[
        {"name": "instruction", "type": "string", "description": "The goal you need to achieve"}
    ],
    outputs=[
        {"name": "result", "type": "string", "description": "The result of the tool call"}
    ],
    tools=toolkits  # Pass toolkits directly
)

# Optional: Save and load the agent configuration
mcp_agent.save_module("examples/mcp_agent.json")
mcp_agent.load_module("examples/mcp_agent.json", llm_config=openai_config, tools=toolkits)

# Execute the agent with a realistic task
message = mcp_agent(
    inputs={"instruction": "Search for recent research papers on machine learning and artificial intelligence"}
)

print(f"Response from {mcp_agent.name}:")
print(message.content.result)

# Clean up when done
toolkit.disconnect()
```

This example demonstrates:
1. Setting up the MCP toolkit and getting available tools
2. Configuring a customized agent with the tools
3. Executing the agent with specific inputs

The agent can now use any of the MCP tools available through the toolkit.

### 4.2 Integration with Workflow and Agent Generation

MCP tools can also be integrated into workflow and agent generation systems. Here's a complete example from generation to execution:

```python
from evoagentx.tools import MCPToolkit
from evoagentx.agents.agent_manager import AgentManager
from evoagentx.workflow.workflow_generator import WorkFlowGenerator
from evoagentx.workflow import WorkFlow
from evoagentx.models.openai_model import OpenAILLM
from evoagentx.models import OpenAILLMConfig
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize LLM
llm_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key=OPENAI_API_KEY)
llm = OpenAILLM(llm_config)

# Initialize MCP toolkit
toolkit = MCPToolkit(config_path="mcp_config.json")
mcp_toolkits = toolkit.get_toolkits()

# Create agent manager and workflow generator
agent_manager = AgentManager(tools=mcp_toolkits)
workflow_generator = WorkFlowGenerator(llm=llm, tools=mcp_toolkits)

# Generate workflow
workflow_graph = workflow_generator.generate_workflow(
    goal="Search arXiv for research papers and create a summary report"
)

# Add agents to manager
agent_manager.add_agents_from_workflow(workflow_graph=workflow_graph, llm_config=llm_config)

# Create and execute workflow
workflow = WorkFlow(graph=workflow_graph, llm=llm, agent_manager=agent_manager)
result = workflow.execute(inputs={
    "research_topic": "machine learning",
    "max_papers": 5,
    "output_format": "summary"
})

if result.status == "success":
    print(f"Workflow result: {result.result}")
else:
    print(f"Workflow failed: {result.displayable_error}")

# Clean up
toolkit.disconnect()
```

This example demonstrates:
1. **Workflow Generation**: Creating a workflow graph from a high-level goal
2. **Agent Management**: Adding MCP-enabled agents to the agent manager
3. **Workflow Execution**: Running the complete workflow with inputs
4. **MCP Integration**: Using MCP tools throughout the workflow process

## 5. Error Handling and Best Practices

### 5.1 Proper Error Handling

When working with MCP tools, it's important to handle errors gracefully:

```python
from evoagentx.tools import MCPToolkit
from fastmcp.exceptions import ClientError, FastMCPError
import logging

# Set up logging to see MCP connection details
logging.basicConfig(level=logging.INFO)

try:
    # Initialize MCP toolkit with error handling
    toolkit = MCPToolkit(config_path="examples/mcp.config")
    
    # Get toolkits with timeout handling
    toolkits = toolkit.get_toolkits()
    
    if not toolkits:
        print("No MCP servers connected. Check your configuration.")
        return
    
    # Use tools with error handling
    for toolkit in toolkits:
        tools = toolkit.get_tools()
        for tool in tools:
            try:
                # Call tool with appropriate parameters for arXiv search
                if "arxiv" in tool.name.lower() or "search" in tool.name.lower():
                    result = tool(query="machine learning", max_results=3)
                else:
                    result = tool(query="test query")
                print(f"Tool {tool.name} result: {result}")
            except ClientError as e:
                print(f"Client error with tool {tool.name}: {e}")
            except FastMCPError as e:
                print(f"MCP protocol error with tool {tool.name}: {e}")
            except Exception as e:
                print(f"Unexpected error with tool {tool.name}: {e}")

except FileNotFoundError:
    print("MCP configuration file not found")
except json.JSONDecodeError:
    print("Invalid JSON in MCP configuration file")
except Exception as e:
    print(f"Failed to initialize MCP toolkit: {e}")
finally:
    # Always clean up
    if 'toolkit' in locals():
        toolkit.disconnect()
```

### 5.2 Configuration Best Practices

Create robust MCP configurations with proper error handling:

```json
{
    "mcpServers": {
        "arxiv": {
            "command": "uv",
            "args": [
                "tool", 
                "run", 
                "arxiv-mcp-server",
                "--storage-path", "./data/"
            ],
            "timeout": 45
        },
    }
}
```

### 5.3 Connection Management

Use context managers for automatic cleanup:

```python
from evoagentx.tools.mcp import MCPClient

# Using MCPClient directly with context manager
server_config = {
    "mcpServers": {
        "arxiv-server": {
            "command": "uv",
            "args": [
                "tool", 
                "run", 
                "arxiv-mcp-server",
                "--storage-path", "./data/"
            ]
        }
    }
}

try:
    with MCPClient(server_config) as toolkits:
        for toolkit in toolkits:
            print(f"Connected to: {toolkit.name}")
            # Use tools here
except Exception as e:
    print(f"Failed to connect to MCP server: {e}")
# Automatic cleanup happens here
```

## 6. Troubleshooting

Here are some common issues and their solutions:

- **Connection Timeout**: If you encounter timeout errors connecting to MCP servers, increase the timeout value in your configuration (default is 120.0 seconds).

- **Missing Tools**: If expected tools aren't available, check if the MCP server is running correctly and whether it's properly configured to expose those tools.

- **Tool Execution Errors**: FastMCP 2.0 provides specific error types:
  - `ClientError`: Indicates client-side issues (connection, configuration, etc.)
  - `FastMCPError`: Base class for server-side MCP protocol errors
  Check the error message for details about what went wrong.

- **Server Not Starting**: If an MCP server fails to start, verify the command path and environment variables in your configuration.

- **Empty Toolkit List**: If `get_toolkits()` returns an empty list, check that your MCP servers are properly configured and running.

For more information about MCP and FastMCP, visit:
- [Model Context Protocol](https://github.com/modelcontextprotocol/protocol)
- [FastMCP Documentation](https://gofastmcp.com/clients/client)
