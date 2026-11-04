# EvoAgentX Quickstart Guide

This quickstart guide will walk you through the essential steps to set up and start using EvoAgentX. In this tutorial, you'll learn how to:

1. Configure your API keys to access LLMs 
2. Automatically create and execute workflows 


## Installation
```bash
pip install git+https://github.com/EvoAgentX/EvoAgentX.git
```
Please refere to [Installation Guide](./installation.md) for more details about the installation. 

## API Key & LLM Setup 

The first step to execute a workflow in EvoAgentX is configuring your API keys to access LLMs. There are two recommended methods to configure your API keys:

### Method 1: Set Environment Variables in the Terminal

This method sets the API key directly in your system environment.

For Linux/macOS: 
```bash
export OPENAI_API_KEY=<your-openai-api-key>
```

For Windows Command Prompt: 
```cmd 
set OPENAI_API_KEY=<your-openai-api-key>
```

For Windows PowerShell:
```powershell
$env:OPENAI_API_KEY="<your-openai-api-key>" # " is required 
```

Once set, you can access the key in your Python code with:
```python
import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
```

### Method 2: Use a `.env` File 

You can also store your API key in a `.env` file inside the root folder of your project.

Create a file named `.env` with the following content:
```bash
OPENAI_API_KEY=<your-openai-api-key>
```

Then, in your Python code, you can load the environment settings using `python-dotenv`:
```python
from dotenv import load_dotenv 
import os 

load_dotenv() # Loads environment variables from .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
```
🔐 Tip: Never commit your `.env` file to public platform (e.g., GitHub). Add it to `.gitignore`.

### Configure and Use the LLM in EvoAgentX
Once your API key is configured, you can initialize and use the LLM as follows:

```python
from evoagentx.models import OpenAILLMConfig, OpenAILLM

# Load the API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Define LLM configuration
openai_config = OpenAILLMConfig(
    model="gpt-4o-mini",       # Specify the model name
    openai_key=OPENAI_API_KEY, # Pass the key directly
    stream=True,               # Enable streaming response
    output_response=True       # Print response to stdout
)

# Initialize the language model
llm = OpenAILLM(config=openai_config)

# Generate a response from the LLM
response = llm.generate(prompt="What is Agentic Workflow?")
```

You can find more details about supported LLM types and their parameters in the [LLM module guide](./modules/llm.md).


## Automatic WorkFlow Generation and Execution 

Once your API key and language model are configured, you can automatically generate and execute agentic workflows in EvoAgentX. This section walks you through the core steps: generating a workflow from a goal, instantiating agents, and running the workflow to get results.

First, let's import the necessary modules:

```python
from evoagentx.workflow import WorkFlowGenerator, WorkFlowGraph, WorkFlow
from evoagentx.agents import AgentManager
```

### Step 1: Generate WorkFlow and Agents 
Use the `WorkFlowGenerator` to automatically create a workflow based on a natural language goal:
```python
goal = "Generate html code for the Tetris game that can be played in the browser."
wf_generator = WorkFlowGenerator(llm=llm)
workflow_graph: WorkFlowGraph = wf_generator.generate_workflow(goal=goal)
```
`WorkFlowGraph` is a data structure that stores the overall workflow plan — including task nodes and their relationships — but does not yet include executable agents.

You can optionally **visualize** or **save** the generated workflow:
```python
# Visualize the workflow without blocking later steps (optional).
# Node colors update while the workflow runs. Use display(block=True) to wait.
workflow_graph.display()

# Save the workflow to a JSON file (optional)
workflow_graph.save_module("/path/to/save/workflow_demo.json")
```
We provide an example generated workflow [here](https://github.com/EvoAgentX/EvoAgentX/blob/main/examples/output/tetris_game/workflow_demo_4o_mini.json). You can **reload** the saved workflow:
```python
workflow_graph = WorkFlowGraph.from_file("/path/to/save/workflow_demo.json")
```

### Step 2: Create and Manage Executable Agents 

Use `AgentManager` to instantiate and manage agents based on the workflow graph:
```python
agent_manager = AgentManager()
agent_manager.add_agents_from_workflow(workflow_graph, llm_config=openai_config)
```

### Step 3: Execute the Workflow 
Once agents are ready, you can create a `WorkFlow` instance and run it:
```python
workflow = WorkFlow(graph=workflow_graph, agent_manager=agent_manager, llm=llm)
result = workflow.execute()
if result.status == "success":
    print(result.result)
else:
    print(result.displayable_error)
```

`WorkFlow.execute()` returns a `WorkflowResult` object. By default, `result.result` is the structured workflow output as a `dict`. If you want the older text-extraction behavior, call `workflow.execute(extract_output=True)`.

For a complete working example, check out the [full workflow demo](https://github.com/EvoAgentX/EvoAgentX/blob/main/examples/workflow_demo.py).
