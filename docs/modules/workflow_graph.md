# Workflow Graph

## Introduction

The `WorkFlowGraph` class is a fundamental component in the EvoAgentX framework for creating, managing, and executing complex AI agent workflows. It provides a structured way to define task dependencies, execution order, and the flow of data between tasks.

A workflow graph represents a collection of tasks (nodes) and their dependencies (edges) that need to be executed in a specific order to achieve a goal. The `SequentialWorkFlowGraph` is a specialized implementation that focuses on linear workflows with a single path from start to end.

## Architecture

### WorkFlowGraph Architecture

A `WorkFlowGraph` consists of several key components:

1. **Nodes (WorkFlowNode)**: 
   
    Each node represents a task or operation in the workflow, with the following properties:

    - `name`: A unique identifier for the task
    - `description`: Detailed description of what the task does
    - `inputs`: List of input parameters required by the task, each input parameter is an instance of `Parameter` class. 
    - `outputs`: List of output parameters produced by the task, each output parameter is an instance of `Parameter` class. 
    - `agents` (optional): List of agents that can execute this task, each agent should be a **string** that matches the name of the agent in the `agent_manager` or a **dictionary** that specifies the agent name and its configuration, which will be used to create a `CustomizeAgent` instance in the `agent_manager`.  Please refer to the [Customize Agent](./customize_agent.md) documentation for more details about the agent configuration. 
    - `action_graph` (optional): An instance of `ActionGraph` class, where each action is an instance of the `Operator` class. Please refer to the [Action Graph](./action_graph.md) documentation for more details about the action graph. 
    - `status`: Current execution state of the task (PENDING, RUNNING, COMPLETED, FAILED).

    !!! note 
        1. You should provide either `agents` or `action_graph` to execute the task. If both are provided, `action_graph` will be used. 

        2. If you provide a set of `agents`, these agents will work together to complete the task. When executing the task using `WorkFlow`, the system will automatically determine the execution sequence (actions) based on the agent information and execution history. Specifically, when executing the task, `WorkFlow` will analyze all the possible actions within these agents and repeatly select the best action to execute based on the task description and execution history. 

        3. If you provide an `action_graph`, it will be directly used to complete the task. When executing the task with `WorkFlow`, the system will execute the actions in the order defined by the `action_graph` and return the results.  


2. **Edges (WorkFlowEdge)**: 
   
    Edges represent dependencies between tasks, defining execution order and data flow. Each edge has:

    - `source`: Name of the source node (where the edge starts)
    - `target`: Name of the target node (where the edge ends) 
    - `priority` (optional): numeric priority to influence execution order

3. **Graph Structure**:
   
    Internally, the workflow is represented as a directed graph where:

    - Nodes represent tasks
    - Edges represent dependencies and data flow between tasks
    - The graph structure supports both linear sequences and more complex patterns:
        - Fork-join patterns (parallel execution paths that rejoin later)
        - Conditional branches
        - Potential cycles (loops) in the workflow

4. **Node States**:
   
    Each node in the workflow can be in one of the following states:
    
    - `PENDING`: The task is waiting to be executed
    - `RUNNING`: The task is currently being executed
    - `COMPLETED`: The task has been successfully executed
    - `FAILED`: The task execution has failed

### SequentialWorkFlowGraph Architecture

The `SequentialWorkFlowGraph` is a specialized implementation of `WorkFlowGraph` that automatically infers node connections to create a linear workflow. It's designed for simpler use cases where tasks need to be executed in sequence, with outputs from one task feeding into the next.

#### Input Format

The `SequentialWorkFlowGraph` accepts a simplified input format that makes it easy to define linear workflows. Instead of explicitly defining nodes and edges, you provide a list of tasks in the order they should be executed. Each task is defined as a dictionary with the following fields:

- `name` (required): A unique identifier for the task
- `description` (required): Detailed description of what the task does
- `inputs` (required): List of input parameters for the task
- `outputs` (required): List of output parameters produced by the task
- `prompt` (required): The prompt template to guide the agent's behavior
- `system_prompt` (optional): System message to provide context to the agent
- `output_parser` (optional): The output parser to parse the output of the task
- `parse_mode` (optional): Mode for parsing outputs, defaults to "str"
- `parse_func` (optional): Custom function for parsing outputs
- `title_format` (optional): Title format used when `parse_mode` is "title", e.g. "## {title}"

The parameters related to prompts and parsing will be used to create a `CustomizeAgent` instance in the `agent_manager`. Please refer to the [Customize Agent](./customize_agent.md) documentation for more details about the agent configuration. 

#### Internal Conversion to WorkFlowGraph

Internally, `SequentialWorkFlowGraph` automatically converts this simplified task list into a complete `WorkFlowGraph` by:

1. **Creating WorkFlowNode instances**: For each task in the input list, it creates a `WorkFlowNode` with appropriate properties. During this process:

    - It converts the task definition into a node with inputs, outputs, and an associated agent.
    - It automatically generates a unique agent name based on the task name.
    - It configures the agent with the provided prompt, system_prompt, and parsing options.

2. **Inferring edge connections**: It examines the input and output parameters of each task and automatically creates `WorkFlowEdge` instances to connect tasks where outputs from one task match the inputs of another.

3. **Building the graph structure**: Finally, it constructs the complete directed graph representing the workflow, with all nodes and edges properly connected.

This automatic conversion process makes it significantly easier to define sequential workflows without needing to manually specify all the graph components.

## Usage

### Basic WorkFlowGraph Creation & Execution 

```python
from evoagentx.workflow.workflow_graph import WorkFlowNode, WorkFlowGraph, WorkFlowEdge
from evoagentx.workflow.workflow import WorkFlow 
from evoagentx.agents import AgentManager, CustomizeAgent 
from evoagentx.models import OpenAILLMConfig, OpenAILLM 

llm_config = OpenAILLMConfig(model="gpt-4o-mini", openai_key="xxx", stream=True, output_response=True)
llm = OpenAILLM(llm_config)

agent_manager = AgentManager()

data_extraction_agent = CustomizeAgent(
    name="DataExtractionAgent",
    description="Extract data from source",
    inputs=[{"name": "data_source", "type": "string", "description": "Source data location"}],
    outputs=[{"name": "extracted_data", "type": "string", "description": "Extracted data"}],
    prompt="Extract data from source: {data_source}",
    llm_config=llm_config
)  

data_transformation_agent = CustomizeAgent(
    name="DataTransformationAgent",
    description="Transform data",
    inputs=[{"name": "extracted_data", "type": "string", "description": "Extracted data"}],
    outputs=[{"name": "transformed_data", "type": "string", "description": "Transformed data"}],
    prompt="Transform data: {extracted_data}",
    llm_config=llm_config
)

# add agents to the agent manager for workflow execution 
data_extraction_agent = agent_manager.add_agents(agents = [data_extraction_agent, data_transformation_agent])

# Create workflow nodes
task1 = WorkFlowNode(
    name="Task1",
    description="Extract data from source",
    inputs=[{"name": "data_source", "type": "string", "description": "Source data location"}],
    outputs=[{"name": "extracted_data", "type": "string", "description": "Extracted data"}],
    agents=["DataExtractionAgent"] # should match the name of the agent in the agent manager
)

task2 = WorkFlowNode(
    name="Task2",
    description="Transform data",
    inputs=[{"name": "extracted_data", "type": "string", "description": "Data to transform"}],
    outputs=[{"name": "transformed_data", "type": "string", "description": "Transformed data"}],
    agents=["DataTransformationAgent"] # should match the name of the agent in the agent manager
)

task3 = WorkFlowNode(
    name="Task3",
    description="Analyze data and generate insights",
    inputs=[{"name": "transformed_data", "type": "string", "description": "Data to analyze"}],
    outputs=[{"name": "insights", "type": "string", "description": "Generated insights"}],
    agents=[
        {
            "name": "DataAnalysisAgent",
            "description": "Analyze data and generate insights",
            "inputs": [{"name": "transformed_data", "type": "string", "description": "Data to analyze"}],
            "outputs": [{"name": "insights", "type": "string", "description": "Generated insights"}],
            "prompt": "Analyze data and generate insights: {transformed_data}",
            "parse_mode": "str",
        } # will be used to create a `CustomizeAgent` instance in the `agent_manager`
    ]
)

# Create workflow edges
edge1 = WorkFlowEdge(source="Task1", target="Task2")
edge2 = WorkFlowEdge(source="Task2", target="Task3")

# Create the workflow graph
workflow_graph = WorkFlowGraph(
    goal="Extract, transform, and analyze data to generate insights",
    nodes=[task1, task2, task3],
    edges=[edge1, edge2]
)

# add agents to the agent manager for workflow execution 
agent_manager.add_agents_from_workflow(workflow_graph, llm_config=llm_config)

# create a workflow instance for execution 
workflow = WorkFlow(graph=workflow_graph, agent_manager=agent_manager, llm=llm)
result = workflow.execute(inputs={"data_source": "xxx"})
if result.status == "success":
    print(result.result)
```

`WorkFlow.execute()` returns a `WorkflowResult` object. By default, the workflow result is the structured output data as a `dict`. Pass `extract_output=True` if you want the older text-extraction behavior.

### Creating a SequentialWorkFlowGraph

```python
from evoagentx.workflow.workflow_graph import SequentialWorkFlowGraph

# Define tasks with their inputs, outputs, and prompts
tasks = [
    {
        "name": "DataExtraction",
        "description": "Extract data from the specified source",
        "inputs": [
            {"name": "data_source", "type": "string", "required": True, "description": "Source data location"}
        ],
        "outputs": [
            {"name": "extracted_data", "type": "string", "required": True, "description": "Extracted data"}
        ],
        "prompt": "Extract data from the following source: {data_source}", 
        "parse_mode": "str"
    },
    {
        "name": "DataTransformation",
        "description": "Transform the extracted data",
        "inputs": [
            {"name": "extracted_data", "type": "string", "required": True, "description": "Data to transform"}
        ],
        "outputs": [
            {"name": "transformed_data", "type": "string", "required": True, "description": "Transformed data"}
        ],
        "prompt": "Transform the following data: {extracted_data}", 
        "parse_mode": "str"
    },
    {
        "name": "DataAnalysis",
        "description": "Analyze data and generate insights",
        "inputs": [
            {"name": "transformed_data", "type": "string", "required": True, "description": "Data to analyze"}
        ],
        "outputs": [
            {"name": "insights", "type": "string", "required": True, "description": "Generated insights"}
        ],
        "prompt": "Analyze the following data and generate insights: {transformed_data}", 
        "parse_mode": "str"
    }
]

# Create the sequential workflow graph
sequential_workflow_graph = SequentialWorkFlowGraph(
    goal="Extract, transform, and analyze data to generate insights",
    tasks=tasks
)
```

### Saving and Loading a Workflow

```python
# Save workflow
workflow_graph.save_module("examples/output/my_workflow.json")

# For SequentialWorkFlowGraph, use save_module and get_graph_info
sequential_workflow_graph.save_module("examples/output/my_sequential_workflow.json")
```

### Visualizing the Workflow

```python
# Open a non-blocking window. Node colors update while the workflow runs.
# Pass block=True to wait until the window is closed.
workflow_graph.display()
workflow_graph.close_display()
```

The `WorkFlowGraph` and `SequentialWorkFlowGraph` classes provide a flexible and powerful way to design complex agent workflows, track their execution, and manage the flow of data between tasks. 
