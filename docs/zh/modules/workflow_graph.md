# 工作流图

## 简介

`WorkFlowGraph` 类是 EvoAgentX 框架中的一个基础组件，用于创建、管理和执行复杂的 AI 代理工作流。它提供了一种结构化的方式来定义任务依赖关系、执行顺序和任务之间的数据流。

工作流图表示一组任务（节点）及其依赖关系（边），这些任务需要按特定顺序执行以实现目标。`SequentialWorkFlowGraph` 是一个专门的实现，专注于从开始到结束的单一路径的线性工作流。

## 架构

### 工作流图架构

`WorkFlowGraph` 由几个关键组件组成：

1. **节点（WorkFlowNode）**：
   
    每个节点代表工作流中的一个任务或操作，具有以下属性：

    - `name`：任务的唯一标识符
    - `description`：任务功能的详细描述
    - `inputs`：任务所需的输入参数列表，每个输入参数是 `Parameter` 类的实例
    - `outputs`：任务产生的输出参数列表，每个输出参数是 `Parameter` 类的实例
    - `agents`（可选）：可以执行此任务的代理列表，每个代理应该是一个与 `agent_manager` 中代理名称匹配的**字符串**，或者是一个指定代理名称和配置的**字典**，该配置将用于在 `agent_manager` 中创建 `CustomizeAgent` 实例。有关代理配置的更多详细信息，请参阅[自定义代理](./customize_agent.md)文档。
    - `action_graph`（可选）：`ActionGraph` 类的实例，其中每个动作都是 `Operator` 类的实例。有关动作图的更多详细信息，请参阅[动作图](./action_graph.md)文档。
    - `status`：任务的当前执行状态（PENDING、RUNNING、COMPLETED、FAILED）

    !!! note 
        1. 您应该提供 `agents` 或 `action_graph` 来执行任务。如果两者都提供，将使用 `action_graph`。

        2. 如果您提供一组 `agents`，这些代理将协同工作以完成任务。使用 `WorkFlow` 执行任务时，系统将根据代理信息和执行历史自动确定执行顺序（动作）。具体来说，执行任务时，`WorkFlow` 将分析这些代理中的所有可能动作，并根据任务描述和执行历史重复选择最佳动作执行。

        3. 如果您提供 `action_graph`，它将直接用于完成任务。使用 `WorkFlow` 执行任务时，系统将按照 `action_graph` 定义的顺序执行动作并返回结果。

2. **边（WorkFlowEdge）**：
   
    边表示任务之间的依赖关系，定义执行顺序和数据流。每条边具有：

    - `source`：源节点名称（边的起点）
    - `target`：目标节点名称（边的终点）
    - `priority`（可选）：影响执行顺序的数值优先级

3. **图结构**：
   
    在内部，工作流表示为有向图，其中：

    - 节点表示任务
    - 边表示任务之间的依赖关系和数据流
    - 图结构支持线性序列和更复杂的模式：
        - 分叉-合并模式（稍后重新汇合的并行执行路径）
        - 条件分支
        - 工作流中潜在的循环

4. **节点状态**：
   
    工作流中的每个节点可以处于以下状态之一：
    
    - `PENDING`：任务等待执行
    - `RUNNING`：任务正在执行
    - `COMPLETED`：任务已成功执行
    - `FAILED`：任务执行失败

### 顺序工作流图架构

`SequentialWorkFlowGraph` 是 `WorkFlowGraph` 的专门实现，它自动推断节点连接以创建线性工作流。它专为更简单的用例设计，其中任务需要按顺序执行，一个任务的输出作为下一个任务的输入。

#### 输入格式

`SequentialWorkFlowGraph` 接受简化的输入格式，使定义线性工作流变得容易。您不需要显式定义节点和边，而是按照执行顺序提供任务列表。每个任务都定义为一个字典，包含以下字段：

- `name`（必需）：任务的唯一标识符
- `description`（必需）：任务功能的详细描述
- `inputs`（必需）：任务的输入参数列表
- `outputs`（必需）：任务产生的输出参数列表
- `prompt`（必需）：指导代理行为的提示模板
- `system_prompt`（可选）：为代理提供上下文的系统消息
- `output_parser`（可选）：用于解析任务输出的解析器
- `parse_mode`（可选）：解析输出的模式，默认为 "str"
- `parse_func`（可选）：用于解析输出的自定义函数
- `title_format`（可选）：当 `parse_mode` 为 "title" 时使用的标题格式，例如 "## {title}"

与提示和解析相关的参数将用于在 `agent_manager` 中创建 `CustomizeAgent` 实例。有关代理配置的更多详细信息，请参阅[自定义代理](./customize_agent.md)文档。

#### 内部转换为工作流图

在内部，`SequentialWorkFlowGraph` 通过以下方式自动将此简化的任务列表转换为完整的 `WorkFlowGraph`：

1. **创建 WorkFlowNode 实例**：对于输入列表中的每个任务，它创建一个具有适当属性的 `WorkFlowNode`。在此过程中：

    - 它将任务定义转换为具有输入、输出和关联代理的节点
    - 它根据任务名称自动生成唯一的代理名称
    - 它使用提供的提示、系统提示和解析选项配置代理

2. **推断边连接**：它检查每个任务的输入和输出参数，并自动创建 `WorkFlowEdge` 实例来连接任务，其中一个任务的输出与另一个任务的输入匹配

3. **构建图结构**：最后，它构建表示工作流的完整有向图，所有节点和边都正确连接

这种自动转换过程使得定义顺序工作流变得更容易，无需手动指定所有图组件。

## 使用方法

### 基本工作流图创建与执行

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

# 将代理添加到代理管理器以执行工作流
data_extraction_agent = agent_manager.add_agents(agents = [data_extraction_agent, data_transformation_agent])

# 创建工作流节点
task1 = WorkFlowNode(
    name="Task1",
    description="Extract data from source",
    inputs=[{"name": "data_source", "type": "string", "description": "Source data location"}],
    outputs=[{"name": "extracted_data", "type": "string", "description": "Extracted data"}],
    agents=["DataExtractionAgent"] # 应该与代理管理器中的代理名称匹配
)

task2 = WorkFlowNode(
    name="Task2",
    description="Transform data",
    inputs=[{"name": "extracted_data", "type": "string", "description": "Data to transform"}],
    outputs=[{"name": "transformed_data", "type": "string", "description": "Transformed data"}],
    agents=["DataTransformationAgent"] # 应该与代理管理器中的代理名称匹配
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
        } # 将用于在 agent_manager 中创建 CustomizeAgent 实例
    ]
)

# 创建工作流边
edge1 = WorkFlowEdge(source="Task1", target="Task2")
edge2 = WorkFlowEdge(source="Task2", target="Task3")

# 创建工作流图
workflow_graph = WorkFlowGraph(
    goal="Extract, transform, and analyze data to generate insights",
    nodes=[task1, task2, task3],
    edges=[edge1, edge2]
)

# 将代理添加到代理管理器以执行工作流
agent_manager.add_agents_from_workflow(workflow_graph, llm_config=llm_config)

# 创建工作流实例以执行
workflow = WorkFlow(graph=workflow_graph, agent_manager=agent_manager, llm=llm)
result = workflow.execute(inputs={"data_source": "xxx"})
if result.status == "success":
    print(result.result)
```

`WorkFlow.execute()` 会返回一个 `WorkflowResult` 对象。默认情况下，工作流结果是结构化的 `dict` 输出。若想保留旧的文本抽取行为，可以传入 `extract_output=True`。

### 创建顺序工作流图

```python
from evoagentx.workflow.workflow_graph import SequentialWorkFlowGraph

# 定义任务及其输入、输出和提示
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

# 创建顺序工作流图
sequential_workflow_graph = SequentialWorkFlowGraph(
    goal="Extract, transform, and analyze data to generate insights",
    tasks=tasks
)
```

### 保存和加载工作流

```python
# 保存工作流
workflow_graph.save_module("examples/output/my_workflow.json")

# 对于 SequentialWorkFlowGraph，使用 save_module 和 get_graph_info
sequential_workflow_graph.save_module("examples/output/my_sequential_workflow.json")
```

### 可视化工作流

```python
# 打开非阻塞窗口。工作流运行时节点颜色会随之更新。
# 传入 block=True 会等待窗口关闭。
workflow_graph.display()
workflow_graph.close_display()
```

`WorkFlowGraph` 和 `SequentialWorkFlowGraph` 类提供了一种灵活而强大的方式来设计复杂的代理工作流、跟踪其执行并管理任务之间的数据流。 
