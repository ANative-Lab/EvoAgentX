# EvoAgentX 快速开始指南

本快速开始指南将引导你完成使用 EvoAgentX 的基础步骤。在本教程中，你将学习如何：
1. 配置用于访问 LLM 的 API 密钥  
2. 自动创建并执行工作流  

## 安装
```bash
pip install git+https://github.com/EvoAgentX/EvoAgentX.git
```
请参阅 [安装指南](./installation.md) 获取更多详细信息。

## API 密钥 和 LLM 设置

要在 EvoAgentX 中执行工作流，首先需要配置用于访问大模型（LLM）的 API 密钥。推荐以下两种方式：

### 方法一：在终端设置环境变量

此方法直接在系统环境中设置 API 密钥。

对于 Linux/macOS：
```bash
export OPENAI_API_KEY=<你的-openai-api-key>
```

对于 Windows 命令提示符：
```cmd
set OPENAI_API_KEY=<你的-openai-api-key>
```

对于 Windows PowerShell：
```powershell
$env:OPENAI_API_KEY="<你的-openai-api-key>"  # 引号是必需的
```

设置完成后，可在 Python 中这样获取：
```python
import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
```

### 方法二：使用 `.env` 文件

也可以在项目根目录下创建 `.env` 文件来存储 API 密钥。

文件内容示例：
```bash
OPENAI_API_KEY=<你的-openai-api-key>
```

然后在 Python 中使用 `python-dotenv` 加载：
```python
from dotenv import load_dotenv 
import os 

load_dotenv()  # 从 .env 文件加载环境变量
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
```

🔐 提示：切勿将 `.env` 文件提交到公共平台（如 GitHub），请将其添加到 `.gitignore`。

### 在 EvoAgentX 中配置并使用 LLM

配置好 API 密钥后，可按如下方式初始化并使用 LLM：
```python
from evoagentx.models import OpenAILLMConfig, OpenAILLM

# 从环境加载 API 密钥
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 定义 LLM 配置
openai_config = OpenAILLMConfig(
    model="gpt-4o-mini",       # 指定模型名称
    openai_key=OPENAI_API_KEY, # 直接传入密钥
    stream=True,               # 启用流式响应
    output_response=True       # 打印响应到标准输出
)

# 初始化语言模型
llm = OpenAILLM(config=openai_config)

# 从 LLM 生成响应
response = llm.generate(prompt="What is Agentic Workflow?")
```

你可以在 [LLM 模块指南](./modules/llm.md) 中找到更多关于支持的 LLM 类型及其参数的详细信息。

## 自动工作流生成与执行

配置完成后，即可在 EvoAgentX 中自动生成并执行智能工作流。本节展示生成工作流、实例化代理并运行的核心步骤。

首先，导入必要的模块：

```python
from evoagentx.workflow import WorkFlowGenerator, WorkFlowGraph, WorkFlow
from evoagentx.agents import AgentManager
```

### 第一步：生成工作流与任务图
使用 `WorkFlowGenerator` 基于自然语言目标自动创建工作流：
```python
goal = "Generate html code for the Tetris game that can be played in the browser."
wf_generator = WorkFlowGenerator(llm=llm)
workflow_graph: WorkFlowGraph = wf_generator.generate_workflow(goal=goal)
```
`WorkFlowGraph` 是一个数据结构，用于存储整体工作流计划，包括任务节点及其关系，但尚未包含可执行的代理。

可选：可视化或保存生成的工作流：
```python
# 可视化工作流且不阻塞后续步骤（可选）。
# 运行过程中节点颜色会更新。使用 display(block=True) 会等待窗口关闭。
workflow_graph.display()

# 将工作流保存为 JSON 文件（可选）
workflow_graph.save_module("/path/to/save/workflow_demo.json")
```
我们提供了一个生成的工作流示例 [here](https://github.com/EvoAgentX/EvoAgentX/blob/main/examples/output/tetris_game/workflow_demo_4o_mini.json)。你可以重新加载保存的工作流：
```python
workflow_graph = WorkFlowGraph.from_file("/path/to/save/workflow_demo.json")
```

### 第二步：创建并管理执行代理

使用 `AgentManager` 基于工作流图实例化并管理代理：
```python
agent_manager = AgentManager()
agent_manager.add_agents_from_workflow(workflow_graph, llm_config=openai_config)
```

### 第三步：执行工作流
代理准备就绪后，可以创建 `WorkFlow` 实例并运行：
```python
workflow = WorkFlow(graph=workflow_graph, agent_manager=agent_manager, llm=llm)
result = workflow.execute()
if result.status == "success":
    print(result.result)
else:
    print(result.displayable_error)
```

`WorkFlow.execute()` 会返回一个 `WorkflowResult` 对象。默认情况下，`result.result` 是结构化的工作流输出，类型为 `dict`。如果你想保留旧的文本抽取行为，可以调用 `workflow.execute(extract_output=True)`。

更多示例请参见 [完整工作流演示](https://github.com/EvoAgentX/EvoAgentX/blob/main/examples/workflow_demo.py)。
