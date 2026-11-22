# LLM

## 简介

LLM（大语言模型）模块为 EvoAgentX 框架提供了与各种语言模型提供商交互的统一接口。它抽象了特定提供商的实现细节，为生成文本、管理成本和处理响应提供了一致的 API。

## 支持的 LLM 提供商

EvoAgentX 目前支持以下 LLM 提供商：

### OpenAILLM

这是访问 OpenAI 语言模型的主要实现。它处理 GPT-4、GPT-3.5-Turbo 和其他 OpenAI 模型的认证、请求格式化和响应解析。

**基本用法：**

```python
from evoagentx.models import OpenAILLMConfig, OpenAILLM

# Configure the model
config = OpenAILLMConfig(
    model="gpt-4o-mini",  
    openai_key="your-api-key",
    temperature=0.7,
    max_tokens=1000
)

# Initialize the model
llm = OpenAILLM(config=config)

# Generate text
response = llm.generate(
    prompt="Explain quantum computing in simple terms.",
    system_message="You are a helpful assistant that explains complex topics simply."
)
```

### LiteLLM

LiteLLM 是 [LiteLLM 项目](https://github.com/BerriAI/litellm) 的适配器，该项目提供了一个统一的 Python SDK 和代理服务器，用于使用 OpenAI API 格式调用超过 100 个 LLM API。它支持 Bedrock、Azure、OpenAI、VertexAI、Cohere、Anthropic、Sagemaker、HuggingFace、Replicate 和 Groq 等提供商。多亏了这个项目，EvoAgentX 中的 `LiteLLM` 模型类可以通过单一接口无缝访问各种 LLM 提供商。

**基本用法：**

为了与 LiteLLM 无缝集成，您应该使用 LiteLLM 平台定义的命名约定来指定模型名称。例如，对于 Claude 3.0 Opus，您需要指定 `anthropic/claude-3-opus-20240229`。您可以在其官方文档中找到支持的提供商和模型名称的完整列表：[https://docs.litellm.ai/docs/providers](https://docs.litellm.ai/docs/providers)。

```python
from evoagentx.models import LiteLLMConfig, LiteLLM

# Configure the model
config = LiteLLMConfig(
    model="anthropic/claude-3-opus-20240229", 
    anthropic_key="your-anthropic-api-key",
    temperature=0.7,
    max_tokens=1000
)

# Initialize the model
llm = LiteLLM(config=config)

# Generate text
response = llm.generate(
    prompt="Design a system for autonomous vehicles.",
    system_message="You are an expert in autonomous systems design."
)
```

### SiliconFlowLLM

SiliconFlowLLM 是 [SiliconFlow 平台](https://www.siliconflow.com/) 上托管模型的适配器，该平台通过 OpenAI 兼容的 API 提供对开源和专有模型的访问。它使您能够通过使用 SiliconFlow 平台的命名约定指定模型名称来集成 Qwen、DeepSeek 或 Mixtral 等模型。

得益于 SiliconFlow 的统一接口，EvoAgentX 中的 `SiliconFlowLLM` 模型类允许使用相同的 API 格式在 SiliconFlow 上托管的各种强大 LLM 之间无缝切换。

**基本用法：**

```python
from evoagentx.models import SiliconFlowConfig, SiliconFlowLLM

# Configure the model
config = SiliconFlowConfig(
    model="deepseek-ai/DeepSeek-V3",
    siliconflow_key="your-siliconflow-api-key",
    temperature=0.7,
    max_tokens=1000
)

# Initialize the model
llm = SiliconFlowLLM(config=config)

# Generate text
response = llm.generate(
    prompt="Write a poem about artificial intelligence.",
    system_message="You are a creative poet."
)
```

### NovitaLLM

NovitaLLM 是 [Novita AI 平台](https://novita.ai/) 上托管模型的适配器，该平台通过 OpenAI 兼容的 API 提供对开源和专有模型的访问。

得益于 Novita 兼容 OpenAI 的接口，EvoAgentX 中的 `NovitaLLM` 模型类允许使用相同的 API 格式在 Novita 上托管的不同模型之间无缝切换。

**基本用法：**

```python
from evoagentx.models import NovitaConfig, NovitaLLM

# Configure the model
config = NovitaConfig(
    model="deepseek/deepseek-v4-pro",
    novita_key="your-novita-api-key",
    temperature=0.7,
    max_tokens=1000
)

# Initialize the model
llm = NovitaLLM(config=config)

# Generate text
response = llm.generate(
    prompt="Write a poem about artificial intelligence.",
    system_message="You are a creative poet."
)
```

### CheaperInferenceLLM

CheaperInferenceLLM 是 [Cheaper Inference](https://cheaperinference.com/) 的适配器。Cheaper Inference 是一个兼容 OpenAI 的 LLM 网关，使用一个 API 密钥即可访问多家实验室的模型。

EvoAgentX 中的 `CheaperInferenceLLM` 模型类对 Cheaper Inference 上的所有模型使用相同的 API 格式。可以在 [cheaperinference.com/signup](https://cheaperinference.com/signup) 获取 API 密钥。

**基本用法：**

```python
from evoagentx.models import CheaperInferenceConfig, CheaperInferenceLLM

# Configure the model
config = CheaperInferenceConfig(
    model="gpt-5.4-mini",
    cheaperinference_key="your-cheaperinference-api-key",
    temperature=0.7,
    max_tokens=1000
)

# Initialize the model
llm = CheaperInferenceLLM(config=config)

# Generate text
response = llm.generate(
    prompt="Write a poem about artificial intelligence.",
    system_message="You are a creative poet."
)
```

### OpenRouterLLM

OpenRouterLLM 是 [OpenRouter 平台](https://openrouter.ai/) 的适配器，该平台通过统一的 API 提供对各种提供商的语言模型的访问。它支持来自 Anthropic、Google、Meta、Mistral AI 等提供商的模型，所有这些都可以通过单一接口访问。

EvoAgentX 中的 `OpenRouterLLM` 模型类使您能够轻松地在 OpenRouter 上托管的不同模型之间切换，同时保持一致的 API 格式。这使得您可以轻松尝试不同的模型，为您的特定用例找到最佳选择。

**基本用法：**

```python
from evoagentx.models import OpenRouterConfig, OpenRouterLLM

# Configure the model
config = OpenRouterConfig(
    model="openai/gpt-4o-mini",  # 或 OpenRouter 支持的任何其他模型
    openrouter_key="your-openrouter-api-key",
    temperature=0.7,
    max_tokens=1000
)

# Initialize the model
llm = OpenRouterLLM(config=config)

# Generate text
response = llm.generate(
    prompt="Analyze the impact of artificial intelligence on healthcare.",
    system_message="You are an AI ethics expert specializing in healthcare applications."
)
```

### Aliyun LLM

AliyunLLM 是 EvoAgentX 框架的实现，用于访问 Aliyun通义千文系列模型。它提供了与 Aliyun DashScope API 的无缝集成，并支持 Tongyi Qianqian 的多种模型，包括 qwen-turbo、qwen-plus、qwen-max 等。我们已提供参考成本供您参考；然而，请以实际费用为准。

要使用 DashScope API 与 AliyunLLM 进行集成，需要一个来自百炼平台的 API 密钥。以下步骤概述了该过程：

**基本用法**

在您的 bash 终端中执行以下命令以设置 API 密钥：

```bash
export DASHSCOPE_API_KEY="您的-api-密钥"
```

使用python调用模型的样例如下：

```python
from evoagentx.models import AliyunLLM, AliyunLLMConfig

# 配置模型
config = AliyunLLMConfig(
    model="qwen-turbo",  # 您可以使用 qwen-turbo、qwen-plus、qwen-max 等。
    aliyun_api_key="您的 DASHSCOPE_API_KEY",
    temperature=0.7,
    max_tokens=2000,
    stream=False,
    output_response=True
)

# 初始化模型
llm = AliyunLLM(config)

# 生成文本
response = llm.generate(
    prompt="Explain quantum computing in simple terms.",
    system_message="You are a helpful assistant that explains complex topics simply."
)
```

### 本地 LLM

我们现已支持在任务重中调用本地模型，这种方法基于 LiteLLM 框架打造，提供熟悉的用户体验。以 Ollama 为例，请您可以以下步骤操作：

1. 下载您需要的模型，例如 `ollama3`。
2. 在本地运行该模型。
3. 配置设置，指定 `api_base`（通常为端口 `11434`）并将 `is_local` 设置为 `True`。

现在，您可以无缝使用本地模型了！

**基本用法：**

```python

from evoagentx.models.model_configs import LiteLLMConfig
from evoagentx.models import LiteLLM

# use local model
config = LiteLLMConfig(
    model="ollama/llama3",
    api_base="http://localhost:11434",
    is_local=True,
    temperature=0.7,
    max_tokens=1000,
    output_response=True
)

# Generate 
llm = LiteLLM(config)
response = llm.generate(prompt="What is Agentic Workflow?")

```

## 核心功能

EvoAgentX 中的所有 LLM 实现都提供了一组一致的核心功能，用于生成文本和管理生成过程。

### Generate 函数

`generate` 函数是使用语言模型生成文本的主要方法：

```python
def generate(
    self,
    prompt: Optional[Union[str, List[str]]] = None,
    system_message: Optional[Union[str, List[str]]] = None,
    messages: Optional[Union[List[dict],List[List[dict]]]] = None,
    parser: Optional[Type[LLMOutputParser]] = None,
    parse_mode: Optional[str] = "json", 
    parse_func: Optional[Callable] = None,
    **kwargs
) -> Union[LLMOutputParser, List[LLMOutputParser]]:
    """
    Generate text based on the prompt and optional system message.

    Args:
        prompt: Input prompt(s) to the LLM.
        system_message: System message(s) for the LLM.
        messages: Chat message(s) for the LLM, already in the required format (either `prompt` or `messages` must be provided).
        parser: Parser class to use for processing the output into a structured format.
        parse_mode: The mode to use for parsing, must be the `parse_mode` supported by the `parser`. 
        parse_func: A function to apply to the parsed output.
        **kwargs: Additional generation configuration parameters.
        
    Returns:
        For single generation: An LLMOutputParser instance.
        For batch generation: A list of LLMOutputParser instances.
    """
```

#### 输入方式

在 EvoAgentX 中，有几种使用 `generate` 函数向 LLM 提供输入的方式：

**方法 1：提示和系统消息**

1. **提示（Prompt）**：您想要获得响应的具体查询或指令。

2. **系统消息（System Message）**（可选）：指导模型整体行为和角色的指令。这为模型应该如何响应设置了上下文。

这些组件被转换为语言模型可以理解的标准消息格式：

```python
# Simple example using prompt and system message
response = llm.generate(
    prompt="What are three ways to improve productivity?",
    system_message="You are a productivity expert providing concise, actionable advice."
)
```

在后台，这被转换为具有适当角色的消息：

```python
messages = [
    {"role": "system", "content": "You are a productivity expert providing concise, actionable advice."},
    {"role": "user", "content": "What are three ways to improve productivity?"}
]
```

**方法 2：直接使用消息**

对于更复杂的对话或当您需要精确控制消息格式时，您可以直接使用 `messages` 参数：

```python
# Direct use of messages for multi-turn conversation
response = llm.generate(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, who are you?"},
        {"role": "assistant", "content": "I'm an AI assistant designed to help with various tasks."},
        {"role": "user", "content": "Can you help me with programming?"}
    ]
)
```

#### 批量生成

对于批量处理，您可以提供提示/系统消息列表或消息列表。例如：

```python
# Batch processing example
responses = llm.generate(
    prompt=["What is machine learning?", "Explain neural networks."],
    system_message=["You are a data scientist.", "You are an AI researcher."]
)
```

#### 输出解析

`generate` 函数提供了灵活的选项来解析和结构化来自语言模型的原始文本输出：

- **parser**：接受一个类（通常继承自 `LLMOutputParser/ActionOutput`），该类定义了解析输出的结构。如果未提供，LLM 输出将不会被解析。在这两种情况下，都可以通过返回对象的 `.content` 属性访问原始 LLM 输出。
- **parse_mode**：确定如何将原始 LLM 输出解析为解析器定义的结构，有效选项为：`'str'`、`'json'`（默认）、`'xml'`、`'title'`、`'custom'`。
- **parse_func**：用于在更复杂的场景中处理解析的自定义函数，仅在 `parse_mode` 为 `'custom'` 时使用。

结构化输出示例：
```python
from evoagentx.models import LLMOutputParser 
from pydantic import Field

class CodeWriterOutput(LLMOutputParser):
    thought: str = Field(description="Thought process for writing the code") 
    code: str = Field(description="The generated code")

prompt = """
Write a Python function to calculate Fibonacci numbers. 

Your output should always be in the following format:

## thought 
[Your thought process for writing the code]

## code
[The generated code]
"""
response = llm.generate(
    prompt=prompt,
    parser=CodeWriterOutput,
    parse_mode="title"
)

print("Thought:\n", response.thought)
print("Code:\n", response.code)
```

##### 解析模式

EvoAgentX 支持几种解析策略：

1. **"str"**：直接使用原始输出作为解析器中定义的每个字段。
2. **"json"**（默认）：从输出中的 JSON 字符串提取字段。
3. **"xml"**：从与字段名称匹配的 XML 标签中提取内容。
4. **"title"**：从 markdown 章节中提取内容（默认格式："## {title}"）。
5. **"custom"**：使用由 `parse_func` 指定的自定义解析函数。

!!! note 
    对于 `'json'`、`'xml'` 和 `'title'`，您应该通过 `prompt` 指示 LLM 以可以被解析器解析的指定格式输出内容。否则，解析将失败。

    1. 对于 `'json'`，您应该指示 LLM 输出一个包含与解析器类中的字段名称匹配的键的有效 JSON 字符串。如果原始 LLM 输出中有多个 JSON 字符串，只会解析第一个。

    2. 对于 `xml`，您应该指示 LLM 输出包含与解析器类中的字段名称匹配的 XML 标签的内容，例如 `<{field_name}>...</{field_name}>`。如果有多个具有相同字段名称的 XML 标签，只会使用第一个。

    3. 对于 `title`，您应该指示 LLM 输出包含标题与解析器类中的字段名称完全匹配的 markdown 章节的内容。默认标题格式是 "## {title}"。您可以通过在 `generate` 函数中设置 `title_format` 参数来更改它，例如 `generate(..., title_format="### {title}")`。`title_format` 必须包含 `{title}` 作为字段名称的占位符。

##### 自定义解析函数

为了获得最大的灵活性，您可以使用 `parse_func` 定义自定义解析函数：

```python
from evoagentx.models import LLMOutputParser
from evoagentx.core.module_utils import extract_code_block

class CodeOutput(LLMOutputParser):
    code: str = Field(description="The generated code")

# Use custom parsing
response = llm.generate(
    prompt="Write a Python function to calculate Fibonacci numbers.",
    parser=CodeOutput,
    parse_mode="custom",
    parse_func=lambda content: {"code": extract_code_block(content)[0]}
)
```

!!! note 
    解析函数应该有一个接收原始 LLM 输出的输入参数 `content`，并返回一个字典，其键与解析器类中的字段名称匹配。

### 异步生成函数

对于需要异步操作的应用程序，`async_generate` 函数提供了与 `generate` 函数相同的功能，但以非阻塞方式运行：

```python
async def async_generate(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        system_message: Optional[Union[str, List[str]]] = None,
        messages: Optional[Union[List[dict],List[List[dict]]]] = None,
        parser: Optional[Type[LLMOutputParser]] = None,
        parse_mode: Optional[str] = "json", 
        parse_func: Optional[Callable] = None,
        **kwargs
    ) -> Union[LLMOutputParser, List[LLMOutputParser]]:
    """
    基于提示和可选的系统消息异步生成文本。

    参数：
        prompt: 输入到 LLM 的提示。
        system_message: LLM 的系统消息。
        messages: LLM 的聊天消息，已经是所需格式（必须提供 `prompt` 或 `messages` 之一）。
        parser: 用于将输出处理为结构化格式的解析器类。
        parse_mode: 用于解析的模式，必须是 `parser` 支持的 `parse_mode`。
        parse_func: 应用于解析输出的函数。
        **kwargs: 额外的生成配置参数。
        
    返回：
        单次生成：一个 LLMOutputParser 实例。
        批量生成：LLMOutputParser 实例列表。
    """
```

### 流式响应

EvoAgentX 支持来自 LLM 的流式响应，这使您可以在生成过程中逐令牌查看模型的输出，而不是等待完整响应。这对于生成长篇内容或提供更交互式的体验特别有用。

有两种启用流式传输的方式：

#### 在 LLM 配置中配置流式传输

您可以在初始化 LLM 时通过设置配置中的适当参数来启用流式传输：

```python
# 在初始化时启用流式传输
config = OpenAILLMConfig(
    model="gpt-4o-mini",
    openai_key="your-api-key",
    stream=True,  # 启用流式传输
    output_response=True  # 实时将令牌打印到控制台
)

llm = OpenAILLM(config=config)

# 现在所有对 generate() 的调用都将默认使用流式传输
response = llm.generate(
    prompt="Write a story about space exploration."
)
```

#### 在生成方法中启用流式传输

或者，您可以为特定的生成调用启用流式传输：

```python
# 使用默认非流式行为初始化的 LLM
config = OpenAILLMConfig(
    model="gpt-4o-mini",
    openai_key="your-api-key"
)

llm = OpenAILLM(config=config)

# 仅为此特定调用覆盖设置
response = llm.generate(
    prompt="Write a story about space exploration.",
    stream=True,  # 仅为此调用启用流式传输
    output_response=True  # 实时将令牌打印到控制台
)
```
