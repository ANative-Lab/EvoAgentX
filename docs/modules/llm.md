# LLM

## Introduction

The LLM (Large Language Model) module provides a unified interface for interacting with various language model providers in the EvoAgentX framework. It abstracts away provider-specific implementation details, offering a consistent API for generating text, managing costs, and handling responses.

## Supported LLM Providers

EvoAgentX currently supports the following LLM providers:

### OpenAILLM

The primary implementation for accessing OpenAI's language models. It handles authentication, request formatting, and response parsing for models like GPT-4, GPT-3.5-Turbo, and other OpenAI models.

**Basic Usage:**

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

LiteLLM is an adapter for the [LiteLLM project](https://github.com/BerriAI/litellm), which provides a unified Python SDK and proxy server for calling over 100 LLM APIs using the OpenAI API format. It supports providers such as Bedrock, Azure, OpenAI, VertexAI, Cohere, Anthropic, Sagemaker, HuggingFace, Replicate, and Groq. Thanks to this project, the `LiteLLM` model class in EvoAgentX can be used to seamlessly access a wide range of LLM providers through a single interface. 

**Basic Usage:**

To faciliate seamless integration with LiteLLM, you should specify the model name using the naming convention defied in the LiteLLM platform. For example, you need to specify `anthropic/claude-3-opus-20240229` for Claude 3.0 Opus. You can find a full list of supported providers and model names in their official documentation: [https://docs.litellm.ai/docs/providers](https://docs.litellm.ai/docs/providers).


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

SiliconFlowLLM is an adapter for models hosted on the [SiliconFlow platform](https://www.siliconflow.com/), which offers access to both open-source and proprietary models via an OpenAI-compatible API. It enables you to integrate models like Qwen, DeepSeek, or Mixtral by specifying their names using the SiliconFlow platform's naming conventions.

Thanks to SiliconFlow's unified interface, the `SiliconFlowLLM` model class in EvoAgentX allows seamless switching between a variety of powerful LLMs hosted on SiliconFlow using the same API format.

**Basic Usage:**

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

NovitaLLM is an adapter for models hosted on the [Novita AI platform](https://novita.ai/), which offers access to open-source and proprietary LLMs via an OpenAI-compatible API.

Thanks to Novita's OpenAI-compatible interface, the `NovitaLLM` model class in EvoAgentX allows seamless switching between models hosted on Novita using the same API format.

**Basic Usage:**

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

CheaperInferenceLLM is an adapter for [Cheaper Inference](https://cheaperinference.com/), an OpenAI-compatible LLM gateway. One API key gives access to models from several labs.

The `CheaperInferenceLLM` model class in EvoAgentX uses the same API format for all models on Cheaper Inference. You can get an API key at [cheaperinference.com/signup](https://cheaperinference.com/signup).

**Basic Usage:**

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

OpenRouterLLM is an adapter for the [OpenRouter platform](https://openrouter.ai/), which provides access to a wide range of language models from various providers through a unified API. It supports models from providers like Anthropic, Google, Meta, Mistral AI, and more, all accessible through a single interface.

The `OpenRouterLLM` model class in EvoAgentX enables you to easily switch between different models hosted on OpenRouter while maintaining a consistent API format. This makes it simple to experiment with different models and find the best one for your specific use case.

**Basic Usage:**

```python
from evoagentx.models import OpenRouterConfig, OpenRouterLLM

# Configure the model
config = OpenRouterConfig(
    model="openai/gpt-4o-mini",  # or any other model supported by OpenRouter
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

AliyunLLM is an implementation of the EvoAgentX framework for accessing the Aliyun Tongyi Qianqian family of models. It provides seamless integration with Aliyun DashScope API and supports various models of Tongyiqianqian, including qwen-turbo, qwen-plus, qwen-max and so on. We have included reference costs for your consideration; however, please note that actual expenses should be regarded as the definitive amount.

To utilize the DashScope API with AliyunLLM, an API key from Bailian is required. The following steps outline the process:

**Basic Usage:**

Execute the following command in your bash terminal to set the API key:

```bash
export DASHSCOPE_API_KEY="your-api-key-here"
```

You can use python to call the model as below template.

```python
from evoagentx.models import AliyunLLM, AliyunLLMConfig

# Configure the model
config = AliyunLLMConfig(
    model="qwen-turbo",  # you can use qwen-turbo, qwen-plus, qwen-max and so on.
    aliyun_api_key="Your DASHSCOPE_API_KEY",
    temperature=0.7,
    max_tokens=2000,
    stream=False,
    output_response=True
)

# Initialize the model
llm = AliyunLLM(config)

# Generate text
response = llm.generate(
    prompt="Explain quantum computing in simple terms.",
    system_message="You are a helpful assistant that explains complex topics simply."
)
```


### Local LLM

We now support calling local models for your tasks, built on the LiteLLM framework for a familiar user experience. For example, to use Ollama, follow these steps:

1. Download the desired model, such as `ollama3`.
2. Run the model locally.
3. Configure the settings by specifying `api_base` (typically port `11434`) and setting `is_local` to `True`.

You're now ready to leverage your local model seamlessly!

**Basic Usage(ollama):**

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
**Basic Usage(vllm: https://docs.litellm.ai/docs/providers/vllm):**

```python
from evoagentx.models.model_configs import LiteLLMConfig
from evoagentx.models import LiteLLM

# use local model
config = LiteLLMConfig(
    model="hosted_vllm/model_name",  # replace model_name with the actual model name you have
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


## Core Functions

All LLM implementations in EvoAgentX provide a consistent set of core functions for generating text and managing the generation process.

### Generate Function

The `generate` function is the primary method for producing text with language models:

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

#### Inputs 

In EvoAgentX, there are several ways to provide inputs to LLMs using the `generate` function:

**Method 1: Prompt and System Message**

1. **Prompt**: The specific query or instruction for which you want a response. 

2. **System Message** (optional): Instructions that guide the model's overall behavior and role. This sets the context for how the model should respond.

Together, these components are converted into a standardized message format that the language model can understand:

```python
# Simple example with prompt and system message
response = llm.generate(
    prompt="What are three ways to improve productivity?",
    system_message="You are a productivity expert providing concise, actionable advice."
)
```

Behind the scenes, this gets converted into messages with appropriate roles:

```python
messages = [
    {"role": "system", "content": "You are a productivity expert providing concise, actionable advice."},
    {"role": "user", "content": "What are three ways to improve productivity?"}
]
```

**Method 2: Using Messages Directly**

For more complex conversations or when you need precise control over the message format, you can use the `messages` parameter directly:

```python
# Using messages directly for a multi-turn conversation
response = llm.generate(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, who are you?"},
        {"role": "assistant", "content": "I'm an AI assistant designed to help with various tasks."},
        {"role": "user", "content": "Can you help me with programming?"}
    ]
)
```

#### Batch Generation 

For batch processing, you can provide lists of prompts/system messages or list of messages. For example: 

```python
# Batch processing example
responses = llm.generate(
    prompt=["What is machine learning?", "Explain neural networks."],
    system_message=["You are a data scientist.", "You are an AI researcher."]
)
```

##### Parse Modes

EvoAgentX supports several parsing strategies:

1. **"str"**: Uses the raw output as-is for each field defined in the parser.
2. **"json"** (default): Extracts fields from a JSON string in the output.
3. **"xml"**: Extracts content from XML tags matching field names.
4. **"title"**: Extracts content from markdown sections (default format: "## {title}").
5. **"custom"**: Uses a custom parsing function specified by `parse_func`.

!!! note 
    For `'json'`, `'xml'` and `'title'`, you should instruct the LLM (through the `prompt`) to output the content in the specified format that can be parsed by the parser. Otherwise, the parsing will fail. 

    1. For `'json'`, you should instruct the LLM to output a valid JSON string containing keys that match the field names in the parser class. If there are multiple JSON string in the raw LLM output, only the first one will be parsed.  

    2. For `xml`, you should instruct the LLM to output content that contains XML tags matching the field names in the parser class, e.g., `<{field_name}>...</{field_name}>`. If there are multiple XML tags with the same field name, only the first one will be used. 

    3. For `title`, you should instruct the LLM to output content that contains markdown sections with the title exactly matching the field names in the parser class. The default title format is "## {title}". You can change it by setting the `title_format` parameter in the `generate` function, e.g., `generate(..., title_format="### {title}")`. The `title_format` must contain `{title}` as a placeholder for the field name.  

##### Custom Parsing Function

For maximum flexibility, you can define a custom parsing function with `parse_func`:

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
    The `parse_func` should have an input parameter `content` that receives the raw LLM output, and return a dictionary with keys matching the field names in the parser class.  

### Async Generate Function

For applications requiring asynchronous operation, the `async_generate` function provides the same functionality as the `generate` function, but in a non-blocking manner:

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
    Asynchronously generate text based on the prompt and optional system message.

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

### Streaming Responses

EvoAgentX supports streaming responses from LLMs, which allows you to see the model's output as it's being generated token by token, rather than waiting for the complete response. This is especially useful for long-form content generation or providing a more interactive experience.

There are two ways to enable streaming:

#### Configure Streaming in the LLM Config

You can enable streaming when initializing the LLM by setting appropriate parameters in the config:

```python
# Enable streaming at initialization time
config = OpenAILLMConfig(
    model="gpt-4o-mini",
    openai_key="your-api-key",
    stream=True,  # Enable streaming
    output_response=True  # Print tokens to console in real-time
)

llm = OpenAILLM(config=config)

# All calls to generate() will now stream by default
response = llm.generate(
    prompt="Write a story about space exploration."
)
```

#### Enable Streaming in the Generate Method

Alternatively, you can enable streaming for specific generate calls:

```python
# LLM initialized with default non-streaming behavior
config = OpenAILLMConfig(
    model="gpt-4o-mini",
    openai_key="your-api-key"
)

llm = OpenAILLM(config=config)

# Override for this specific call
response = llm.generate(
    prompt="Write a story about space exploration.",
    stream=True,  # Enable streaming for this call only
    output_response=True  # Print tokens to console in real-time
)
```
