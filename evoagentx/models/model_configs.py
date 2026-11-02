
from pydantic import BaseModel, Field
from typing import Optional, Union, List

# import torch 

from ..core.base_config import BaseConfig


#### LLM Configs
class LLMConfig(BaseConfig):

    llm_type: str
    model: str 
    output_response: bool = Field(default=False, description="Whether to output LLM response.")


class OpenAILLMConfig(LLMConfig):

    llm_type: str = "OpenAILLM"
    openai_key: Optional[str] = Field(default=None, description="the API key used to authenticate OpenAI requests")

    # generation parameters
    temperature: Optional[float] = Field(default=None, description="the temperature used to scaling logits")
    max_tokens : Optional[int] = Field(default=None, description="maximum number of generated tokens. This value is now deprecated in favor of max_completion_tokens, and is not compatible with o1 series models.")
    max_completion_tokens: Optional[int] = Field(default=None, description="An upper bound for the number of tokens that can be generated for a completion, including visible output tokens and reasoning tokens. Commonly used in OpenAI's o1 series models.")
    top_p: Optional[float] = Field(default=None, description="Only sample from tokens with cumulative probability greater than top_p when generating text.")
    n: Optional[int] = Field(default=None, description="How many chat completion choices to generate for each input message.")
    stream: Optional[bool] = Field(default=None, description=" If set to true, it sends partial message deltas. Tokens will be sent as they become available, with the stream terminated by a [DONE] message.")
    stream_options: Optional[dict] = Field(default=None, description="Options for streaming response. Only set this when you set stream: true")
    timeout: Optional[Union[float, int]] = Field(default=None, description="Timeout in seconds for completion requests (Defaults to 600 seconds)")

    # tools 
    tools: Optional[List] = Field(default=None, description="A list of tools the model may call. Currently, only functions are supported as a tool. Use this to provide a list of functions the model may generate JSON inputs for.")
    tool_choice: Optional[Union[str, dict]] = Field(default=None, description="Controls which (if any) function is called by the model. none means the model will not call a function and instead generates a message. auto means the model can pick between generating a message or calling a function. required forces the model to call a tool. Specifying a particular function via {\"type\": \"function\", \"function\": {\"name\": \"my_function\"}} forces the model to call that function.")
    parallel_tool_calls: Optional[bool] = Field(default=None, description="Whether to enable parallel function calling during tool use. OpenAI default is true.")
    
    # reasoning parameters 
    reasoning_effort: Optional[str] = Field(default=None, description="Constrains effort on reasoning for reasoning models. Currently supported values are low, medium, and high. Reducing reasoning effort can result in faster responses and fewer tokens used on reasoning in a response.")

    # token probabilities
    logprobs: Optional[bool] = Field(default=None, description="Whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the content of message.")
    top_logprobs: Optional[int] = Field(default=None, description="An integer between 0 and 5 specifying the number of most likely tokens to return at each token position, each with an associated log probability. logprobs must be set to true if this parameter is used.")

    # predicted outputs 
    prediction: Optional[dict] = Field(default=None, description="Configuration for a Predicted Output, which can greatly improve response times when large parts of the model response are known ahead of time. This is most common when you are regenerating a file with only minor changes to most of the content.")

    # output format
    modalities: Optional[List] = Field(default=None, description="Output types that you would like the model to generate for this request. Most models are capable of generating text, which is the default: [\"text\"]")
    response_format: Optional[Union[BaseModel, dict]] = Field(default=None, description=" An object specifying the format that the model must output.")

# ==== Azure OpenAI Configuration ====
class AzureOpenAIConfig(LLMConfig):
    llm_type: str = "AzureOpenAILLM"
    azure_endpoint: str = Field(..., description="Azure OpenAI service endpoint URL")
    azure_key: str = Field(..., description="Azure OpenAI API key for authentication")
    api_version: Optional[str] = Field(default="2024-12-01-preview", description="Azure OpenAI API version to use")
    # 'model' field inherited from LLMConfig will be used to specify the deployment name
    # generation parameters (temperature, max_tokens, etc.) inherited from OpenAILLMConfig


class LiteLLMConfig(LLMConfig):

    llm_type: str = "LiteLLM"
    api_base: Optional[str] = Field(default=None, description="Base URL for the LLM API (e.g., http://localhost:11434/v1 for Ollama)") 
    is_local: Optional[bool] = Field(default=False, description="Whether the model is running locally (e.g., Ollama)")
    api_key: Optional[str] = Field(default=None, description="the API key used to authenticate generic OpenAI-compatible requests (e.g., LM Studio, FastChat, LocalAI)")

    # LLM keys
    openai_key: Optional[str] = Field(default=None, description="the API key used to authenticate OpenAI requests")
    anthropic_key: Optional[str] = Field(default=None, description="the API key used to authenticate Anthropic requests")
    deepseek_key: Optional[str] = Field(default=None, description="the API key used to authenticate Deepseek requests")
    gemini_key: Optional[str] = Field(default=None, description="the API key used to authenticate Gemini requests")
    meta_llama_key: Optional[str] = Field(default=None, description="the API key used to authenticate Meta Llama requests")
    openrouter_key: Optional[str] = Field(default=None, description="the API key used to authenticate OpenRouter requests")
    openrouter_base: Optional[str] = Field(default="https://openrouter.ai/api/v1", description="the base URL used to authenticate OpenRouter requests")
    perplexity_key: Optional[str] = Field(default=None, description="the API key used to authenticate Perplexity requests")
    groq_key: Optional[str] = Field(default=None, description="the API key used to authenticate Groq requests")
    
    # Azure OpenAI keys
    azure_endpoint: Optional[str] = Field(default=None, description="Azure OpenAI service endpoint URL")
    azure_key: Optional[str] = Field(default=None, description="Azure OpenAI API key for authentication")
    api_version: Optional[str] = Field(default=None, description="Azure OpenAI API version to use")

    # generation parameters 
    temperature: Optional[float] = Field(default=None, description="the temperature used to scaling logits")
    max_tokens : Optional[int] = Field(default=None, description="maximum number of generated tokens")
    max_completion_tokens: Optional[int] = Field(default=None, description="An upper bound for the number of tokens that can be generated for a completion, including visible output tokens and reasoning tokens. Commonly used in OpenAI's o1 series models.")
    top_p: Optional[float] = Field(default=None, description="Only sample from tokens with cumulative probability greater than top_p when generating text.")
    n: Optional[int] = Field(default=None, description="How many chat completion choices to generate for each input message.")
    stream: Optional[bool] = Field(default=None, description=" If set to true, it sends partial message deltas. Tokens will be sent as they become available, with the stream terminated by a [DONE] message.")
    stream_options: Optional[dict] = Field(default=None, description="Options for streaming response. Only set this when you set stream: true")
    timeout: Optional[Union[float, int]] = Field(default=None, description="Timeout in seconds for completion requests (Defaults to 600 seconds)")

    # tools
    tools: Optional[List] = Field(default=None, description="A list of tools the model may call. Currently, only functions are supported as a tool. Use this to provide a list of functions the model may generate JSON inputs for.")
    tool_choice: Optional[Union[str, dict]] = Field(default=None, description="Controls which (if any) function is called by the model. none means the model will not call a function and instead generates a message. auto means the model can pick between generating a message or calling a function. required forces the model to call a tool. Specifying a particular function via {\"type\": \"function\", \"function\": {\"name\": \"my_function\"}} forces the model to call that function.")
    parallel_tool_calls: Optional[bool] = Field(default=None, description="Whether to enable parallel function calling during tool use. OpenAI default is true.")

    # token probabilities
    logprobs: Optional[bool] = Field(default=None, description="Whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the content of message.")
    top_logprobs: Optional[int] = Field(default=None, description="An integer between 0 and 5 specifying the number of most likely tokens to return at each token position, each with an associated log probability. logprobs must be set to true if this parameter is used.")

    # output format
    response_format: Optional[Union[BaseModel, dict]] = Field(default=None, description=" An object specifying the format that the model must output.")

    def __str__(self):
        return self.model


class SiliconFlowConfig(LLMConfig):

    # LLM keys
    llm_type: str = "SiliconFlowLLM"
    siliconflow_key: Optional[str] = Field(default=None, description="the API key used to authenticate SiliconFlow requests") 

    # generation parameters 
    temperature: Optional[float] = Field(default=None, description="the temperature used to scaling logits")
    max_tokens : Optional[int] = Field(default=None, description="maximum number of generated tokens")
    max_completion_tokens: Optional[int] = Field(default=None, description="An upper bound for the number of tokens that can be generated for a completion, including visible output tokens and reasoning tokens. Commonly used in OpenAI's o1 series models.")
    top_p: Optional[float] = Field(default=None, description="Only sample from tokens with cumulative probability greater than top_p when generating text.")
    n: Optional[int] = Field(default=None, description="How many chat completion choices to generate for each input message.")
    stream: Optional[bool] = Field(default=None, description=" If set to true, it sends partial message deltas. Tokens will be sent as they become available, with the stream terminated by a [DONE] message.")
    stream_options: Optional[dict] = Field(default=None, description="Options for streaming response. Only set this when you set stream: true")
    timeout: Optional[Union[float, int]] = Field(default=None, description="Timeout in seconds for completion requests (Defaults to 600 seconds)")

    # tools
    tools: Optional[List] = Field(default=None, description="A list of tools the model may call. Currently, only functions are supported as a tool. Use this to provide a list of functions the model may generate JSON inputs for.")
    tool_choice: Optional[Union[str, dict]] = Field(default=None, description="Controls which (if any) function is called by the model. none means the model will not call a function and instead generates a message. auto means the model can pick between generating a message or calling a function. required forces the model to call a tool. Specifying a particular function via {\"type\": \"function\", \"function\": {\"name\": \"my_function\"}} forces the model to call that function.")
    parallel_tool_calls: Optional[bool] = Field(default=None, description="Whether to enable parallel function calling during tool use. OpenAI default is true.")

    # token probabilities
    logprobs: Optional[bool] = Field(default=None, description="Whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the content of message.")
    top_logprobs: Optional[int] = Field(default=None, description="An integer between 0 and 5 specifying the number of most likely tokens to return at each token position, each with an associated log probability. logprobs must be set to true if this parameter is used.")

    # output format
    response_format: Optional[Union[BaseModel, dict]] = Field(default=None, description=" An object specifying the format that the model must output.")

    def __str__(self):
        return self.model


class AtlasCloudConfig(LLMConfig):

    llm_type: str = "AtlasCloudLLM"
    atlascloud_key: Optional[str] = Field(
        default=None,
        description="the API key used to authenticate Atlas Cloud requests",
    )

    # generation parameters
    temperature: Optional[float] = Field(default=None, description="the temperature used to scaling logits")
    max_tokens: Optional[int] = Field(default=None, description="maximum number of generated tokens")
    max_completion_tokens: Optional[int] = Field(default=None, description="An upper bound for the number of tokens that can be generated for a completion, including visible output tokens and reasoning tokens.")
    top_p: Optional[float] = Field(default=None, description="Only sample from tokens with cumulative probability greater than top_p when generating text.")
    n: Optional[int] = Field(default=None, description="How many chat completion choices to generate for each input message.")
    stream: Optional[bool] = Field(default=None, description="If set to true, it sends partial message deltas as they become available.")
    stream_options: Optional[dict] = Field(default=None, description="Options for streaming responses.")
    timeout: Optional[Union[float, int]] = Field(default=None, description="Timeout in seconds for completion requests.")

    # tools and output format
    tools: Optional[List] = Field(default=None, description="A list of tools the model may call.")
    tool_choice: Optional[Union[str, dict]] = Field(default=None, description="Controls which tool the model may call.")
    parallel_tool_calls: Optional[bool] = Field(default=None, description="Whether to enable parallel function calling during tool use.")
    response_format: Optional[Union[BaseModel, dict]] = Field(default=None, description="An object specifying the format that the model must output.")

    def __str__(self):
        return self.model


class NovitaConfig(LLMConfig):

    # LLM keys
    llm_type: str = "NovitaLLM"
    novita_key: Optional[str] = Field(default=None, description="the API key used to authenticate Novita AI requests")

    # generation parameters
    temperature: Optional[float] = Field(default=None, description="the temperature used to scaling logits")
    max_tokens : Optional[int] = Field(default=None, description="maximum number of generated tokens")
    max_completion_tokens: Optional[int] = Field(default=None, description="An upper bound for the number of tokens that can be generated for a completion, including visible output tokens and reasoning tokens. Commonly used in OpenAI's o1 series models.")
    top_p: Optional[float] = Field(default=None, description="Only sample from tokens with cumulative probability greater than top_p when generating text.")
    n: Optional[int] = Field(default=None, description="How many chat completion choices to generate for each input message.")
    stream: Optional[bool] = Field(default=None, description=" If set to true, it sends partial message deltas. Tokens will be sent as they become available, with the stream terminated by a [DONE] message.")
    stream_options: Optional[dict] = Field(default=None, description="Options for streaming response. Only set this when you set stream: true")
    timeout: Optional[Union[float, int]] = Field(default=None, description="Timeout in seconds for completion requests (Defaults to 600 seconds)")

    # tools
    tools: Optional[List] = Field(default=None, description="A list of tools the model may call. Currently, only functions are supported as a tool. Use this to provide a list of functions the model may generate JSON inputs for.")
    tool_choice: Optional[Union[str, dict]] = Field(default=None, description="Controls which (if any) function is called by the model. none means the model will not call a function and instead generates a message. auto means the model can pick between generating a message or calling a function. required forces the model to call a tool. Specifying a particular function via {\"type\": \"function\", \"function\": {\"name\": \"my_function\"}} forces the model to call that function.")
    parallel_tool_calls: Optional[bool] = Field(default=None, description="Whether to enable parallel function calling during tool use. OpenAI default is true.")

    # token probabilities
    logprobs: Optional[bool] = Field(default=None, description="Whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the content of message.")
    top_logprobs: Optional[int] = Field(default=None, description="An integer between 0 and 5 specifying the number of most likely tokens to return at each token position, each with an associated log probability. logprobs must be set to true if this parameter is used.")

    # output format
    response_format: Optional[Union[BaseModel, dict]] = Field(default=None, description=" An object specifying the format that the model must output.")

    def __str__(self):
        return self.model


# def get_default_device():
#     return "cuda" if torch.cuda.is_available() else "cpu"

class OpenRouterConfig(LLMConfig):
    llm_type: str = "OpenRouterLLM"
    
    # LLM keys
    openrouter_key: Optional[str] = Field(default=None, description="the API key used to authenticate OpenRouter requests")
    openrouter_base: Optional[str] = Field(default="https://openrouter.ai/api/v1", description="the base URL used to authenticate OpenRouter requests")
    openrouter_model_base: Optional[str] = Field(default="https://openrouter.ai/api/v1/models", description="the model url to access model details")
    # generation parameters 
    temperature: Optional[float] = Field(default=None, description="the temperature used to scaling logits")
    top_p: Optional[float] = Field(default=None, description="This setting limits the model's choices to a percentage of likely tokens: only the top tokens whose probabilities add up to P. A lower value makes the model's responses more predictable, while the default setting allows for a full range of token choices.")
    top_k: Optional[int] = Field(default=None, description="This limits the model's choice of tokens at each step, making it choose from a smaller set. A value of 1 means the model will always pick the most likely next token, leading to predictable results.")
    frequency_penalty: Optional[float] = Field(default=None, description="Controls repetition of tokens based on frequency in input. Range: -2.0 to 2.0. Higher values reduce repetition of frequent tokens.")
    presence_penalty: Optional[float] = Field(default=None, description="Adjusts repetition of specific tokens from input. Range: -2.0 to 2.0. Higher values reduce repetition.")
    repetition_penalty: Optional[float] = Field(default=None, description="Reduces repetition of tokens from input. Range: 0.0 to 2.0. Higher values make repetition less likely.")
    min_p: Optional[float] = Field(default=None, description="Minimum probability for a token relative to most likely token. Range: 0.0 to 1.0.")
    top_a: Optional[float] = Field(default=None, description="Consider only tokens with 'sufficiently high' probabilities based on most likely token. Range: 0.0 to 1.0.")
    seed: Optional[int] = Field(default=None, description="For deterministic sampling. Repeated requests with same seed and parameters should return same result.")
    max_tokens: Optional[int] = Field(default=None, description="Upper limit for number of tokens model can generate. Must be 1 or above.")
    logit_bias: Optional[dict] = Field(default=None, description="Map of token IDs to bias values (-100 to 100) to adjust token selection probabilities.")
    logprobs: Optional[bool] = Field(default=None, description="Whether to return log probabilities of output tokens.")
    top_logprobs: Optional[int] = Field(default=None, description="Number of most likely tokens to return at each position (0-20) with log probabilities.")
    response_format: Optional[Union[BaseModel, dict]] = Field(default=None, description="Forces model to produce specific output format (e.g. JSON mode).")
    structured_outputs: Optional[bool] = Field(default=None, description="Whether model can return structured outputs using response_format json_schema.")
    stop: Optional[List[str]] = Field(default=None, description="Stop generation if model encounters any token in this array.")
    tools: Optional[List] = Field(default=None, description="Tool calling parameter following OpenAI's tool calling request shape.")
    tool_choice: Optional[Union[str, dict]] = Field(default=None, description="Controls which tool is called by model. Can be 'none', 'auto', 'required', or specific tool configuration.")

    stream: Optional[bool] = Field(default=None, description="If set to true, it sends partial message deltas. Tokens will be sent as they become available, with the stream terminated by a [DONE] message.")
    extra_body: Optional[dict] = Field(default=None, description="Additional request body parameters for provider-specific features.")
    enable_prompt_caching: Optional[bool] = Field(default=False, description="Opt into OpenRouter prompt caching for providers that require explicit cache_control breakpoints (Anthropic, Gemini, Qwen). Defaults to False because cache WRITES on these providers are billed at a premium (e.g. Anthropic ~1.25x input price, Gemini adds storage fees, Qwen applies a write multiplier), so a single non-repeated call costs more than without caching — it only pays off across repeated calls sharing a prompt prefix. Agent loops such as CustomizeAction enable it per-call. Automatic-caching providers (OpenAI, DeepSeek, Grok, Moonshot) are unaffected.")

    def __str__(self):
        return self.model


class AliyunLLMConfig(LLMConfig):
    llm_type: str = "AliyunLLM"
    aliyun_api_key: Optional[str] = Field(default=None, description="The API key used to authenticate Aliyun requests (i.e. the DASHSCOPE_API_KEY)")
    aliyun_base_url: Optional[str] = Field(default=None, description="The OpenAI-compatible base URL for the Aliyun Bailian endpoint. It is workspace-specific, e.g. 'https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1'")

    # generation parameters
    temperature: Optional[float] = Field(default=None, description="The temperature used to control randomness in generation. Higher values increase diversity.")
    top_p: Optional[float] = Field(default=None, description="Nucleus sampling parameter. Only sample from tokens with cumulative probability greater than top_p.")
    max_tokens: Optional[int] = Field(default=None, description="Maximum number of tokens to generate in the response.")
    top_k: Optional[int] = Field(default=None, description="Top-k sampling parameter. Only sample from the top k tokens at each step.")
    repetition_penalty: Optional[float] = Field(default=None, description="Penalty for repeated tokens. Higher values discourage repetition.")
    stream: Optional[bool] = Field(default=None, description="If set to true, enables streaming response where partial results are sent as they become available.")
    timeout: Optional[Union[float, int]] = Field(default=None, description="Timeout in seconds for completion requests (defaults to 600 seconds).")

    # tools
    tools: Optional[List] = Field(default=None, description="A list of tools or functions the model may call. Aliyun supports function calling for specific models.")
    tool_choice: Optional[Union[str, dict]] = Field(default=None, description="Controls whether the model should call a tool. Options include 'none' (no tool call), 'auto' (model decides), 'required' (force a tool call), or a specific tool via {\"type\": \"function\", \"function\": {\"name\": \"my_function\"}}.")
    
    # model-specific parameters
    enable_search: Optional[bool] = Field(default=None, description="Whether to enable web search augmentation for the model, if supported.")

    # output format
    response_format: Optional[Union[BaseModel, dict]] = Field(default=None, description="Specifies the format of the model output, e.g., JSON schema for structured responses.")
    output_modalities: Optional[List] = Field(default=None, description="Output types the model should generate, e.g., ['text', 'image'] for multimodal models.")

    # token probabilities
    logprobs: Optional[bool] = Field(default=None, description="Whether to return log probabilities of output tokens. Supported by some Aliyun models.")
    top_logprobs: Optional[int] = Field(default=None, description="Number of most likely tokens to return with log probabilities at each position. Requires logprobs to be true.")
    
    def __str__(self):
        return self.model
