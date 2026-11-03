import json
from typing import List, Optional, Union

from ..models import BaseLLM
from ..prompts.template import ChatTemplate, PromptTemplate, StringTemplate
from ..prompts.tool_calling import TOOL_CALL_FORMAT, TOOL_CALLING_HISTORY_PROMPT
from ..prompts.utils import DEFAULT_SYSTEM_PROMPT
from ..tools.tool import ToolResult


class ContextManager:
    def __init__(
        self,
        llm: BaseLLM,
        system_prompt: Optional[str] = None
    ):
        self.context = []
        self.llm = llm

        # "native": pass tools to the model and exchange structured tool_calls /
        # role:tool messages. "default": describe tools in the prompt and parse a
        # textual <tool_call> block from the response. Decided per-LLM by whether the
        # provider supports the native function-calling protocol.
        if llm.supports_native_tool_calling():
            self.mode = "native"
        else:
            self.mode = "default"
        
        if system_prompt is not None:
            self.context.append({"role": "system", "content": system_prompt})


    def add_system_prompt(self, system_prompt: str):
        if len(self.context) == 0:
            self.context.append({"role": "system", "content": system_prompt})
            return
        
        for i, msg in enumerate(self.context):
            if msg["role"] == "system":
                self.context[i]["content"] += "\n\n" + system_prompt
                return
        
        self.context.insert(0, {"role": "system", "content": system_prompt})


    def replace_system_prompt(self, system_prompt: str):
        for i, msg in enumerate(self.context):
            if msg["role"] == "system":
                self.context[i]["content"] = system_prompt
                return
        
        self.context.insert(0, {"role": "system", "content": system_prompt})


    def add_prompt_template(self, prompt_template: PromptTemplate, sys_msg: Optional[str] = None, **kwargs):
        format_kwargs = dict(kwargs)
        template = prompt_template
        if self.mode == "native":
            # In native mode the tool schema is sent via the model `tools`
            # parameter, so remove both template-owned tools and call-time tools
            # from the rendered prompt.
            format_kwargs["tools"] = None
            if getattr(prompt_template, "tools", None):
                template = prompt_template.copy(tools=None)

        if isinstance(template, ChatTemplate):
            prompts = template.format(**format_kwargs)

            if prompts[0]["role"] == "system":
                self.replace_system_prompt(prompts[0]["content"])
                self.context.extend(prompts[1:])
            else:
                self.context.extend(prompts)

        elif isinstance(template, StringTemplate):
            # `StringTemplate.format()` returns one consolidated prompt, so mirror the
            # `self.prompt` path: `sys_msg` (or the default) becomes the system message
            # and the whole formatted string becomes the user message.
            self.add_system_prompt(sys_msg or DEFAULT_SYSTEM_PROMPT)
            self.add_user_prompt(template.format(**format_kwargs))

        else:
            raise TypeError(f"Invalid prompt template type {type(template)}.")


    def add_user_prompt(self, user_prompt: Union[str, list]):
        if len(self.context) == 0 or self.context[-1]["role"] != "user":
            self.context.append({"role": "user", "content": user_prompt})
            return

        if isinstance(user_prompt, list):
            last_msg = self.context[-1]["content"]

            if isinstance(last_msg, str):
                merged_prompt = [{"type": "text", "text": last_msg}, *user_prompt]
                self.context[-1]["content"] = merged_prompt
            else:
                self.context[-1]["content"] = last_msg + user_prompt
        
        elif isinstance(self.context[-1]["content"], list):
            self.context[-1]["content"] = [
                *self.context[-1]["content"],
                {"type": "text", "text": user_prompt},
            ]
        else:
            self.context[-1]["content"] += "\n\n" + user_prompt


    def add_tool_results(self, tool_results: List[ToolResult]):
        if self.mode == "default":
            formatted_tool_results = []
            for result in tool_results:
                formatted_tool_results.append({
                    "tool_name": result.metadata.tool_name,
                    "result": result.result
                })

            tool_results_str = json.dumps(formatted_tool_results, indent=4, ensure_ascii=False)
            self.context.append({
                "role": "user",
                "content": TOOL_CALLING_HISTORY_PROMPT.format(results=tool_results_str)
            })

        elif self.mode == "native":
            for result in tool_results:
                result_str = json.dumps(result.result, indent=4, ensure_ascii=False)

                self.context.append({
                    "role": "tool",
                    "content": result_str,
                    "tool_call_id": getattr(result, "id", None)
                })


    def add_llm_response(self, llm_response: Optional[str] = None, tool_calls: Optional[List[dict]] = None):

        if llm_response is None and tool_calls is None:
            raise ValueError("Either `llm_response` or `tool_calls` must be provided.")

        if self.mode == "default":
            response = llm_response or ""

            if tool_calls is not None:
                response += TOOL_CALL_FORMAT.format(tool_calls=json.dumps(tool_calls, indent=4, ensure_ascii=False))

            self.context.append({"role": "assistant", "content": response})

        elif self.mode == "native":
            if not tool_calls:
                self.context.append({"role": "assistant", "content": llm_response})
                return
            
            formatted_tool_calls = []

            for tool_call in tool_calls:
                tool_args = json.dumps(tool_call["function_args"], indent=4, ensure_ascii=False)
                formatted_tool_calls.append(
                    {
                        "id": tool_call.get("id"),
                        "function": {
                            "name": tool_call["function_name"],
                            "arguments": tool_args
                        },
                        "type": "function"
                    }
                )

            self.context.append({
                "role": "assistant", 
                "content": llm_response, 
                "tool_calls": formatted_tool_calls
            })


    def get_system_prompt(self) -> str:
        for msg in self.context:
            if msg["role"] == "system":
                return msg["content"]
        return ""
