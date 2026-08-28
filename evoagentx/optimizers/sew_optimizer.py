import yaml 
import regex
import random
import inspect
import numpy as np
from pydantic import Field 
from copy import deepcopy
import xml.etree.ElementTree as ET
from typing import Literal, Union, Optional, List

from .optimizer import Optimizer
from ..core.logging import logger
from ..models.base_model import BaseLLM 
from ..benchmark.benchmark import Benchmark
from ..workflow.action_graph import ActionGraph
from ..core.callbacks import suppress_logger_info
from ..workflow.workflow_graph import SequentialWorkFlowGraph
from ..prompts.workflow.sew_optimizer import mutation_prompts, thinking_styles

VALID_SCHEMES = ["python", "yaml", "code", "core", "bpmn"]

class SEWWorkFlowScheme:

    """
    The scheme of the workflow for SEW optimizer.
    """
    def __init__(self, graph: SequentialWorkFlowGraph, **kwargs):
        self.graph = graph # the workflow graph to be transformed
        self.kwargs = kwargs

    def convert_to_scheme(self, scheme: str) -> str:
        """
        Transform the WorkflowGraph to the desired scheme.
        """
        if scheme not in VALID_SCHEMES:
            raise ValueError(f"Invalid scheme: {scheme}. The scheme should be one of {VALID_SCHEMES}.") 
        if scheme == "python":
            repr = self.get_workflow_python_repr()
        elif scheme == "yaml":
            repr = self.get_workflow_yaml_repr()
        elif scheme == "code":
            repr = self.get_workflow_code_repr()
        elif scheme == "core":
            repr = self.get_workflow_core_repr()
        elif scheme == "bpmn":
            repr = self.get_workflow_bpmn_repr()
        return repr

    def parse_from_scheme(self, scheme: str, repr: str) -> SequentialWorkFlowGraph:
        """
        Parse the SequentialWorkFlowGraph from the given scheme and representation.
        """
        if scheme not in VALID_SCHEMES:
            raise ValueError(f"Invalid scheme: {scheme}. The scheme should be one of {VALID_SCHEMES}.")
        if scheme == "python":
            graph = self.parse_workflow_python_repr(repr)
        elif scheme == "yaml":
            graph = self.parse_workflow_yaml_repr(repr)
        elif scheme == "code":
            graph = self.parse_workflow_code_repr(repr)
        elif scheme == "core":
            graph = self.parse_workflow_core_repr(repr)
        elif scheme == "bpmn":
            graph = self.parse_workflow_bpmn_repr(repr)
        return graph

    def _get_workflow_repr_info(self) -> List[dict]:
        """
        Get the information for the workflow representation.
        """
        info = []
        for node in self.graph.nodes:
            task_name = node.name
            input_names = [param.name for param in node.inputs] 
            output_names = [param.name for param in node.outputs]
            task_info = {
                "task_name": task_name,
                "input_names": input_names,
                "output_names": output_names
            }
            info.append(task_info)
        return info
    
    def _convert_to_func_name(self, name: str) -> str:
        """
        Convert the task name to the function name.
        """
        name = name.lower().strip()
        name = name.replace(' ', '_').replace('-', '_')
        name = ''.join(c for c in name if c.isalnum() or c == '_')
        # Replace multiple consecutive underscores with a single underscore
        name = regex.sub(r'_+', "_", name)
        # Remove leading/trailing underscores
        name = name.strip('_')
        return name
    
    def _convert_to_title(self, name: str) -> str:
        func_name = self._convert_to_func_name(name)
        words = func_name.split('_')
        return ' '.join(word.capitalize() for word in words)
    
    def get_workflow_python_repr(self) -> str: 
        repr_info = self._get_workflow_repr_info()
        if not repr_info:
            return ""
        
        python_workflow_info = [] 
        for task_info in repr_info:
            name = self._convert_to_func_name(task_info['task_name'])
            input_names = [f'{input_name}' for input_name in task_info['input_names']]
            output_names = [f'{output_name}' for output_name in task_info['output_names']]
            python_workflow_info.append(
                "{{'name': '{name}', 'args': {args}, 'outputs': {outputs}}}".format(
                    name=name,
                    args=input_names,
                    outputs=output_names
                )
            )
        python_workflow_repr = "steps = [\n" + ",\n".join(python_workflow_info) + "\n]"
        return python_workflow_repr
    
    def get_workflow_yaml_repr(self) -> str:
        repr_info = self._get_workflow_repr_info() 
        if not repr_info:
            return ""
        
        yaml_workflow_info = []
        for task_info in repr_info:
            name = self._convert_to_func_name(task_info['task_name'])
            input_names = "\n".join([f'    - {input_name}' for input_name in task_info['input_names']])
            output_names = "\n".join([f'    - {output_name}' for output_name in task_info['output_names']])
            yaml_workflow_info.append(
                "- name: {name}\n  args:\n{input_names}\n  outputs:\n{output_names}".format(
                    name=name,
                    input_names=input_names,
                    output_names=output_names
                )
            )
        yaml_workflow_repr = "\n\n".join(yaml_workflow_info)
        return yaml_workflow_repr

    def get_workflow_code_repr(self) -> str:
        repr_info = self._get_workflow_repr_info()
        if not repr_info:
            return ""
            
        workflow_lines = []
        for task_info in repr_info:
            # Convert task name to snake_case
            name = self._convert_to_func_name(task_info['task_name'])
            
            # Format inputs and outputs
            inputs = ", ".join(task_info['input_names'])
            outputs = ", ".join(task_info['output_names'])
            
            # Create the line in format: task_name(inputs) -> outputs
            line = f"{name}({inputs}) -> {outputs}"
            workflow_lines.append(line)
            
        # Join all lines with newlines
        workflow_repr = "\n".join(workflow_lines)
        
        return workflow_repr

    def get_workflow_bpmn_repr(self) -> str:

        repr_info = self._get_workflow_repr_info()
        if not repr_info:
            return ""
            
        # Start the BPMN XML
        bpmn_lines = [
            '<definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL">',
            '<process id="software_dev_workflow" isExecutable="true">',
            '    <startEvent id="start" />'
        ]
        
        # Add tasks
        for i, task_info in enumerate(repr_info):
            task_name = self._convert_to_func_name(task_info['task_name'])
            task_title = self._convert_to_title(task_info['task_name'])
            bpmn_lines.append(f'    <task id="{task_name}" name="{task_title}" />')
            
        bpmn_lines.append('    <endEvent id="end" />')
        bpmn_lines.append('')
        bpmn_lines.append('    <!-- Workflow connections -->')
        
        # Add sequence flows
        # First flow from start to first task
        if repr_info:
            first_task_id = self._convert_to_func_name(repr_info[0]['task_name'])
            bpmn_lines.append(f'    <sequenceFlow id="flow1" sourceRef="start" targetRef="{first_task_id}" />')
            
        # Flows between tasks
        for i in range(len(repr_info) - 1):
            source_id = self._convert_to_func_name(repr_info[i]['task_name'])
            target_id = self._convert_to_func_name(repr_info[i + 1]['task_name'])
            flow_num = i + 2
            bpmn_lines.append(f'    <sequenceFlow id="flow{flow_num}" sourceRef="{source_id}" targetRef="{target_id}" />')
            
        # Last flow from last task to end
        if repr_info:
            last_task_id = self._convert_to_func_name(repr_info[-1]['task_name'])
            flow_num = len(repr_info) + 1
            bpmn_lines.append(f'    <sequenceFlow id="flow{flow_num}" sourceRef="{last_task_id}" targetRef="end" />')
            
        # Close tags
        bpmn_lines.append('</process>')
        bpmn_lines.append('</definitions>')
        
        return '\n'.join(bpmn_lines)
    
    def get_workflow_core_repr(self) -> str:

        repr_info = self._get_workflow_repr_info()
        if not repr_info:
            return ""
            
        workflow_lines = []
        for i, task_info in enumerate(repr_info, 1):
            # Convert task name to title case
            task_name = self._convert_to_title(task_info['task_name'])
            # Create the line with the specified format
            next_step = i + 1
            line = f"Step {i}::: Process ::: {task_name}:::next::Step {next_step}"
            workflow_lines.append(line)
            
        # Add the terminal step
        last_step = len(repr_info) + 1
        workflow_lines.append(f"Step {last_step}::: Terminal ::: End of Workflow:::")
        
        return "\n".join(workflow_lines)

    def _find_task_index(self, step: dict, graph_repr_info: List[dict]) -> int:
        """
        Find the index of the task in the original workflow graph. If the task is not found, return -1. 

        Args:
            step (dict): The step of the workflow.
            graph_repr_info (List[dict]): The information of the original workflow graph.
        
        Returns:
            int: The index of the task.
        """
        def _is_task_name_match(task_name: str, another_name: str) -> bool:
            return self._convert_to_func_name(task_name) == self._convert_to_func_name(another_name)

        def _is_task_inputs_match(task_inputs: List[str], another_inputs: List[str]) -> bool:
            return len(set(task_inputs) & set(another_inputs)) == len(task_inputs)
        
        def _is_task_outputs_match(task_outputs: List[str], another_outputs: List[str]) -> bool:
            return len(set(task_outputs) & set(another_outputs)) == len(task_outputs)
        
        for i, task in enumerate(graph_repr_info):
            if _is_task_name_match(task["task_name"], step["name"]) and _is_task_inputs_match(task["input_names"], step["args"]) and _is_task_outputs_match(task["output_names"], step["outputs"]):
                return i
        return -1

    def create_workflow_graph_from_steps(
        self, 
        steps: List[dict]
    ) -> SequentialWorkFlowGraph:
        
        """
        Create a new workflow graph from the steps.
        Since both the inputs and outputs are provided, new tasks will be created in the new workflow graph. 
        It is used for the `python` `yaml` and `code` representations. 

        Args:
            steps (List[dict]): The steps of the workflow. The steps are in the format of:
                [
                    {
                        "name": str,
                        "args": List[str],
                        "outputs": List[str]
                    }
                ]
        
        Returns:
            SequentialWorkFlowGraph: The new workflow graph.
        """
        original_workflow_config = self.graph.get_graph_info()
        repr_info = self._get_workflow_repr_info()
        new_tasks = [] 
        for step in steps: 
            task_index = self._find_task_index(step=step, graph_repr_info=repr_info)
            if task_index == -1:
                # create a new task
                task_name = step["name"]
                description = f"Task to {task_name.lower()}. " 
                if step["args"]: 
                    description += f"Takes {', '.join(step['args'])} as input. " 
                if step["outputs"]: 
                    description += f"Produces {', '.join(step['outputs'])} as output."
                
                new_task = {
                    "name": task_name, 
                    "description": description,
                    "inputs": [
                        {
                            "name": input_name, 
                            "type": "str", 
                            "description": f"Input parameter {input_name} for {task_name}"
                        } for input_name in step["args"]
                    ], 
                    "outputs": [
                        {
                            "name": output_name, 
                            "type": "str", 
                            "description": f"Output parameter {output_name} from {task_name}"
                        } for output_name in step["outputs"]
                    ], 
                    "prompt": "to be updated",
                    "llm_config": original_workflow_config["tasks"][0]["llm_config"], 
                    "parse_mode": "str"     
                }
                new_tasks.append(new_task)
            else:
                # copy the task from the original workflow graph
                new_tasks.append(deepcopy(original_workflow_config["tasks"][task_index]))
        # create new workflow configuration 
        new_workflow_config = {
            "goal": original_workflow_config["goal"],
            "tasks": new_tasks
        }

        # create new workflow graph
        new_graph = SequentialWorkFlowGraph.from_dict(new_workflow_config)
        return new_graph

    def create_workflow_graph_from_task_names(
        self,
        task_names: Optional[List[str]] = None,
        task_titles: Optional[List[str]] = None
    ) -> SequentialWorkFlowGraph:
        """
        Create a new workflow graph from the task names or titles. 
        Since only the task names or titles are provided, the tasks in the new workflow graph will be copied from the original workflow graph. 
        It is used for the `bpmn` and `core` representations. 

        Args:
            task_names (Optional[List[str]]): The names of the tasks.
            task_titles (Optional[List[str]]): The titles of the tasks.
        
        Returns:
            SequentialWorkFlowGraph: The new workflow graph.
        """
        if task_names:
            original_workflow_config = self.graph.get_graph_info()
            tasks = task_names
            original_tasks = {self._convert_to_func_name(task["name"]): task for task in original_workflow_config["tasks"]} 
        elif task_titles:
            original_workflow_config = self.graph.get_graph_info()
            tasks = task_titles 
            original_tasks = {self._convert_to_title(task["name"]): task for task in original_workflow_config["tasks"]}
        else:
            raise ValueError("No task names or titles provided.")

        new_tasks = []
        for task in tasks:
            if task not in original_tasks:
                raise ValueError(f"Task {task} not found in the original workflow.")
            new_tasks.append(deepcopy(original_tasks[task]))
        
        # create new workflow configuration 
        new_workflow_config = {
            "goal": original_workflow_config["goal"],
            "tasks": new_tasks
        }

        # create new workflow graph
        new_graph = SequentialWorkFlowGraph.from_dict(new_workflow_config)
        return new_graph

    def parse_workflow_python_repr(self, repr: str) -> SequentialWorkFlowGraph:
        """
        Parse the workflow from the python representation. The input format is:
        steps = [
            {"name": task_name, "args": [input1, input2, ...],"outputs": [output1, output2, ...]}, 
            {"name": another_task_name, "args": [input1, input2, ...],"outputs": [output1, output2, ...]}, 
            ...
        ]
        """
        try:
            # extract ```python ``` 
            code_block = regex.search(r'```python\s*(.*?)\s*```', repr, regex.DOTALL)
            if not code_block:
                raise ValueError("No Python code block found in the representation")
            code_block = code_block.group(1).strip()
            # relevant_lines = [] 
            # for line in code_block.splitlines():
            #     line = line.strip()
            #     if not line or line.startswith("#") or line.startswith("```"):
            #         continue
            #     if all(key in line for key in ["name", "args", "outputs"]):
            #         relevant_lines.append(line)
            # steps_str = "[\n" + "\n".join(relevant_lines) + "\n]"
            # steps = eval(steps_str)
            steps = eval(code_block.replace("steps = ", "").strip())
            new_graph = self.create_workflow_graph_from_steps(steps=steps)
            return new_graph
        except Exception as e:
            logger.warning(f"Failed to parse workflow string: {e}. Return the original workflow.")
        
        return self.graph
    
    def parse_workflow_yaml_repr(self, repr: str) -> SequentialWorkFlowGraph:
        """
        Parse the workflow from the yaml representation. The input format is:
        - name: task_name
          args:
            - input1
            - input2
          outputs:
            - output1
        """
        try:
            # extract ```yaml ``` 
            match = regex.search(r'```yaml\s*(.*?)\s*```', repr, regex.DOTALL) 
            if not match:
                raise ValueError("No YAML code block found in the representation")
            yaml_block = match.group(1).strip()
            steps = yaml.safe_load(yaml_block)
            # relevant_lines = []  
            # in_step = False  
            # for line in yaml_block.splitlines(): 
            #     stripped_line = line.strip() 
            #     if stripped_line.startswith('- name:'):
            #         in_step = True 
            #         relevant_lines.append(line) 
            #     elif in_step and (
            #         stripped_line.startswith('args:') or 
            #         stripped_line.startswith('outputs:') or 
            #         stripped_line.startswith('- ')
            #     ):
            #         relevant_lines.append(line)
            #     elif not stripped_line: 
            #         in_step = False  
            # yaml_step = "\n".join(relevant_lines)
            # steps = yaml.safe_load(yaml_step)
            new_graph = self.create_workflow_graph_from_steps(steps=steps)
            return new_graph
        except Exception as e:
            logger.warning(f"Failed to parse workflow string: {e}. Return the original workflow.")

        return self.graph
    
    def parse_workflow_code_repr(self, repr: str) -> SequentialWorkFlowGraph:
        """
        Parse the workflow from the code representation. 
        The input format is:
        task_name(input1, input2, ...) -> output1, output2, ...
        another_task_name(input1, input2, ...) -> output1, output2, ...
        ...
        """
        try:
            # extract ```code ``` 
            match = regex.search(r'```code\s*(.*?)\s*```', repr, regex.DOTALL)
            if not match:
                raise ValueError("No code block found in the representation")
            code_block = match.group(1).strip()
            lines = [line.strip() for line in code_block.split("\n") if line.strip() and "->" in line]
            steps = []
            for line in lines:
                # Remove any leading numbers and dots (e.g., "1. ")
                line = regex.sub(r'^\d+\.\s*', '', line)
                func_part, output_part = line.split('->')
                func_part = func_part.strip()
                name = func_part[:func_part.index('(')]
                args_str = func_part[func_part.index('(') + 1:func_part.rindex(')')]
                args = [arg.strip() for arg in args_str.split(',') if arg.strip()]
                outputs = [out.strip() for out in output_part.split(',') if out.strip()]
                step = {"name": name, "args": args, "outputs": outputs}
                steps.append(step)
            if not steps:
                raise ValueError("No steps found in the workflow.")
            new_graph = self.create_workflow_graph_from_steps(steps=steps)
            return new_graph
        except Exception as e:
            logger.warning(f"Failed to parse workflow string: {e}. Return the original workflow.")

        return self.graph
    
    def parse_workflow_bpmn_repr(self, repr: str) -> SequentialWorkFlowGraph:
        """
        Parse the workflow from the BPMN XML representation.
        
        The input format is BPMN XML with:
        - task elements defining the tasks
        - sequenceFlow elements defining the order of tasks
        
        Will extract ordered task names from the sequence flows and create a workflow.
        """
        try:
            # extract ```bpmn ``` 
            match = regex.search(r'```bpmn\s*(.*?)\s*```', repr, regex.DOTALL) 
            if not match:
                raise ValueError("No BPMN code block found in the representation")
            bpmn_block = match.group(1).strip()
            # Parse XML string
            root = ET.fromstring(bpmn_block)
            
            # Define namespace for BPMN XML
            ns = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}
            
            # Get process element
            process = root.find('bpmn:process', ns) or root.find('process')
            
            if process is None:
                raise ValueError("No process element found in BPMN XML")
                
            # Create a dictionary of all tasks
            tasks = {}
            # for task in process.findall('.//task', ns) or process.findall('.//task'):
            for task in process.findall("bpmn:task", ns): 
                tasks[task.get('id')] = task.get('name')
            
            # Get sequence flows and order them
            flows = {}
            ordered_tasks = []
            current_ref = 'start'
            
            # Create dictionary of source -> target
            # for flow in process.findall('.//sequenceFlow', ns) or process.findall('.//sequenceFlow'):
            for flow in process.findall("bpmn:sequenceFlow", ns): 
                flows[flow.get('sourceRef')] = flow.get('targetRef')
            
            # Follow the sequence flows to get ordered tasks
            while current_ref in flows:
                next_ref = flows[current_ref]
                if next_ref in tasks:  # Only add if it's a task (not end event)
                    ordered_tasks.append(tasks[next_ref])
                current_ref = next_ref
            
            # Create new workflow graph using the ordered task names
            new_graph = self.create_workflow_graph_from_task_names(task_titles=ordered_tasks)
            return new_graph
            
        except Exception as e:
            logger.warning(f"Failed to parse BPMN workflow string: {e}. Return the original workflow.")
        
        return self.graph
        
    def parse_workflow_core_repr(self, repr: str) -> SequentialWorkFlowGraph:
        """
        Parse the workflow from the Core representation.
        
        The input format is:
        Step 1::: Process ::: Task Name:::next::Step 2
        Step 2::: Process ::: Another Task:::next::Step 3
        ...
        Step N::: Terminal ::: End of Workflow:::
        
        Will extract task names from Process steps and create a workflow.
        """
        try:
            # extract ```core ```
            match = regex.search(r'```core\s*(.*?)\s*```', repr, regex.DOTALL) 
            if not match:
                raise ValueError("No core code block found in the representation")
            core_block = match.group(1).strip()
            # Split into lines and remove empty lines
            lines = [line.strip() for line in core_block.split('\n') if line.strip()]
            
            # Initialize flows and tasks dictionaries
            flows = {}  # step -> next_step
            tasks = {}  # step -> task_title
            
            # First pass: build flows and tasks mappings
            for line in lines:
                parts = line.split(':::')
                current_step = parts[0].strip()
                step_type = parts[1].strip()
                
                if step_type == 'Process':
                    # Extract task title and next step 
                    task_title = parts[2].strip()
                    tasks[current_step] = task_title 
                    if len(parts) > 3 and "next" in parts[3]: 
                        next_step = parts[3].split("::")[-1].strip()
                        flows[current_step] = next_step
                elif step_type == 'Terminal':
                    flows[current_step] = None
            
            # Second pass: follow flows to build ordered task list
            ordered_tasks = []
            current_step = 'Step 1'
            
            while current_step in flows:
                if current_step in tasks:  # Only add if it's a Process step
                    ordered_tasks.append(tasks[current_step])
                current_step = flows[current_step]
            # Create new workflow graph using the ordered task titles
            new_graph = self.create_workflow_graph_from_task_names(task_titles=ordered_tasks)
            return new_graph
            
        except Exception as e:
            logger.warning(f"Failed to parse Core workflow string: {e}. Return the original workflow.")
        
        return self.graph


class SimplePromptBreeder:

    def __init__(self, llm: BaseLLM, **kwargs):
        self.llm = llm
        self.kwargs = kwargs

    def generate_mutation_prompt(self, task_description: str, **kwargs) -> str:
        """
        Generate the mutation prompt for optimization.
        """
        thinking_style = random.choice(thinking_styles)
        hyper_mutation_prompt = thinking_style + "\n\nProblem Description: " + task_description + ".\n" + "Output: "
        # print(">>>>>>>>>> Hyper mutation prompt: <<<<<<<<<<<\n", hyper_mutation_prompt)
        mutation_prompt = self.llm.generate(
            prompt=hyper_mutation_prompt, 
            system_message="You are a helpful assistant",
        ).content
        return mutation_prompt
    
    def get_mutation_prompt(self, task_description: str, order: Literal["zero-order", "first-order"], **kwargs) -> str:
        """
        Get the mutation prompt for optimization.
        """
        if order == "zero-order":
            mutation_prompt = self.generate_mutation_prompt(task_description=task_description)
        elif order == "first-order":
            mutation_prompt = random.choice(mutation_prompts)
        else:
            raise ValueError(f"Invalid order: {order}. The order should be either 'zero-order' or 'first-order'.")
        return mutation_prompt

    def generate_prompt(self, task_description: str, prompt: str, order: Literal["zero-order", "first-order"], **kwargs) -> str:
        """
        Generate the prompt for optimization. 
        
        Args:
            task_description (str): The description of the task, normally the goal of the workflow. 
            prompt (str): The prompt to optimize.
            order (Literal["zero-order", "first-order"]): The order of the mutation prompt.
        
        Returns:
            str: The optimized prompt.
        """
        mutation_prompt = self.get_mutation_prompt(task_description=task_description, order=order)
        prompt = mutation_prompt + "\n\nINSTRUCTION:\n\n" + prompt
        # print(">>>>>>>>>> Prompt: <<<<<<<<<<<\n", prompt)
        new_prompt = self.llm.generate(
            prompt=prompt, 
            system_message="You are a helpful assistant",
        ).content
        return new_prompt


class SEWOptimizer(Optimizer):

    graph: Union[SequentialWorkFlowGraph, ActionGraph] = Field(description="The workflow to optimize.")
    repr_scheme: str = Field(default="python", description="The scheme to represent the workflow.")
    optimize_mode: Literal["all", "structure", "prompt"] = Field(default="all", description="The mode to optimize the workflow.")
    order: Literal["zero-order", "first-order"] = Field(default="zero-order", description="Whether to use zero-order (using hyper-mutation prompt) or first-order (using mutation prompt) optimization.")

    def init_module(self, **kwargs):
        self._snapshot: List[dict] = []
        self._prompt_breeder = SimplePromptBreeder(llm=self.llm) # generate prompt for optimization
        self._convergence_check_counter = 0 
        self._best_score = float("-inf")
        if isinstance(self.graph, ActionGraph):
            if self.optimize_mode != "prompt":
                raise ValueError(
                    f"{type(self).__name__} only support prompt optimization when `graph` is an `ActionGraph`. "
                    f"The `optimize_mode` should be set to `prompt`, but got {self.optimize_mode}."
                )

    def optimize(self, dataset: Benchmark, **kwargs):

        if isinstance(self.graph, SequentialWorkFlowGraph):
            logger.info(f"Optimizing the {type(self.graph).__name__} workflow with {self.repr_scheme} representation.")
        elif isinstance(self.graph, ActionGraph):
            logger.info(f"Optimizing the {type(self.graph).__name__} graph ...")
        graph: Union[SequentialWorkFlowGraph, ActionGraph] = self.graph 
        logger.info("Run initial evaluation on the original workflow ...")
        with suppress_logger_info():
            metrics = self.evaluate(dataset, eval_mode="dev", graph=graph)
        logger.info(f"Initial metrics: {metrics}")
        self.log_snapshot(graph=graph, metrics=metrics)

        for i in range(self.max_steps):
            try:
                # perform a step of optimization
                graph = self.step()
                # evaluate the workflow
                if (i + 1) % self.eval_every_n_steps == 0:
                    logger.info(f"Evaluate the workflow at step {i+1} ...")
                    with suppress_logger_info():
                        metrics = self.evaluate(dataset, eval_mode="dev")
                    logger.info(f"Step {i+1} metrics: {metrics}")
                    self.log_snapshot(graph=graph, metrics=metrics)
            except Exception as e:
                logger.warning(f"Error in step {i}: {e}. Skip this step.")
                continue
            if self.convergence_check():
                logger.info(f"Convergence check passed at step {i+1}. Stop the optimization.")
                break
        
        if i == self.max_steps - 1:
            logger.info(f"Reach the maximum number of steps {self.max_steps}. Stop the optimization.")
        
        # set self.graph to the best graph
        logger.info("Restore the best graph from the snapshot ...")
        self.restore_best_graph()
    
    def step(self, **kwargs) -> Union[SequentialWorkFlowGraph, ActionGraph]:
        """
        Take a step of optimization and return the optimized graph.
        """
        graph = self._select_graph_with_highest_score(return_metrics=False)
        if isinstance(graph, SequentialWorkFlowGraph):
            new_graph = self._workflow_graph_step(graph)
        elif isinstance(graph, ActionGraph):
            new_graph = self._action_graph_step(graph)
        else:
            raise ValueError(f"Invalid graph type: {type(graph)}. The graph should be an instance of `WorkFlowGraph` or `ActionGraph`.")
        return new_graph
    
    def evaluate(
        self, 
        dataset: Benchmark, 
        eval_mode: str = "test", 
        graph: Optional[Union[SequentialWorkFlowGraph, ActionGraph]] = None,
        indices: Optional[List[int]] = None,
        sample_k: Optional[int] = None,
        **kwargs
    ) -> dict:
        """
        Evaluate the workflow. If `graph` is provided, use the provided graph for evaluation. Otherwise, use the graph in the optimizer. 
        
        Args:
            dataset (Benchmark): The dataset to evaluate the workflow on.
            eval_mode (str): The evaluation mode. Choices: ["test", "dev", "train"].
            graph (Union[WorkFlowGraph, ActionGraph], optional): The graph to evaluate. If not provided, use the graph in the optimizer.
            indices (List[int], optional): The indices of the data to evaluate the workflow on.
            sample_k (int, optional): The number of data to evaluate the workflow on. If provided, a random sample of size `sample_k` will be used.
        
        Returns:
            dict: The metrics of the workflow evaluation.
        """
        graph = graph if graph is not None else self.graph
        metrics_list = []
        for i in range(self.eval_rounds):
            eval_info = [
                f"[{type(graph).__name__}]", 
                f"Evaluation round {i+1}/{self.eval_rounds}", 
                f"Mode: {eval_mode}"
            ]
            if indices is not None:
                eval_info.append(f"Indices: {len(indices)} samples")
            if sample_k is not None:
                eval_info.append(f"Sample size: {sample_k}")
            logger.info(" | ".join(eval_info))
            metrics = self.evaluator.evaluate(
                graph=graph, 
                benchmark=dataset, 
                eval_mode=eval_mode, 
                indices=indices, 
                sample_k=sample_k,
                **kwargs
            )
            metrics_list.append(metrics)
        avg_metrics = self.evaluator._calculate_average_score(metrics_list)
        
        return avg_metrics
    
    def log_snapshot(self, graph: Union[SequentialWorkFlowGraph, ActionGraph], metrics: dict):
        
        if isinstance(graph, SequentialWorkFlowGraph):
            graph_info = graph.get_graph_info()
        elif isinstance(graph, ActionGraph):
            # TODO check if the action graph is valid 
            graph_info = graph
        else:
            raise ValueError(f"Invalid graph type: {type(graph)}. The graph should be an instance of `SequentialWorkFlowGraph` or `ActionGraph`.")
        
        self._snapshot.append(
            {
                "index": len(self._snapshot),
                "graph": deepcopy(graph_info),
                "metrics": metrics,
            }
        )

    def _select_graph_with_highest_score(self, return_metrics: bool = False) -> Union[SequentialWorkFlowGraph, ActionGraph]:

        if len(self._snapshot) == 0:
            return self.graph
        snapshot_scores = [np.mean(list(snapshot["metrics"].values())) for snapshot in self._snapshot]
        best_index = np.argmax(snapshot_scores)

        if isinstance(self.graph, SequentialWorkFlowGraph):
            graph = SequentialWorkFlowGraph.from_dict(self._snapshot[best_index]["graph"])
        elif isinstance(self.graph, ActionGraph):
            # TODO check if the action graph is valid
            graph = self._snapshot[best_index]["graph"]
        else:
            raise ValueError(f"Invalid graph type: {type(self.graph)}. The graph should be an instance of `SequentialWorkFlowGraph` or `ActionGraph`.")
        
        if return_metrics:
            return graph, self._snapshot[best_index]["metrics"]
        return graph
    
    def restore_best_graph(self):

        best_graph, best_metrics = self._select_graph_with_highest_score(return_metrics=True)
        logger.info(f"Restore the best graph from snapshot with metrics {best_metrics} ...")
        self.graph = best_graph

    def _wfg_structure_optimization_step(self, graph: SequentialWorkFlowGraph) -> SequentialWorkFlowGraph:
        """
        optinize the structure of the workflow graph and return the optimized graph.
        Args:
            graph (SequentialWorkFlowGraph): The workflow graph to optimize.
        
        Returns:
            SequentialWorkFlowGraph: The optimized workflow graph.  
        """
        graph_scheme = SEWWorkFlowScheme(graph=graph)
        graph_repr = graph_scheme.convert_to_scheme(scheme=self.repr_scheme)
        if self.repr_scheme == "python":
            output_format = "\n\nALWAYS wrap the refined workflow in ```python\n``` format and DON'T include any other text within the code block!"
        elif self.repr_scheme == "yaml":
            output_format = "\n\nALWAYS wrap the refined workflow in ```yaml\n``` format and DON'T include any other text within the code block!"
        elif self.repr_scheme == "code":
            output_format = "\n\nALWAYS wrap the refined workflow in ```code\n``` format and DON'T include any other text within the code block!"
        elif self.repr_scheme == "core":
            output_format = "\n\nALWAYS wrap the refined workflow in ```core\n``` format and DON'T include any other text within the code block!"
        elif self.repr_scheme == "bpmn":
            output_format = "\n\nALWAYS wrap the refined workflow in ```bpmn\n``` format and DON'T include any other text within the code block!"
        else:
            raise ValueError(f"Invalid representation scheme: {self.repr_scheme}. The scheme should be one of {VALID_SCHEMES}.")
        prompt = "Task Description: " + graph.goal + "\n\nWorkflow Steps: " + graph_repr + output_format
        new_graph_repr = self._prompt_breeder.generate_prompt(task_description=graph.goal, prompt=prompt, order=self.order)
        new_graph = graph_scheme.parse_from_scheme(scheme=self.repr_scheme, repr=new_graph_repr)
        return new_graph
    
    @staticmethod
    def _validate_refined_prompt(new_prompt: str, original_prompt: str, input_names: List[str]) -> bool:
        """
        Validate a refined prompt returned by the prompt breeder before adopting it.

        The LLM may occasionally echo the meta-instructions used to request the
        refinement (or drop the input placeholders), in which case adopting the
        output verbatim silently corrupts the workflow node.

        Args:
            new_prompt (str): The refined prompt returned by the prompt breeder.
            original_prompt (str): The current prompt of the task/operator.
            input_names (List[str]): Names of the task inputs. Placeholders (e.g. ``{question}``)
                that are present in ``original_prompt`` must also be present in ``new_prompt``.

        Returns:
            bool: True if the refined prompt is safe to adopt, False otherwise.
        """
        if not new_prompt or not new_prompt.strip():
            return False
        lowered = new_prompt.lower()
        meta_markers = ("please refine the instruction", "only output the refined instruction")
        if any(marker in lowered for marker in meta_markers):
            return False
        for name in input_names:
            placeholder = "{" + name + "}"
            if placeholder in original_prompt and placeholder not in new_prompt:
                return False
        return True

    def _wfg_prompt_optimization_step(self, graph: SequentialWorkFlowGraph) -> SequentialWorkFlowGraph:

        task_description = graph.goal
        graph_scheme = SEWWorkFlowScheme(graph=graph)
        graph_repr = graph_scheme.convert_to_scheme(scheme=self.repr_scheme)
        graph_info = graph.get_graph_info()
        for i, task in enumerate(graph_info["tasks"]):
            original_prompt = task["prompt"]
            optimization_prompt = "Task Description: " + task_description + "\n\nWorkflow Steps:\n" + graph_repr + f"\n\nINSTRUCTION for the {i+1}-th task:\n\"\"\"\n" + original_prompt + "\n\"\"\""
            optimization_prompt += f"\n\nGiven the above information, please refine the instruction for the {i+1}-th task.\n"
            optimization_prompt += r"Note that you should always use bracket (e.g. `{input_name}`) to wrap the inputs of the tasks in your refined instruction.\n"
            optimization_prompt += "Only output the refined instruction and DON'T include any other text!"
            new_prompt = self._prompt_breeder.generate_prompt(task_description=task_description, prompt=optimization_prompt, order=self.order)
            input_names = [inp.get("name") for inp in task.get("inputs", []) if inp.get("name")]
            if self._validate_refined_prompt(new_prompt, original_prompt, input_names):
                graph_info["tasks"][i]["prompt"] = new_prompt
            else:
                logger.warning(
                    f"Discarding invalid refined prompt for task '{task.get('name', i)}': "
                    "the response echoes the refinement meta-instructions or drops required input placeholders. "
                    "Keeping the original prompt."
                )
        new_graph = SequentialWorkFlowGraph.from_dict(graph_info)
        return new_graph
        
    def _workflow_graph_step(self, graph: SequentialWorkFlowGraph) -> SequentialWorkFlowGraph:

        if self.optimize_mode == "structure" or self.optimize_mode == "all":
            # optimize the structure of the graph    
            graph = self._wfg_structure_optimization_step(graph)
        if self.optimize_mode == "prompt" or self.optimize_mode == "all":
            # optimize the prompt of the graph
            graph = self._wfg_prompt_optimization_step(graph)
        
        return graph
    
    def _action_graph_prompt_optimization_step(self, graph: ActionGraph) -> ActionGraph:

        task_description = graph.description
        graph_info = graph.get_graph_info()
        graph_steps = inspect.getsource(getattr(graph, "execute"))
        for operator_name, operator_info in graph_info["operators"].items():
            original_prompt = operator_info["prompt"]
            optimization_prompt = "Task Description: " + task_description + "\n\nWorkflow Steps:\n" + graph_steps + f"\n\nINSTRUCTION for the `{operator_name}` operator:\n\"\"\"\n" + original_prompt + "\n\"\"\""
            optimization_prompt += "\n\nThe interface of the operator is as follows:\n" + operator_info["interface"]
            optimization_prompt += f"\n\nGiven the above information, please refine the instruction for the `{operator_name}` operator.\n"
            optimization_prompt += r"Note that you should always use bracket (e.g. `{input_name}`) to wrap the inputs of the operator in your refined instruction, "
            optimization_prompt += "and the input names should be EXACTLY the same as those defined in the interface. DON'T use bracket to wrap output names."
            optimization_prompt += "\nOnly output the refined instruction and DON'T include any other text!"
            new_prompt = self._prompt_breeder.generate_prompt(task_description=task_description, prompt=optimization_prompt, order=self.order)
            new_prompt = new_prompt.replace("\"", "").strip()
            if self._validate_refined_prompt(new_prompt, original_prompt, []):
                graph_info["operators"][operator_name]["prompt"] = new_prompt
            else:
                logger.warning(
                    f"Discarding invalid refined prompt for operator '{operator_name}': "
                    "the response echoes the refinement meta-instructions. Keeping the original prompt."
                )
        new_graph = ActionGraph.from_dict(graph_info)
        return new_graph

    def _action_graph_step(self, graph: ActionGraph) -> ActionGraph:
        
        if self.optimize_mode == "prompt":
            graph = self._action_graph_prompt_optimization_step(graph)
        else:
            raise ValueError(f"{type(self).__name__} only support prompt optimization when `self.graph` is an `ActionGraph` instance. "
                    f"The `optimize_mode` should be set to `prompt`, but got {self.optimize_mode}.")
        return graph

    def convergence_check(self, **kwargs) -> bool:
        
        if not self._snapshot:
            logger.warning("No snapshots available for convergence check")
            return False
        
        # Get scores from snapshots
        scores = [np.mean(list(snapshot["metrics"].values())) for snapshot in self._snapshot]
        current_score = scores[-1]

        if current_score > self._best_score:
            self._best_score = current_score
            self._convergence_check_counter = 0
        else:
            self._convergence_check_counter += 1

        if self._convergence_check_counter >= self.convergence_threshold:
            logger.info(f"Early stopping triggered: No improvement for {self.convergence_threshold} iterations")
            # logger.info(f"Score history: {scores[-self.convergence_threshold:]}")
            return True
        return False

    def save(self, path: str, ignore: List[str] = []):
        """
        Save the (optimized) workflow graph to a file. 

        Args:
            path (str): The path to save the workflow graph.
            ignore (List[str]): The keys to ignore when saving the workflow graph.
        """
        self.graph.save_module(path, ignore=ignore)
    