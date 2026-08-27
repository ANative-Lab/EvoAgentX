# -----------------------------------------------------------------------------
# This file re-implements algorithms from the TextGrad project:
#   Repo: https://github.com/zou-group/textgrad
#   Paper: "TextGrad: Automatic 'Differentiation' via Text -- using large language models to backpropagate textual gradients."
#   Authors: Zou et al.
#
# Re-implementation integrated into EvoAgentX with permission from the authors.
# All mistakes or modifications are our own.
#
# Code of Conduct: This project follows the Microsoft Open Source Code of Conduct.
#   https://opensource.microsoft.com/codeofconduct/
# -----------------------------------------------------------------------------

# ruff: noqa: E402
import os
import shutil
from copy import deepcopy
from typing import Any, Iterator, List, Literal, Optional, Tuple, Union

import numpy as np
from pydantic import Field, PositiveInt
from tqdm import tqdm

from ...agents import Agent, CustomizeAgent
from ...benchmark.benchmark import Benchmark
from ...benchmark.scorers import CodeScorer
from ...core.callbacks import suppress_logger_info
from ...core.logging import logger
from ...core.module import BaseModule
from ...evaluators import Evaluator
from ...models.base_model import BaseLLM
from ...prompts import PromptTemplate
from ...workflow.workflow_graph import WorkFlowGraph, WorkFlowNode

# Check if logs folder exists before importing textgrad
log_folder_exists = os.path.exists("./logs")

import textgrad as tg
from textgrad import Variable, EngineLM
from textgrad import logger as tg_logger
from textgrad import sh as tg_file_handler
from textgrad.autograd import StringBasedFunction
from textgrad.loss import MultiFieldEvaluation, TextLoss
from textgrad.optimizer import TextualGradientDescent

from ...prompts.optimizers.textgrad_optimizer import (
    CODE_LOSS_PROMPT,
    GENERAL_LOSS_PROMPT,
    NO_ANSWER_LOSS_PROMPT,
    OPTIMIZER_CONSTRAINTS,
    OPTIMIZER_SYSTEM_PROMPT,
    PERSONAL_FINANCE_ADVISOR_EXAMPLE,
    FITNESS_COACH_EXAMPLE,
    CODE_REVIEW_EXAMPLE,
)

tg_logger.removeHandler(tg_file_handler)
# remove the logs folder created by textgrad
if not log_folder_exists and os.path.exists("./logs"):
    shutil.rmtree("./logs")


class TextGradEngine(EngineLM):
    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def generate(self, prompt: str, system_prompt: str = None, **kwargs):
        with suppress_logger_info():
            response = self.llm.generate(prompt, system_prompt=system_prompt, **kwargs)
            return response.content

    def __call__(self, prompt: str, **kwargs):
        return self.generate(prompt, **kwargs)


class CustomAgentCall:
    """A custom agent call with textgrad.Variable inputs and output."""
    def __init__(self, agent: Agent):
        self.agent = agent
        self.last_outputs: dict[str, str] = dict()
    
    def __call__(
        self, 
        instruction: Variable, 
        system_prompt: Variable,
        **inputs: Variable
    ) -> Variable:

        action = self.agent.actions[0]
        input_names = action.inputs_format.get_attrs()

        agent_inputs = {}

        for key, input_variable in inputs.items():
            if key in input_names:
                agent_inputs[key] = input_variable.value
            else:
                parsed_inputs: dict[str, str] = {k:v for k,v in input_variable.parsed_outputs.items() if k in input_names}
                agent_inputs.update(parsed_inputs)

        with suppress_logger_info():
            outputs = self.agent.execute(action_name=action.name, action_input_data=agent_inputs).content

        self.last_outputs = outputs.to_dict()

        return outputs.content


class TextGradAgent:
    """An agent that takes textgrad.Variable inputs and returns a textgrad.Variable response.
    This class is used to replace EvoAgentX Agent in WorkFlowGraph to allow TextGrad optimization.
    """
    def __init__(
        self, 
        agent: Agent,
        optimize_mode: Literal["all", "system_prompt", "instruction"] = "all"
    ):
        self.name = agent.name

        require_grad = {
            "all": {"system_prompt": True, "instruction": True}, 
            "system_prompt": {"system_prompt": True, "instruction": False}, 
            "instruction": {"system_prompt": False, "instruction": True}
        }

        system_prompt_require_grad = require_grad[optimize_mode]["system_prompt"]
        instruction_require_grad = require_grad[optimize_mode]["instruction"]

        self.system_prompt = Variable(
            agent.system_prompt, 
            requires_grad=system_prompt_require_grad, 
            role_description=f"{self.name}'s system prompt"
        )
        
        self.instruction = Variable(
            agent.actions[0].prompt_template.instruction,
            requires_grad=instruction_require_grad, 
            role_description=f"{self.name}'s instruction prompt"
        )

        self._agent_call = CustomAgentCall(agent)
        self.forward = StringBasedFunction(self._agent_call, agent.description)
        self.output_description = " and ".join(agent.actions[0].outputs_format.get_attr_descriptions().values())
        self.last_output = None

    def __call__(self, inputs: dict[str, Variable]) -> Variable:
        """Given textgrad.Variable inputs, generates a textgrad.Variable output."""

        forward_inputs: dict[str, Variable] = {
            "instruction": self.instruction, 
            "system_prompt": self.system_prompt,
            **inputs
        }

        output_variable = self.forward(forward_inputs, self.output_description)
        output_variable.parsed_outputs = self._agent_call.last_outputs
        self.last_output = output_variable
        return output_variable



class TextGradOptimizer(BaseModule):
    """Uses TextGrad to optimize agents' system prompts and instructions in a multi-agent workflow.
    For more information on TextGrad, see https://github.com/zou-group/textgrad.
    """
    graph: WorkFlowGraph = Field(description="The workflow to optimize.")
    optimize_mode: Literal["all", "system_prompt", "instruction"] = Field(default="all", description="The mode to optimize the workflow. 'all' optimizes both system prompts and instructions, 'system_prompt' only optimizes system prompts, and 'instruction' only optimizes instructions.")
    executor_llm: BaseLLM = Field(default=None, description="The LLM to use for execution.")
    optimizer_llm: BaseLLM = Field(default=None, description="The LLM to use for optimization.")
    batch_size: PositiveInt = Field(default=1, description="The batch size for optimization.")
    max_steps: PositiveInt = Field(default=10, description="The maximum number of optimization steps.")
    evaluator: Evaluator = Field(default=None, description="The evaluator to perform evaluation during optimization.")
    eval_every_n_steps: Optional[PositiveInt] = Field(default=None, description="Evaluate the workflow every `eval_every_n_steps` steps.")
    eval_rounds: PositiveInt = Field(default=1, description="The number of times to evaluate the performance.")
    eval_config: dict = Field(default={}, description="The configuration for evaluation. The keys are the arguments of `TextGradOptimizer.evaluate`.")
    save_interval: Optional[PositiveInt] = Field(default=None, description="Save the workflow every `save_interval` steps.")
    save_path: str = Field(default="./", description="The path to save the optimized workflow.")
    rollback: bool = Field(default=True, description="Whether to rollback to the best graph after each evaluation during optimization.")
    constraints: List[str] = Field(default=[], description="The constraints for optimization. e.g. ['They system prompt must not exceed 100 words.']")


    def init_module(self, **kwargs):
        self._validate_graph_compatibility(self.graph)
        self._snapshot: List[dict] = []
        self.output_lookup = self._create_output_lookup()

    def _is_coding_benchmark(self, dataset: Benchmark) -> bool:
        return isinstance(getattr(dataset, "scorer", None), CodeScorer)

    def _init_textgrad(self, dataset: Benchmark, use_answers: bool = True):
        # Disable TextGrad's short variable value to allow the optimizer to receive the full variable value
        def disable_short_variable_value(self, n_words_offset: int = 10):
            return self.value
        Variable.get_short_value = disable_short_variable_value
            
        # Textgrad engine
        self.optimizer_engine = TextGradEngine(self.optimizer_llm)

        # Textgrad loss
        if use_answers:
            if self._is_coding_benchmark(dataset):
                loss_prompt = CODE_LOSS_PROMPT
                role_descriptions = ["code snippet to evaluate", "the task, the test result of the code snippet, and the correct code"]
            else:
                loss_prompt = GENERAL_LOSS_PROMPT
                role_descriptions = ["response to evaluate", "correct answer"]

            evaluation_instruction = Variable(loss_prompt, requires_grad=False, role_description="evaluation instruction")
            self.loss_fn = MultiFieldEvaluation(evaluation_instruction, role_descriptions, self.optimizer_engine)
        else:
            loss_prompt = NO_ANSWER_LOSS_PROMPT
            evaluation_instruction = Variable(loss_prompt, requires_grad=False, role_description="evaluation instruction")
            self.loss_fn = TextLoss(evaluation_instruction, self.optimizer_engine)

        # Create textgrad agents
        self._create_textgrad_agents()
        
        # Textgrad optimizer
        if self.optimize_mode == "all":
            optimize_variables = self._get_all_system_prompts() + self._get_all_instructions()
        elif self.optimize_mode == "system_prompt":
            optimize_variables = self._get_all_system_prompts()
        elif self.optimize_mode == "instruction":
            optimize_variables = self._get_all_instructions()
        else:
            raise ValueError("Unsupported `optimize_mode`, should be one of 'all', 'system_prompt', 'instruction'.")

        OPTIMIZER_CONSTRAINTS.extend(self.constraints)
        
        self.textgrad_optimizer = TextualGradientDescent(
            parameters=optimize_variables, 
            engine=self.optimizer_engine,
            constraints=OPTIMIZER_CONSTRAINTS,
            optimizer_system_prompt=OPTIMIZER_SYSTEM_PROMPT,
            in_context_examples=[PERSONAL_FINANCE_ADVISOR_EXAMPLE, FITNESS_COACH_EXAMPLE, CODE_REVIEW_EXAMPLE]
        )


    def optimize(self, dataset: Benchmark, use_answers: bool = True, seed: Optional[int] = None) -> None:
        """Optimizes self.graph using `dataset`.
        
        Args:
            dataset (Benchmark): The dataset to use for optimization.
            use_answers (bool): Whether to use the answers (labels) in the training set for optimization.
                If False, `dataset`'s training set does not need to have answers.
                If `eval_every_n_steps` is set to None, we can optimize the workflow without any labeled data.
            seed (Optional[int]): The random seed to use for shuffling the data.
        """
        self._init_textgrad(dataset, use_answers)
 
        def iterator() -> Iterator[Tuple[List[dict[str, str]],  Optional[List[Union[str, dict[str, str]]]]]]:
            epoch = 0
            while True:
                # Shuffle train data every epoch
                effective_seed = seed + epoch if seed is not None else None
                train_data = dataset.get_train_data(sample_k=len(dataset._train_data), seed=effective_seed)
                for i in range(0, len(train_data), self.batch_size):
                    batch = train_data[i:i + self.batch_size]
                    inputs = [self.evaluator.collate_func(x) for x in batch]
                    if use_answers:
                        labels = dataset.get_labels(batch)
                    else:
                        labels = None
                    yield inputs, labels
                epoch += 1

        data_iterator = iterator()

        for step in tqdm(range(self.max_steps)):
            inputs, labels = next(data_iterator)
            self.step(inputs, labels, dataset, use_answers)

            if self.eval_every_n_steps is not None and (step + 1) % self.eval_every_n_steps == 0:
                logger.info(f"Evaluating the workflow at step {step+1} ...")
                with suppress_logger_info():
                    metrics = self.evaluate(dataset, **self.eval_config)
                self.log_snapshot(self.graph, metrics)
                logger.info(f"Step {step+1} metrics: {metrics}")

                # If rollback is enabled, keep track of the best snapshot
                if self.rollback:
                    if len(self._snapshot) == 1:
                        best_snapshot = self._snapshot[-1]
                        best_average_score = np.mean(list(metrics.values()))
                    else:
                        current_average_score = np.mean(list(metrics.values()))
                        
                        if current_average_score >= best_average_score:
                            # If the current average score is better than the best average score, update the best snapshot
                            best_snapshot = self._snapshot[-1]
                            best_average_score = current_average_score
                        else:
                            # If the current average score is worse than the best average score, roll back to the best snapshot
                            logger.info(f"Metrics are worse than the best snapshot which has {best_snapshot['metrics']}. Rolling back to the best snapshot.")
                            best_graph = WorkFlowGraph.from_dict(best_snapshot["graph"])
                            self.graph = best_graph
                            self._create_textgrad_agents()

            if self.save_interval is not None and (step + 1) % self.save_interval == 0:
                logger.info(f"Saving the workflow at step {step+1} ...")
                self.save(os.path.join(self.save_path, f"{dataset.name}_textgrad_step_{step+1}.json"))

        logger.info(f"Reached the maximum number of steps {self.max_steps}. Optimization has finished.")
        self.save(os.path.join(self.save_path, f"{dataset.name}_textgrad_final.json"))

        # Saves the best graph
        if len(self._snapshot) > 0:
            best_graph = self._select_graph_with_highest_score()
            self.save(os.path.join(self.save_path, f"{dataset.name}_textgrad_best.json"), graph=best_graph)

        
    def step(
        self, 
        inputs: list[dict[str, str]], 
        labels: Optional[list[Union[str, dict[str, str]]]], 
        dataset: Benchmark, 
        use_answers: bool = True
    ) -> None:
        """Performs one optimization step using a batch of data."""

        losses = []
        logger.info("Executing workflow...")

        if use_answers:
            if labels is None:
                raise ValueError("Labels must be provided if `use_answers` is True.")

            for input, label in zip(inputs, labels, strict=True):
                output = self.forward(input)
                if isinstance(label, str):
                    label = Variable(label, requires_grad=False, role_description="correct answer for the query")
                elif isinstance(label, dict):
                    if not self._is_coding_benchmark(dataset):
                        raise ValueError("Label must be a string for non-coding benchmarks.")
                    end_node_name = self.graph.find_end_nodes()[0]
                    end_node = self.graph.get_node(end_node_name)
                    output_name = end_node.outputs[0].name
                    code = output.parsed_outputs[output_name]
                    label = self._format_code_label(code, label, dataset)
                    label = Variable(label, requires_grad=False, role_description="the task, the test result, and the correct code")
                loss = self.loss_fn([output, label])
                losses.append(loss)
        else:
            for input in inputs:
                output = self.forward(input)
                loss = self.loss_fn(output)
                losses.append(loss)

        total_loss = tg.sum(losses)
        logger.info("Computing gradients...")
        total_loss.backward(self.optimizer_engine)
        logger.info("Updating agents...")
        self.textgrad_optimizer.step()
        self.textgrad_optimizer.zero_grad()
        self._update_workflow_graph()
        logger.info("Agents updated")


    def forward(self, inputs: dict[str, str]) -> Variable:
        """Returns the final output from the workflow."""
        self._visited_nodes = set()
        end_node = self.graph.find_end_nodes()[0]
        input_variables = self._initial_inputs_to_variables(inputs)
        output = self._compute_node(end_node, input_variables)
        return output


    def evaluate(
        self, 
        dataset: Benchmark, 
        eval_mode: str = "dev", 
        graph: Optional[WorkFlowGraph] = None,
        indices: Optional[List[int]] = None,
        sample_k: Optional[int] = None,
        **kwargs
    ) -> dict:
        """Evaluate the workflow. If `graph` is provided, use the provided graph for evaluation. Otherwise, use the graph in the optimizer. 
        
        Args:
            dataset (Benchmark): The dataset to evaluate the workflow on.
            eval_mode (str): The evaluation mode. Choices: ["test", "dev", "train"].
            graph (WorkFlowGraph, optional): The graph to evaluate. If not provided, use the graph in the optimizer.
            indices (List[int], optional): The indices of the data to evaluate the workflow on.
            sample_k (int, optional): The number of data to evaluate the workflow on. If provided, a random sample of size `sample_k` will be used.
        
        Returns:
            dict: The metrics of the workflow evaluation.
        """
        if graph is None:
            graph = self.graph

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
                update_agents=True, 
                **kwargs
            )
            metrics_list.append(metrics)
        avg_metrics = self.evaluator._calculate_average_score(metrics_list)
        
        return avg_metrics


    def save(self, path: str, graph: Optional[WorkFlowGraph] = None, ignore: List[str] = []) -> None:
        """Save the workflow graph containing the optimized prompts to a file. 

        Args:
            path (str): The path to save the workflow graph.
            graph (WorkFlowGraph, optional): The graph to save. If not provided, use the graph in the optimizer.
            ignore (List[str]): The keys to ignore when saving the workflow graph.
        """
        if graph is None:
            graph = self.graph
        graph.save_module(path, ignore=ignore)


    def log_snapshot(self, graph: WorkFlowGraph, metrics: dict) -> None:
        """Log the snapshot of the workflow."""
        self._snapshot.append(
            {
                "index": len(self._snapshot),
                "graph": deepcopy(graph.get_config()),
                "metrics": metrics,
            }
        )


    def restore_best_graph(self) -> None:
        """Restore the best graph from the snapshot and set it to `self.graph`."""
        if len(self._snapshot) == 0:
            logger.info("No snapshot found. No graph to restore.")
            return

        best_graph, best_metrics = self._select_graph_with_highest_score(return_metrics=True)
        self.graph = best_graph
        logger.info(f"Restored the best graph from snapshot with metrics {best_metrics}")


    def _format_code_label(self, code: str, label:dict[str, str], dataset: Benchmark) -> str:
        """Formats the label for coding tasks to include the task, the test result, and the correct code.

        Args:
            code: The code to evaluate.
            label: A dictionary with keys "task_id", "test", "entry_point", and "canonical_solution".
            dataset: A benchmark instance with a `CodeScorer`.
        
        Returns:
            The formatted label which includes the task, the test result, and the correct code.
        """

        task_id = label["task_id"]
        # example ids are strings; MBPP's raw task_id is an int
        prompt = dataset.get_example_by_id(str(task_id))["prompt"]
        test = label["test"]
        entry_point = label["entry_point"]

        scorer: CodeScorer = dataset.scorer
        state, message = scorer.check_solution(
            task_id=task_id,
            solution=prompt + "\n" + code,
            test=test,
            entry_point=entry_point
        )

        if state != scorer.SUCCESS:
            message = message.replace("Solution", "Failed Code")

        formatted_label = f"## Task:\n{prompt}\n\n## Result on test:\n{message}\n\n## Correct Solution:\n{label['canonical_solution']}"
        return formatted_label


    def _initial_inputs_to_variables(self, initial_inputs: dict[str, str]) -> dict[str, Variable]:
        """Converts inputs to the initial nodes to textgrad variables."""
        variables = {}
        initial_nodes = self.graph.find_initial_nodes()
        for initial_node in initial_nodes:
            for key, value in initial_inputs.items():
                for input in self.graph.get_node(initial_node).inputs:
                    if input.name == key:
                        initial_input_variable = Variable(
                            value,
                            requires_grad=False,
                            role_description=input.description,
                        )
                        variables[key] = initial_input_variable
                        if len(variables) == len(initial_inputs):
                            return variables
        missing_inputs = set(initial_inputs.keys()) - set(variables.keys())
        raise ValueError(f"Initial inputs do not match the inputs of the initial nodes. Missing inputs: {missing_inputs}")



    def _compute_node(self, node: Union[str, WorkFlowNode], initial_inputs: dict[str, Variable]) -> Variable:
        """Computes the output of a node in the workflow graph by recursively computing the required inputs.

        Args:
            node: The node to compute the output of.
            initial_inputs: The initial inputs to the workflow that are not from any node in the workflow (e.g., user query).

        Returns:
            The output of the node as a textgrad.Variable.
        """
        if isinstance(node, str):
            node = self.graph.get_node(node)

        if node.name in self._visited_nodes:
            return node.textgrad_agent.last_output

        input_variables: dict[str, Variable] = {}    # inputs to TextGradAgent
        input_node_names: set[str] = set()           # which nodes we need to compute the output of      

        for input in node.inputs:
            if input.name in initial_inputs:
                input_variables[input.name] = initial_inputs[input.name]
            else:
                input_node_names.add(self.output_lookup[input.name])
        
        # if the input is from another node, compute the output of that node
        for node_name in input_node_names:
            input_variables[node_name] = self._compute_node(node_name, initial_inputs)

        output_variable = node.textgrad_agent(input_variables)
        self._visited_nodes.add(node.name)
        return output_variable
        
   
    def _create_textgrad_agent(self, node: Union[str, WorkFlowNode]) -> TextGradAgent:
        """Creates a textgrad agent for a given node in a WorkFlowGraph."""
        if isinstance(node, str):
            node = self.graph.get_node(node)

        if isinstance(node.agents[0], dict):
            agent_llm = node.agents[0].get("llm")
            agent_llm_config = node.agents[0].get("llm_config")
            if agent_llm is None and agent_llm_config is None:
                node.agents[0]["llm"] = self.executor_llm
            # CustomizeAgent.from_dict creates a CustomizeAgent if dict follows CustomizeAgent format
            # creates an Agent if dict follows Agent format
            agent: Union[CustomizeAgent, Agent] = CustomizeAgent.from_dict(node.agents[0])
        else:
            raise ValueError(f"Unsupported agent type {type(node.agents[0])}. Expected 'dict'.")
            
        textgrad_agent = TextGradAgent(agent, self.optimize_mode)
        return textgrad_agent


    def _create_textgrad_agents(self):
        """Creates textgrad agents for all nodes in the workflow graph."""
        for node in self.graph.nodes:
            node.textgrad_agent = self._create_textgrad_agent(node)


    def _update_agent_prompts(self, agent_dict: dict[str, Any], system_prompt: str, instruction: str) -> dict[str, Any]:
        agent_dict["system_prompt"] = system_prompt
        if "actions" in agent_dict:
            # Agent has actions in its dict
            agent_dict["actions"][0]["prompt_template"] = self._update_agent_instructions(
                agent_dict["actions"][0]["prompt_template"], 
                instruction
            )
        else:
            # CustomizeAgent does not have actions in its dict
            agent_dict["prompt_template"] = self._update_agent_instructions(
                agent_dict["prompt_template"], 
                instruction
            )
        return agent_dict


    def _update_agent_instructions(
        self, 
        prompt_template: Union[PromptTemplate, dict[str, str]], 
        instruction: str
    ) -> Union[PromptTemplate, dict[str, str]]:

        if isinstance(prompt_template, PromptTemplate):
            prompt_template.set_instruction(instruction)
        elif isinstance(prompt_template, dict):
            prompt_template["instruction"] = instruction
        else:
            raise ValueError(f"Unsupported prompt template type {type(prompt_template)}. Expected 'PromptTemplate' or 'dict'.")
        return prompt_template


    def _update_workflow_graph(self):
        """Updates the workflow graph with the latest prompts from the textgrad optimization."""
        for node in self.graph.nodes:
            if isinstance(node.agents[0], dict):
                node.agents[0] = self._update_agent_prompts(
                    node.agents[0], 
                    node.textgrad_agent.system_prompt.value, 
                    node.textgrad_agent.instruction.value
                )
            else:
                raise ValueError(f"Unsupported agent type {type(node.agents[0])}. Expected 'dict'.")


    def _select_graph_with_highest_score(self, return_metrics: bool = False) -> Union[WorkFlowGraph, tuple[WorkFlowGraph, Optional[dict]]]:
        """Select the graph in `self._snapshot` with the highest score."""
        if len(self._snapshot) == 0:
            if return_metrics:
                return self.graph, None
            return self.graph
            
        snapshot_scores = [np.mean(list(snapshot["metrics"].values())) for snapshot in self._snapshot]
        best_index = np.argmax(snapshot_scores)

        graph = WorkFlowGraph.from_dict(self._snapshot[best_index]["graph"])

        if return_metrics:
            return graph, self._snapshot[best_index]["metrics"]
        return graph


    def _get_all_system_prompts(self) -> List[Variable]:
        """Gets all system prompts from the textgrad agents."""
        system_prompts = []
        for node in self.graph.nodes:
            system_prompts.append(node.textgrad_agent.system_prompt)
        return system_prompts

    
    def _get_all_instructions(self) -> List[Variable]:
        """Gets all prompt templates from the textgrad agents."""
        instructions = []
        for node in self.graph.nodes:
            instructions.append(node.textgrad_agent.instruction)
        return instructions


    def _create_output_lookup(self) -> dict[str, str]:
        """Creates a lookup table for output names to node names."""
        output_name_to_node_name = {}
        for node in self.graph.nodes:
            for output in node.outputs:
                output_name_to_node_name[output.name] = node.name
        return output_name_to_node_name


    def _validate_graph_compatibility(self, graph: WorkFlowGraph) -> None:
        """Checks if the graph is compatible with the textgrad optimizer."""
        for node in graph.nodes:
            if len(node.agents) > 1:
                raise ValueError("TextGrad optimizer only supports workflows where every node only has a single agent.")
            else:
                agent = node.agents[0]
                if not isinstance(agent, dict):
                    raise ValueError(f"Unsupported agent type {type(agent)}. Expected 'dict'.")
                else:
                    if "actions" in agent:
                        # Agent has actions in its dict
                        # All agents have a `ContextExtraction` action, filter it out
                        non_ContextExtraction_actions = [
                            action for action in agent["actions"] if action["class_name"] != "ContextExtraction"
                        ]
                        if len(non_ContextExtraction_actions) > 1:
                            raise ValueError(f"TextGrad optimizer only supports workflows where every agent only has a single action. {agent['name']} has {len(non_ContextExtraction_actions)} actions.")
                        if "prompt_template" not in non_ContextExtraction_actions[0]:
                            raise ValueError(f"Please provide a PromptTemplate for {agent['name']}.")
                    else:
                        # CustomizeAgent does not have actions in its dict
                        if "prompt_template" not in agent:
                            raise ValueError(f"Please provide a PromptTemplate for {agent['name']}.")
                    

                    