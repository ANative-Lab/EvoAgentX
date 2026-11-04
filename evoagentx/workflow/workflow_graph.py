import inspect
import json
import threading
from collections import defaultdict
from copy import deepcopy
from enum import Enum
from functools import wraps
from typing import Dict, List, Literal, Optional, Tuple, Union

import networkx as nx
from networkx import MultiDiGraph
from pydantic import Field, field_validator, model_validator

from ..agents import Agent, CustomizeAgent
from ..core.base_config import Parameter
from ..core.logging import logger
from ..core.module import BaseModule
from ..core.module_utils import recursive_to_dict
from ..prompts.utils import DEFAULT_SYSTEM_PROMPT
from ..prompts.workflow.sew_workflow import SEW_WORKFLOW
from ..utils.async_utils import is_method_overridden
from ..utils.utils import (
    generate_dynamic_class_name,
    make_parent_folder,
    pydantic_to_parameters,
    recursive_remove,
    remove_none,
    validate_param,
)
from .action_graph import ActionGraph


class WorkFlowNodeState(str, Enum):
    """
    Enumeration of possible states for workflow nodes.
    
    This enum defines the lifecycle states of a workflow node:
    - PENDING: The node is waiting to be executed
    - RUNNING: The node is currently being executed
    - COMPLETED: The node has been successfully executed
    - FAILED: The node execution has failed
    """
    PENDING="pending"
    RUNNING="running"
    COMPLETED = "completed"
    FAILED = "failed"


class WorkFlowNode(BaseModule):
    """
    Represents a node in a workflow graph.
    
    A workflow node represents a specific task in the workflow with its
    inputs, outputs, and execution metadata. It can have associated agents
    that execute the task and track its execution status.
    
    Attributes:
        name: A unique identifier for the task within a workflow
        description: Detailed description of what the task does
        inputs: List of input parameters required by the task
        outputs: List of output parameters produced by the task
        reason: Optional justification for this task's existence
        agents: Optional list of agents that can execute this task
        action_graph: Optional graph of actions to execute this task
        status: Current execution state of the task
    """

    name: str # A short name of the task. Should be unique in a single workflow
    description: str # A detailed description of the task
    inputs: List[Parameter] = Field(default_factory=list)  # may be empty for nodes with no external input
    outputs: List[Parameter] = Field(default_factory=list)
    reason: Optional[str] = None
    agents: Optional[List] = None
    action_graph: Optional[ActionGraph] = None
    status: Optional[WorkFlowNodeState] = WorkFlowNodeState.PENDING

    @staticmethod
    def _warn_agent_dict_class_name(agent: dict, node_name: Optional[str] = None) -> None:
        if not isinstance(agent, dict) or "class_name" not in agent:
            return

        agent_name = agent.get("name", "<unknown>")
        node_context = f" in node '{node_name}'" if node_name else ""
        logger.warning(
            "Agent dict '{}'{} contains class_name='{}'. BaseModule will convert "
            "this dict into an Agent instance during construction/loading. If you "
            "intended to keep it as a plain agent config dict for AgentManager, "
            "remove or rename the top-level 'class_name' key.",
            agent_name,
            node_context,
            agent.get("class_name"),
        )

    @classmethod
    def _warn_agent_dicts_with_class_name(cls, agents, node_name: Optional[str] = None) -> None:
        if not agents:
            return
        for agent in agents:
            cls._warn_agent_dict_class_name(agent, node_name=node_name)

    def __init__(self, **kwargs):
        self._warn_agent_dicts_with_class_name(kwargs.get("agents"), node_name=kwargs.get("name"))
        super().__init__(**kwargs)

    @field_validator('agents', mode="before")
    @classmethod
    def check_agent_format(cls, agents: List[Union[str, Dict, Agent]]):
        if agents is None:
            return None

        validated_agents = []
        for agent in agents:
            if isinstance(agent, str):
                validated_agents.append(agent)
            elif isinstance(agent, Agent):
                validated_agents.append(agent)
            elif isinstance(agent, dict):
                cls._warn_agent_dict_class_name(agent)
                assert "name" in agent and "description" in agent, \
                    "must provide the name and description of an agent when specifying an agent with a dict."
                validated_agents.append(agent)
            else:
                raise TypeError(f"'{type(agent)}' is an unknown agent type!")
        return validated_agents

    @model_validator(mode="after")
    def check_action_graph(self):
        """
        Validates that:
        1. All required parameters of execute/async_execute methods are included in inputs
        2. The execute/async_execute methods return dictionaries
        3. All output parameters are present in the returned dictionaries
        """
        if self.action_graph is None:
            return self
        
        # Get input parameter names from the node's input parameters
        input_param_names = {param.name for param in self.inputs if param.required}
        output_param_names = {param.name for param in self.outputs if param.required}
        
        implemented_methods = []

        def check_method_signature(method_name):
            """Helper function to check method signature against input parameters"""

            if not is_method_overridden(self.action_graph, ActionGraph, method_name):
                return
            implemented_methods.append(method_name)

            # Get method signature
            method = getattr(self.action_graph, method_name)
            method_sig = inspect.signature(method)
            
            # Only consider parameters other than self, *args, and **kwargs as required
            required_params = []
            for name, param in method_sig.parameters.items():
                if name != 'self' and param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                    if param.default == param.empty:  # This is a required parameter
                        required_params.append(name)
            
            # Check if all required parameters are in inputs
            missing_inputs = set(required_params) - input_param_names
            if missing_inputs:
                raise ValueError(f"`{method_name}` method requires parameters that are not in `inputs`: {missing_inputs}")
        
        # Check execute method
        check_method_signature("execute")
        # Check async_execute method if it exists
        check_method_signature("async_execute")

        if not implemented_methods:
            raise ValueError(
                f"ActionGraph '{type(self.action_graph).__name__}' must implement `execute` or `async_execute`."
            )

        # Monkey-patch execute and async_execute to check returns at runtime
        def check_method_return(method_name, result):
            if not isinstance(result, dict):
                raise TypeError(f"{method_name} must return a dictionary, got {type(result)}")
            
            # Check if all output keys are in the result
            missing_outputs = output_param_names - set(result.keys())
            if missing_outputs:
                raise ValueError(f"{method_name} return value is missing required outputs: {missing_outputs}")
            
            return result

        # Replace the methods with our patched versions
        if "execute" in implemented_methods:
            original_execute = self.action_graph.execute

            @wraps(original_execute)
            def patched_execute(*args, **kwargs):
                result = original_execute(*args, **kwargs)
                return check_method_return("execute", result)

            self.action_graph.execute = patched_execute

        if "async_execute" in implemented_methods:
            original_async_execute = self.action_graph.async_execute

            @wraps(original_async_execute)
            async def patched_async_execute(*args, **kwargs):
                result = original_async_execute(*args, **kwargs)
                if inspect.isawaitable(result):
                    result = await result
                return check_method_return("async_execute", result)

            self.action_graph.async_execute = patched_async_execute
        
        return self

    def to_dict(self, exclude_none: bool = True, ignore: List[str] = [], **kwargs) -> dict:
        
        agents_dict: List[Union[str, dict]] = []
        if self.agents:
            for agent in self.agents:
                if isinstance(agent, str):
                    agents_dict.append(agent)
                elif isinstance(agent, Agent):
                    agents_dict.append(agent.get_config())
                elif isinstance(agent, dict):
                    agent_dict = recursive_to_dict(agent)
                    # for CustomizeAgent: a callable parse_func is not serializable, store its name
                    if "parse_func" in agent_dict and callable(agent_dict["parse_func"]):
                        agent_dict["parse_func"] = agent_dict["parse_func"].__name__
                    agents_dict.append(agent_dict)
                else:
                    raise TypeError(f"'{type(agent)}' is an unknown agent type!")

        if exclude_none:
            agents_dict = remove_none(agents_dict)

        if ignore:
            agents_dict = recursive_remove(agents_dict, ignore)

        if "agents" not in ignore:
            ignore = [*ignore, "agents"]

        data = super().to_dict(exclude_none=exclude_none, ignore=ignore, **kwargs)
        data["agents"] = agents_dict
        return data

    def get_agents(self) -> List[str]:
        """
        Return the names of all agents associated with this node.
        """
        agent_names = []
        if not self.agents:
            return []
        
        for agent in self.agents:
            if isinstance(agent, str):
                agent_names.append(agent)
            elif isinstance(agent, dict):
                agent_names.append(agent["name"])
            elif isinstance(agent, Agent):
                agent_names.append(agent.name)
            else:
                raise TypeError(f"{type(agent)} is an unknown agent type!")
        return agent_names
    
    def set_agents(self, agents: List[Union[str, Dict, Agent]]):
        self.agents = agents

    def get_status(self) -> WorkFlowNodeState:
        return self.status
    
    def set_status(self, state: WorkFlowNodeState):
        self.status = state
    
    @property
    def is_complete(self) -> bool:
        return self.status == WorkFlowNodeState.COMPLETED
    
    def get_task_info(self) -> str:

        def format_parameters(params: List[Parameter]) -> str:
            if not params:
                return "None"
            return "\n".join(f"  - {param.name} ({param.type}): {param.description}" for param in params)
        
        desc = (
            f"Name: {self.name}\n"
            f"Description: {self.description}\n"
            f"Inputs:\n{format_parameters(self.inputs)}\n"
            f"Outputs:\n{format_parameters(self.outputs)}\n"
        )
        return desc

    def get_input_names(self, required: bool = False) -> List[str]:
        if required:
            return [param.name for param in self.inputs if param.required]
        else:
            return [param.name for param in self.inputs]
    
    def get_output_names(self, required: bool = False) -> List[str]:
        if required:
            return [param.name for param in self.outputs if param.required]
        else:
            return [param.name for param in self.outputs]

    def check_agents(self):
        """
        Checks if any agent assigned to this node accept the node inputs and if any agent outputs the node outputs.
        """
        if not self.agents:
            # A node may be executed either by agents or by an action_graph. When an
            # action_graph is provided, the node is satisfied without any agents, and
            # its inputs/outputs are validated against the action_graph elsewhere.
            if self.action_graph is not None:
                return
            raise ValueError(f"No agents assigned to node '{self.name}'")

        node_inputs_dict = {}
        inputs_in_agents = {}
        for node_input in self.inputs:
            node_inputs_dict[node_input.name] = node_input
            inputs_in_agents[node_input.name] = False

        node_outputs_dict = {}
        outputs_in_agents = {}
        for node_output in self.outputs:
            node_outputs_dict[node_output.name] = node_output
            outputs_in_agents[node_output.name] = False

        in_agents = {"inputs": inputs_in_agents, "outputs": outputs_in_agents}
        node_inputs_outputs = {"inputs": node_inputs_dict, "outputs": node_outputs_dict}


        def _check_agent_dict(agent_dict: dict, inputs_or_outputs: Literal["inputs", "outputs"]) -> dict:

            # "input" or "output"
            input_or_output = inputs_or_outputs[:-1]

            for agent_input_or_output in agent_dict[inputs_or_outputs]:
                if agent_input_or_output["name"] in node_inputs_outputs[inputs_or_outputs]:
                    in_agents[inputs_or_outputs][agent_input_or_output["name"]] = True
                    agent_input_or_output_param = Parameter(**agent_input_or_output)
                    validate_param(
                        node_inputs_outputs[inputs_or_outputs][agent_input_or_output["name"]],
                        agent_input_or_output_param,
                        f"node '{self.name}' {input_or_output}",
                        f"agent '{agent_dict['name']}' {input_or_output}",
                    )

            return agent_dict


        def _check_agent(agent: Agent, inputs_or_outputs: Literal["inputs", "outputs"]) -> Agent:

            # "input" or "output"
            input_or_output = inputs_or_outputs[:-1]

            for i, agent_actions in enumerate(agent.actions):
                if agent_actions.name == "ContextExtraction":
                    continue

                action_format_name = "inputs_format" if inputs_or_outputs == "inputs" else "outputs_format"
                action_format = getattr(agent_actions, action_format_name)
                
                ignore = ["class_name"]

                if not action_format._is_content_defined_in_subclass():
                    ignore.append("content")
                
                action_params = pydantic_to_parameters(action_format, ignore=ignore)

                for param in action_params:
                    if param.name in node_inputs_outputs[inputs_or_outputs]:
                        in_agents[inputs_or_outputs][param.name] = True
                        validate_param(
                            node_inputs_outputs[inputs_or_outputs][param.name],
                            param,
                            f"node '{self.name}' {input_or_output}",
                            f"agent action '{agent_actions.name}' {input_or_output}",
                        )

                action_params = [param.to_dict(ignore=["class_name"]) for param in action_params]
                if inputs_or_outputs == "inputs":                      
                    required_action_format = CustomizeAgent.create_action_input(action_params, agent_actions.name)
                else:
                    required_action_format = CustomizeAgent.create_action_output(action_params, agent_actions.name)
                setattr(agent.actions[i], action_format_name, required_action_format)
            return agent

        
        def _check_customize_agent(agent: CustomizeAgent, inputs_or_outputs: Literal["inputs", "outputs"]) -> CustomizeAgent:
            # input or output
            input_or_output = inputs_or_outputs[:-1]

            for agent_input_or_output in getattr(agent, inputs_or_outputs):
                param_name = agent_input_or_output.name
                if param_name in node_inputs_outputs[inputs_or_outputs]:
                    in_agents[inputs_or_outputs][param_name] = True
                    validate_param(
                        node_inputs_outputs[inputs_or_outputs][param_name],
                        agent_input_or_output,
                        f"node '{self.name}' {input_or_output}",
                        f"agent '{agent.name}' {input_or_output}",
                    )

            return agent


        # String agents are references to agents registered elsewhere (e.g. in an
        # AgentManager) and are not resolved here, so their inputs/outputs cannot be
        # inspected for structural validation.
        has_unresolved_agents = any(isinstance(agent, str) for agent in self.agents)

        for i, agent in enumerate(self.agents):
            if isinstance(agent, str):
                continue
            elif isinstance(agent, Agent):
                if isinstance(agent, CustomizeAgent):
                    self.agents[i] = _check_customize_agent(agent, "inputs")
                    self.agents[i] = _check_customize_agent(agent, "outputs")
                else:
                    self.agents[i] = _check_agent(agent, "inputs")
                    self.agents[i] = _check_agent(agent, "outputs")
            elif isinstance(agent, dict):
                self.agents[i] = _check_agent_dict(agent, "inputs")
                self.agents[i] = _check_agent_dict(agent, "outputs")
            else:
                raise TypeError(f"{type(agent)} is an unknown agent type!")

        # When the node has unresolved (string) agent references, their inputs/outputs
        # are unknown here, so coverage of the node's inputs/outputs cannot be verified.
        if has_unresolved_agents:
            return

        if not all(in_agents["inputs"].values()):
            missing_inputs = [input_name for input_name, in_agent in in_agents["inputs"].items() if not in_agent]
            raise ValueError(f"Not all inputs of node '{self.name}' are used by agents: {missing_inputs}")
        if not all(in_agents["outputs"].values()):
            missing_outputs = [output_name for output_name, in_agent in in_agents["outputs"].items() if not in_agent]
            raise ValueError(f"Not all outputs of node '{self.name}' can be found in agents: {missing_outputs}")

    def update_inputs(self, inputs: List[Parameter]):
        self.inputs = inputs
        self.update_agent_inputs()

    def update_outputs(self, outputs: List[Parameter]):
        self.outputs = outputs
        self.update_agent_outputs()

    def update_agent_inputs(self):
        """Update all agent inputs to match the node's inputs"""
        self._update_agent_params("inputs")

    def update_agent_outputs(self):
        """Update all agent outputs to match the node's outputs"""
        self._update_agent_params("outputs")

    @staticmethod
    def _get_agent_param_names(agent, param_type: Literal["inputs", "outputs"]) -> set:
        if isinstance(agent, Agent) and hasattr(agent, param_type):
            return {p.name for p in getattr(agent, param_type)}
        elif isinstance(agent, dict):
            return {p["name"] for p in agent.get(param_type, [])}
        return set()

    def _update_agent_params(self, param_type: Literal["inputs", "outputs"]):
        if not self.agents:
            logger.warning(f"Node '{self.name}' has no agents; {param_type} not updated.")
            return

        node_params_map = {
            param.name: param
            for param in getattr(self, param_type)
        }
        agents = self.agents
        complement_type = "outputs" if param_type == "inputs" else "inputs"

        for i, agent in enumerate(agents):
            if len(agents) == 1:
                # Single-agent: agent must exactly mirror the node's params
                if isinstance(agent, Agent) and hasattr(agent, param_type):
                    setattr(self.agents[i], param_type, list(node_params_map.values()))
                elif isinstance(agent, dict):
                    self.agents[i][param_type] = [p.to_dict(ignore=["class_name"]) for p in node_params_map.values()]
                continue

            # Multi-agent: collect inter-agent param names from other agents'
            # complementary params (e.g. other agents' outputs when syncing inputs)
            inter_agent_names = set()
            for j, other in enumerate(agents):
                if j != i:
                    inter_agent_names |= self._get_agent_param_names(other, complement_type)

            if isinstance(agent, Agent) and hasattr(agent, param_type):
                new_params = []
                for param in getattr(agent, param_type):
                    name = param.name
                    if name in node_params_map:
                        new_params.append(node_params_map[name])
                    elif name in inter_agent_names:
                        new_params.append(param)
                    # else: stale node param — drop it
                setattr(self.agents[i], param_type, new_params)

            elif isinstance(agent, dict):
                new_params = []
                for param in agent.get(param_type, []):
                    name = param["name"]
                    if name in node_params_map:
                        new_params.append(node_params_map[name].to_dict(ignore=["class_name"]))
                    elif name in inter_agent_names:
                        new_params.append(param)
                self.agents[i][param_type] = new_params

        if len(agents) > 1:
            covered = set()
            for agent in agents:
                covered |= self._get_agent_param_names(agent, param_type)
            missing = set(node_params_map.keys()) - covered
            if missing:
                logger.warning(
                    f"Node '{self.name}': the following {param_type} are not handled by any agent "
                    f"and will not be propagated: {sorted(missing)}"
                )


class WorkFlowEdge(BaseModule):
    """
    Represents a directed edge in a workflow graph.
    
    Workflow edges connect tasks (nodes) in the workflow graph, establishing
    execution dependencies and data flow relationships. Each edge has a source
    node, target node, and optional priority to influence execution order.
    
    Attributes:
        source: Name of the source node (where the edge starts)
        target: Name of the target node (where the edge ends)
        priority: Numeric priority value for this edge (higher means higher priority)
    """

    source: str 
    target: str 
    priority: int = 0 

    def __init__(self, edge_tuple: Optional[tuple]=(), **kwargs):
        """
        Initialize a WorkFlowEdge instance with either a tuple or keyword arguments.

        Parameters:
        ----------
            edge_tuple (tuple): a tuple containing the edge attributes in the format: (source, target, priority[optional]). 
                - source (str): the source of the edge. 
                - target (str): the target of the edge. 
                - priority (int, optional): The priority of the edge. Defaults to 0 if not provided.
            
            kwargs (dict): Key-value pairs specifying the edge attributes. These values will override those provided in `args` if both are supplied.

        Notes:
        ----------
            - Attributes provided via `kwargs` take precedence over those from the `args` tuple.
            - If `args` is empty or not provided, only `kwargs` will be used to initialize the instance.
        """
        data = self.init_from_tuple(edge_tuple)
        data.update(kwargs)
        super().__init__(**data)
    
    def init_from_tuple(self, edge_tuple: tuple) -> dict:
        if not edge_tuple:
            return {}
        keys = ["source", "target", "priority"]
        data = {k: v for k, v in zip(keys, edge_tuple)}
        return data
    
    def compare_attrs(self):
        return (self.source, self.target, self.priority)
    
    def __eq__(self, other: "WorkFlowEdge"):
        if not isinstance(other, WorkFlowEdge):
            return NotImplemented
        self_compare_attrs = self.compare_attrs()
        other_compare_attrs = other.compare_attrs()
        return all(self_attr==other_attr for self_attr, other_attr in zip(self_compare_attrs, other_compare_attrs))

    def __hash__(self):
        return hash(self.compare_attrs())


class WorkFlowGraph(BaseModule):
    """
    Represents a complete workflow as a directed graph.
    
    WorkFlowGraph models a workflow as a directed graph where nodes represent tasks
    and edges represent dependencies and data flow between tasks. It provides
    methods for constructing, validating, traversing, and executing workflows.
    
    The graph structure supports advanced features like detecting and handling loops,
    determining execution order, and tracking execution state.
    
    Attributes:
        goal: The high-level objective of this workflow
        nodes: List of WorkFlowNode instances representing tasks
        edges: List of WorkFlowEdge instances representing dependencies
        graph: Internal NetworkX MultiDiGraph or another WorkFlowGraph
        workflow_inputs: List of inputs that the workflow accepts. If not provided, inputs from initial nodes are used.
        workflow_outputs: The final outputs of the workflow. If not provided, outputs from end nodes are used.
    """

    goal: str
    nodes: Optional[List[WorkFlowNode]] = []
    edges: Optional[List[WorkFlowEdge]] = []
    graph: Optional[Union[MultiDiGraph, "WorkFlowGraph"]] = Field(default=None, exclude=True)
    workflow_inputs: Optional[List[Parameter]] = None
    workflow_outputs: Optional[List[Parameter]] = None

    def init_module(self):
        self._lock = threading.Lock()
        self._display_fig = None
        self._display_live = False
        self._display_refreshing = False
        if not self.graph:
            self._init_from_nodes(self.nodes, explicit_edges=self.edges)
        elif isinstance(self.graph, MultiDiGraph):
            self._init_from_multidigraph(self.graph, self.nodes)
        elif isinstance(self.graph, WorkFlowGraph):
            self._init_from_workflowgraph(self.graph, self.nodes)
        else:
            raise TypeError(f"{type(self.graph)} is an unknown type for graph. Supported types: [MultiDiGraph, WorkFlowGraph]")
        
        def _dedup_params(params: List[Parameter]) -> List[Parameter]:
            # Multiple initial/end nodes may share a parameter name (e.g. a common workflow
            # input). Keep the first occurrence so the derived list passes the uniqueness check.
            seen = set()
            deduped = []
            for param in params:
                if param.name in seen:
                    continue
                seen.add(param.name)
                deduped.append(param)
            return deduped

        # If `workflow_inputs` is not provided, set it to the inputs of initial nodes
        if self.workflow_inputs is None:
            initial_nodes = [node for node, in_degree in self.graph.in_degree() if in_degree==0]
            workflow_inputs = []
            for node_name in initial_nodes:
                workflow_inputs.extend(self.get_node(node_name).inputs)
            self.workflow_inputs = _dedup_params(workflow_inputs)

        # If `workflow_outputs` is not provided, set it to the outputs of end nodes
        if self.workflow_outputs is None:
            end_nodes = [node for node, out_degree in self.graph.out_degree() if out_degree==0]
            workflow_outputs = []
            for node_name in end_nodes:
                workflow_outputs.extend(self.get_node(node_name).outputs)
            self.workflow_outputs = _dedup_params(workflow_outputs)

        self.workflow_inputs_dict = {param.name: param for param in self.workflow_inputs}
        self.workflow_outputs_dict = {param.name: param for param in self.workflow_outputs}
        
        self._validate_workflow_structure()
        self._check_workflow_inputs_outputs()
        # NOTE: skip _check_agents during initialization because in workflow generator
        # the graph needs to be constructed before agents are generated
        self.update_graph()
    
    def update_graph(self):
        # call this function when modifying nodes or edges!
        self._loops = self._find_all_loops()

    def _infer_edges_from_nodes(self, nodes: List[WorkFlowNode]) -> List[WorkFlowEdge]:
        """
        Infer edges from the provided nodes by checking whether any output of a node
        matches an input of another node. An edge (source -> target) is created whenever
        at least one output name of `source` appears as an input name of `target`.
        """
        edges = []
        node_inputs = {node.name: set(param.name for param in node.inputs) for node in nodes}
        node_outputs = {node.name: set(param.name for param in node.outputs) for node in nodes}

        for source in nodes:
            for target in nodes:
                if source.name == target.name:
                    continue
                if node_inputs[target.name] & node_outputs[source.name]:
                    edge = WorkFlowEdge(source=source.name, target=target.name)
                    if edge not in edges:
                        edges.append(edge)
        return edges

    def _init_from_nodes(self, nodes: List[WorkFlowNode] = [], explicit_edges: Optional[List[WorkFlowEdge]] = None):
        """
        Initialize the WorkFlowGraph from a set of nodes. Data-flow edges are inferred
        automatically: an edge from node A to node B is created whenever an output of A shares
        a name with an input of B.

        Any user-provided `explicit_edges` are merged *in addition to* the inferred edges
        (deduplicated by `(source, target)`). If an explicit edge has the same `(source, target)`
        as an inferred edge, the explicit edge replaces the inferred edge's metadata (e.g.
        `priority`) while preserving the inferred data-flow topology. This lets users express
        ordering dependencies that carry no shared data, while wrong/incomplete explicit edges can
        no longer silently break the graph. Explicit edges referencing unknown nodes raise;
        explicit edges with no matching input/output are kept but emit a warning (see `add_edge`).
        """
        self.nodes = []
        self.edges = []
        self.graph = MultiDiGraph()
        self.add_nodes(*nodes, update_graph=False)
        inferred_edges = self._infer_edges_from_nodes(self.nodes)
        self.add_edges(*inferred_edges, update_graph=False)

        if explicit_edges:
            seen_pairs = {(edge.source, edge.target) for edge in inferred_edges}
            extra_edges = []
            for edge in explicit_edges:
                pair = (edge.source, edge.target)
                if pair in seen_pairs:
                    self._replace_edge_by_pair(edge)
                    continue
                seen_pairs.add(pair)
                extra_edges.append(edge)
            self.add_edges(*extra_edges, update_graph=False)

    def _replace_edge_by_pair(self, edge: WorkFlowEdge) -> bool:
        """
        Replace an existing edge with the same source/target pair, preserving one edge
        in both `self.edges` and the underlying NetworkX graph.
        """
        if not isinstance(edge, WorkFlowEdge):
            raise ValueError(f"{edge} is not a valid WorkFlowEdge instance!")

        for i, existing_edge in enumerate(self.edges):
            if existing_edge.source != edge.source or existing_edge.target != edge.target:
                continue

            self.edges[i] = edge
            edge_data = self.graph.get_edge_data(edge.source, edge.target, default={})
            for attrs in edge_data.values():
                ref = attrs.get("ref")
                if isinstance(ref, WorkFlowEdge) and ref.source == edge.source and ref.target == edge.target:
                    attrs["ref"] = edge
                    return True
            return True
        return False

    def _init_from_multidigraph(self, graph: MultiDiGraph, nodes: List[WorkFlowNode] = []):
        graph_nodes = [deepcopy(node_attrs["ref"]) for _, node_attrs in graph.nodes(data=True)]
        graph_edges = [deepcopy(edge_attrs["ref"]) for *_, edge_attrs in graph.edges(data=True)]
        graph_nodes = self.merge_nodes(graph_nodes, nodes)
        self._init_from_nodes(nodes=graph_nodes, explicit_edges=graph_edges)

    def _init_from_workflowgraph(self, graph: "WorkFlowGraph", nodes: List[WorkFlowNode] = []):
        graph_nodes = deepcopy(graph.nodes)
        graph_edges = deepcopy(graph.edges)
        graph_nodes = self.merge_nodes(graph_nodes, nodes)
        self._init_from_nodes(nodes=graph_nodes, explicit_edges=graph_edges)

    def _check_isolated_nodes(self):
        """
        If there are isolated nodes, check if their inputs and outputs are workflow inputs and outputs.
        If not, raise error as their inputs and outputs are not connected to the workflow graph.
        """
        isolated_nodes = list(nx.isolates(self.graph))
        if len(self.graph.nodes) > 1 and len(isolated_nodes) > 0:
            
            for node_name in isolated_nodes:
                node = self.get_node(node_name)

                for node_input in node.inputs:
                    if node_input.name not in self.workflow_inputs_dict:
                        error_message = f"Node '{node_name}' is an isolated node and has an input '{node_input.name}' that is not in the workflow inputs: {list(self.workflow_inputs_dict.keys())}"
                        logger.error(error_message)
                        raise ValueError(error_message)

                for node_output in node.outputs:
                    if node_output.name not in self.workflow_outputs_dict:
                        error_message = f"Node '{node_name}' is an isolated node and has an output '{node_output.name}' that is not in the workflow outputs: {list(self.workflow_outputs_dict.keys())}"
                        logger.error(error_message)
                        raise ValueError(error_message)                        

    @staticmethod
    def check_io_duplicates(inputs: List[Parameter], outputs: List[Parameter], context: str) -> None:
        """Helper to ensure input/output names are unique and do not overlap."""
        def check_group(params: List[Parameter], group_type: str):
            seen = set()
            for param in params:
                if param.name in seen:
                    error_message = f"{context} has duplicate {group_type} name: '{param.name}'"
                    logger.error(error_message)
                    raise ValueError(error_message)
                seen.add(param.name)
            return seen

        seen_inputs = check_group(inputs, "input")
        seen_outputs = check_group(outputs, "output")

        overlap = seen_inputs & seen_outputs
        if overlap:
            error_message = f"{context} inputs and outputs share the following name(s), which is not allowed: {list(overlap)}"
            logger.error(error_message)
            raise ValueError(error_message)

    def _check_workflow_io_duplicates(self):
        """Workflow inputs and outputs must be unique and not share any names."""
        WorkFlowGraph.check_io_duplicates(self.workflow_inputs, self.workflow_outputs, "Workflow")

    def _check_node_io_duplicates(self):
        """Within a single node, inputs and outputs must be unique and not share any names."""
        for node in self.nodes:
            WorkFlowGraph.check_io_duplicates(node.inputs, node.outputs, f"Node '{node.name}'")

    def _check_node_output_uniqueness(self):
        """Each output name must appear as a node output in at most one node."""
        output_to_nodes: Dict[str, List[str]] = defaultdict(list)
        
        for node in self.nodes:
            for param in node.outputs:
                output_to_nodes[param.name].append(node.name)
        
        conflicts = {
            name: nodes
            for name, nodes in output_to_nodes.items()
            if len(nodes) > 1
        }

        if conflicts:
            details = "\n".join(
                f"'{name}' produced by {nodes}" for name, nodes in conflicts.items()
            )
            error_message = f"Each node output name must be unique across all nodes. Found conflicts:\n{details}"
            logger.error(error_message)
            raise ValueError(error_message)

    def _check_node_inputs(self):
        """Validate that every node input is sourced from workflow inputs or another node's output."""

        workflow_input_names = set(self.workflow_inputs_dict.keys())
        all_node_outputs = set()

        for node in self.nodes:
            for param in node.outputs:
                all_node_outputs.add(param.name)

        for node in self.nodes:
            node_outputs = {param.name for param in node.outputs}
            valid_sources = workflow_input_names.union(all_node_outputs - node_outputs)

            for param in node.inputs:
                if param.name not in valid_sources:
                    error_message = f"Node '{node.name}' input '{param.name}' is not a workflow input or from another node's output."
                    logger.error(error_message)
                    raise ValueError(error_message)

    def _validate_workflow_structure(self):
        self._check_workflow_io_duplicates()
        self._check_node_io_duplicates()
        self._check_node_output_uniqueness()
        self._check_node_inputs()
        self._check_isolated_nodes()
        
        initial_nodes = self.find_initial_nodes()
        if len(self.graph.nodes) > 1 and not initial_nodes:
            error_message = "There are no initial nodes in the workflow!"
            logger.error(error_message)
            raise ValueError(error_message)

        end_nodes = self.find_end_nodes()
        if len(self.graph.nodes) > 1 and not end_nodes:
            logger.warning("There are no end nodes in the workflow")
    
    def find_initial_nodes(self) -> List[str]:
        # initial_nodes = [node for node, in_degree in self.graph.in_degree() if in_degree==0]

        # initial nodes are nodes that only require workflow inputs
        workflow_input_names = set(self.workflow_inputs_dict.keys())
        initial_nodes = []

        for node in self.nodes:
            required_inputs = set(node.get_input_names(required=True))

            if required_inputs.issubset(workflow_input_names):
                initial_nodes.append(node.name)

        return initial_nodes

    def find_end_nodes(self) -> List[str]:
        # end_nodes = [node for node, out_degree in self.graph.out_degree() if out_degree==0]

        # end nodes are nodes that produce workflow outputs
        workflow_output_names = set(self.workflow_outputs_dict.keys())
        end_node_names = []

        for node in self.nodes:
            node_output_names = set(node.get_output_names())

            if not node_output_names.isdisjoint(workflow_output_names):
                end_node_names.append(node.name)

        return end_node_names
    
    def _find_loops(self, start_node: Union[str, WorkFlowNode]) -> Dict[str, list]:

        if isinstance(start_node, str):
            start_node = self.get_node(node_name=start_node)
        start_node_name = start_node.name

        loops = defaultdict(list)
        def dfs(current_node_name: str, path: List[str]):
            if current_node_name in path:
                # a loop exists
                loops[current_node_name].append(path[path.index(current_node_name):])
                return
            path.append(current_node_name)
            children = self.get_node_children(current_node_name)
            if children:
                for child in children:
                    dfs(child, path)
            path.pop()
        
        dfs(start_node_name, [])
        return loops

    def _find_all_loops(self) -> Dict[str, list]:

        initial_nodes = self.find_initial_nodes()
        if not initial_nodes:
            return {} 
        
        def contain_loop(loops: List[List[str]], new_loop: List[str]):
            if not loops:
                return False
            return frozenset(new_loop) in [frozenset(loop) for loop in loops]
        
        # merge loops from different nodes 
        all_loops = defaultdict(list)
        for initial_node in initial_nodes:
            loops_from_init_node = self._find_loops(initial_node)
            for start_node, loops in loops_from_init_node.items():
                for loop in loops:
                    if not contain_loop(all_loops[start_node], loop):
                        # 合并从相同的start_node开始的环
                        all_loops[start_node].append(loop)
        
        if len(all_loops) <= 1:
            return all_loops
        
        # merge same loops with different starts (因为同一个环可能在之前的遍历中有不同的start_node)
        loop_to_start_nodes = defaultdict(dict)
        for start_node, loops in all_loops.items():
            for loop in loops:
                normalized_loop = frozenset(loop)
                loop_to_start_nodes[normalized_loop][start_node] = loop
        
        all_paths: List[List[str]] = [] 
        # 用深度遍历来判断一个环中的开始节点
        for initial_node in initial_nodes:
            all_paths.extend(self.get_all_paths_from_node(initial_node))
        
        def rank_nodes(nodes: List[str]):
            if len(nodes) == 1:
                return nodes[0]
            path_contain_nodes = None
            for path in all_paths:
                if all(node in path for node in nodes):
                    path_contain_nodes = path
                    break
            if path_contain_nodes is None:
                raise ValueError(f"Couldn't find a path that contain nodes: {nodes}")
            node_indices = [path.index(node) for node in nodes]
            return nodes[node_indices.index(min(node_indices))]
        
        all_loops = defaultdict(list)
        for start_node_loop in loop_to_start_nodes.values():
            first_node = rank_nodes(list(start_node_loop.keys()))
            all_loops[first_node].append(start_node_loop[first_node])
        
        return all_loops

    def add_node(self, node: WorkFlowNode, update_graph: bool = True, **kwargs):

        if not isinstance(node, WorkFlowNode):
            raise ValueError(f"{node} is not a valid WorkFlowNode instance!")
        if self.node_exists(node.name):
            raise ValueError(f"Duplicate node names are not allowed! Found duplicate node name: {node.name}")

        self.nodes.append(node)
        self.graph.add_node(node.name, ref=node)
        if update_graph:
            self.update_graph()

    def add_edge(self, edge: WorkFlowEdge, update_graph: bool = True, **kwargs):

        if not isinstance(edge, WorkFlowEdge):
            raise ValueError(f"{edge} is not a valid WorkFlowEdge instance!")
        for attr, node_name in zip(["source", "target"], [edge.source, edge.target]):
            if not self.node_exists(node_name):
                raise ValueError(f"{attr} node {node_name} does not exists!")
        if self.edge_exists(edge):
            raise ValueError(f"Duplicate edges are not allowed! Found duplicate edges: {edge}")
        
        # check the inputs and outputs of the edge
        source_node = self.get_node(edge.source)
        target_node = self.get_node(edge.target)
        source_output_names = set(param.name for param in source_node.outputs)
        target_input_names = set(param.name for param in target_node.inputs)
        if len(source_output_names & target_input_names) == 0:
            logger.warning(f"The edge ({edge.source}, {edge.target}) has no matching inputs and outputs! You may need to check the inputs and outputs of the nodes to ensure that at least one input of the target node is the output of the source node.")
        
        self.edges.append(edge)
        self.graph.add_edge(edge.source, edge.target, ref=edge)
        if update_graph:
            self.update_graph()

    def add_nodes(self, *nodes: WorkFlowNode, update_graph: bool = True, **kwargs):

        nodes: list = list(nodes)
        nodes.extend([kwargs.pop(var) for var in ["node", "nodes"] if var in kwargs])

        for node in nodes:
            if isinstance(node, (tuple, list)):
                for n in node:
                    self.add_node(n, update_graph=update_graph, **kwargs)
            else:
                self.add_node(node, update_graph=update_graph, **kwargs)

    def add_edges(self, *edges: WorkFlowEdge, update_graph: bool = True, **kwargs):

        edges: list = list(edges)
        edges.extend([kwargs.pop(var) for var in ["edge", "edges"] if var in kwargs])

        for edge in edges:
            if isinstance(edge, (tuple, list)):
                for e in edge:
                    self.add_edge(e, update_graph=update_graph, **kwargs)
            else:
                self.add_edge(edge, update_graph=update_graph, **kwargs)

    def node_exists(self, node: Union[str, WorkFlowNode]) -> bool:
        if isinstance(node, str):
            return node in self.graph.nodes
        elif isinstance(node, WorkFlowNode):
            return node.name in self.graph.nodes
        else:
            raise TypeError("node must be a str or WorkFlowNode instance")
    
    def _edge_exists(self, source: str, target: str, **attr_filters) -> bool:

        if not self.graph.has_edge(source, target):
            return False
        if attr_filters:
            for key, value in attr_filters.items():
                if key not in self.graph[source][target] or self.graph[source][target][key] != value:
                    return False
        return True
    
    def edge_exists(self, edge: Union[Tuple[str, str], WorkFlowEdge], **attr_filters) -> bool:

        """
        Check whether an edge exists in the workflow graph. The input `edge` can either be a tuple or a WorkFlowEdge instance.

        1. If a tuple is passed, it should be (source, target). The function will only determin whether there is an edge between the source node and the target node. 
        If attr_filters is passed, they will also be used to match the edge attributes. 
        2. If a WorkFlowEdge is passed, it will use the __eq__ method in WorkFlowEdge to determine 

        Parameters:
        ----------
            edge (Union[Tuple[str, str], WorkFlowEdge]):
                - If a tuple is provided, it should be in the format `(source, target)`. 
                The method will check whether there is an edge between the source and target nodes.
                If `attr_filters` are provided, they will be used to match edge attributes.
                - If a WorkFlowEdge instance is provided, the method will use the `__eq__` method in WorkFlowEdge 
                to determine whether the edge exists.

            attr_filters (dict, optional):
                Additional attributes to filter edges when `edge` is a tuple.

        Returns:
        -------
            bool: True if the edge exists and matches the filters (if provided); False otherwise.
        """
        if isinstance(edge, tuple):
            assert len(edge) == 2, "edge must be a tuple (source, target) or WorkFlowEdge instance"
            source, target = edge 
            return self._edge_exists(source, target, **attr_filters)
        elif isinstance(edge, WorkFlowEdge):
            return edge in self.edges 
        else:
            raise TypeError("edge must be a tuple (source, target) or WorkFlowEdge instance")
    
    def is_loop_start(self, node: Union[str, WorkFlowNode]) -> bool:
        if len(self._loops) == 0:
            return False
        node_name = node if isinstance(node, str) else node.name
        return node_name in self._loops
    
    def is_loop_end(self, node: Union[str, WorkFlowNode]) -> bool:
        if len(self._loops) == 0:
            return False
        loop_end_nodes = set()
        node_name = node if isinstance(node, str) else node.name
        for loops in self._loops.values():
            loop_end_nodes.update([loop[-1] for loop in loops])
        return node_name in loop_end_nodes
    
    def find_loops_with_start_and_end(self, start_node: Union[str, WorkFlowNode], end_node: Union[str, WorkFlowNode]) -> List[List[str]]:
        if len(self._loops) == 0:
            return []
        start_node_name = start_node if isinstance(start_node, str) else start_node.name
        end_node_name = end_node if isinstance(end_node, str) else end_node.name
        if start_node_name not in self._loops:
            return [] 
        target = []
        for loop in self._loops[start_node_name]:
            if loop[-1] == end_node_name:
                target.append(loop)
        return target

    def merge_nodes(self, nodes: List[WorkFlowNode], new_nodes: List[WorkFlowNode]):

        node_names = {node.name for node in nodes}
        for node in new_nodes:
            if node.name in node_names:
                continue
            nodes.append(node)
        return nodes
    
    def merge_edges(self, edges: List[WorkFlowEdge], new_edges: List[WorkFlowEdge]):

        for edge in new_edges:
            if edge in edges:
                continue
            edges.append(edge)
        return edges 

    def list_nodes(self) -> List[str]:
        """
        return the names of all nodes 
        """
        return [node.name for node in self.nodes]

    def get_node(self, node_name: str) -> WorkFlowNode:
        """
        return a WorkFlowNode instance based on its name.
        """
        if not self.node_exists(node=node_name):
            raise KeyError(f"{node_name} is an invalid node name. Currently available node names: {self.list_nodes()}")
        return self.graph.nodes[node_name]["ref"]
    
    def get_node_status(self, node: Union[str, WorkFlowNode]) -> WorkFlowNodeState:
        if isinstance(node, str):
            node = self.get_node(node_name=node)
        return node.get_status()
    
    @property
    def is_complete(self):
        output_node_names = self.find_end_nodes()
        target_nodes = [self.get_node(name) for name in output_node_names] if output_node_names else self.nodes
        return all(node.is_complete for node in target_nodes)
    
    def reset_graph(self):
        """
        set the status of all nodes to pending
        """
        for node in self.nodes:
            node.set_status(WorkFlowNodeState.PENDING)
        self._refresh_display()

    def set_node_status(self, node: Union[str, WorkFlowNode], new_state: WorkFlowNodeState) -> bool:
        """
        Update the state of a specific node. 

        Args:
            node (Union[str, WorkFlowNode]): The name of a node or the node instance.
            new_state (WorkFlowNodeState): The new state to set.
        
        Returns:
            bool: True if the state was updated successfully, False otherwise.
        """
        flag = False
        try:
            if isinstance(node, str):
                node = self.get_node(node_name=node)
            node.set_status(new_state)
            flag = True
        except Exception as e:
            raise ValueError(f"An error occurs when setting node status: {e}")
        if flag:
            self._refresh_display()
        return flag
    
    def pending(self, node: Union[str, WorkFlowNode]) -> bool:
        return self.set_node_status(node=node, new_state=WorkFlowNodeState.PENDING)
    
    def running(self, node: Union[str, WorkFlowNode]) -> bool:
        return self.set_node_status(node=node, new_state=WorkFlowNodeState.RUNNING)
    
    def completed(self, node: Union[str, WorkFlowNode]) -> bool:
        return self.set_node_status(node=node, new_state=WorkFlowNodeState.COMPLETED)
    
    def failed(self, node: Union[str, WorkFlowNode]) -> bool:
        return self.set_node_status(node=node, new_state=WorkFlowNodeState.FAILED)
    
    def get_node_children(self, node: Union[str, WorkFlowNode]) -> List[str]:
        node_name = node if isinstance(node, str) else node.name
        if not self.node_exists(node=node):
            raise ValueError(f"Node `{node_name}` does not exists!")
        children = list(self.graph.successors(node_name))
        return children
    
    def get_node_predecessors(self, node: Union[str, WorkFlowNode]) -> List[str]:
        node_name = node if isinstance(node, str) else node.name
        if not self.node_exists(node=node):
            raise ValueError(f"Node `{node_name}` does not exists!")
        predecessors = list(self.graph.predecessors(node_name))
        return predecessors

    def get_uncomplete_initial_nodes(self) -> List[str]:
        initial_nodes = self.find_initial_nodes()
        are_initial_nodes_complete = [self.get_node(node_name).is_complete for node_name in initial_nodes]
        uncomplete_initial_nodes = []
        for node_name, is_complete in zip(initial_nodes, are_initial_nodes_complete):
            if not is_complete:
                uncomplete_initial_nodes.append(node_name)
        return uncomplete_initial_nodes 
    
    def get_all_paths_from_node(self, start_node: Union[str, WorkFlowNode]) -> List[List[str]]:

        if isinstance(start_node, str):
            start_node = self.get_node(node_name=start_node)
        start_node_name = start_node.name

        all_paths = []
        visited = set() # handle loop in the graph

        def dfs(current_node_name: str, path: List[str]):
            if current_node_name in visited:
                # 如果一个loop的end node只有指向loop的start node的边，那么添加这条路径
                if path and len(self.get_node_children(path[-1])) == 1:
                    all_paths.append(path.copy())
                return
            path.append(current_node_name)
            visited.add(current_node_name)
            children = self.get_node_children(current_node_name)
            if not children:
                all_paths.append(path.copy())
            else:
                for child in children:
                    dfs(child, path)
            
            path.pop()
            visited.remove(current_node_name)
        
        dfs(start_node_name, [])
        return all_paths

    def find_completed_leaf_nodes(self, start_node: Union[str, WorkFlowNode]) -> List[str]:

        if isinstance(start_node, str):
            start_node = self.get_node(node_name=start_node)
        start_node_name = start_node.name

        paths_starting_from_node = self.get_all_paths_from_node(start_node=start_node_name)
        last_completed_nodes = [] 
        for path in paths_starting_from_node:
            if not path:
                continue
            completed_node = None
            for path_node in path:
                if self.get_node(path_node).is_complete:
                    completed_node = path_node
                else:
                    break
            if completed_node and completed_node not in last_completed_nodes:
                last_completed_nodes.append(completed_node)
        last_completed_nodes = last_completed_nodes[::-1]
        return last_completed_nodes
    
    def find_completed_leaf_nodes_start_from_initial_nodes(self) -> List[str]:

        initial_nodes = self.find_initial_nodes()
        completed_leaf_nodes = []
        for initial_node in initial_nodes:
            for complete_node in self.find_completed_leaf_nodes(start_node=initial_node):
                if complete_node not in completed_leaf_nodes:
                    completed_leaf_nodes.append(complete_node)
        return completed_leaf_nodes
    
    def get_all_children_nodes(self, nodes: List[Union[str, WorkFlowNode]]) -> List[str]:

        node_names = [node if isinstance(node, str) else node.name for node in nodes]
        children_nodes = [] 
        for node_name in node_names:
            for child in self.get_node_children(node_name):
                if child not in children_nodes:
                    children_nodes.append(child)
        return children_nodes
    
    def filter_completed_nodes(self, nodes: List[Union[str, WorkFlowNode]]) -> List[str]:
        """
        remove completed nodes from `nodes`
        """
        node_names = [node if isinstance(node, str) else node.name for node in nodes]
        uncompleted_nodes = [] 
        for node_name in node_names:
            if self.get_node(node_name).is_complete:
                continue
            uncompleted_nodes.append(node_name)
        return uncompleted_nodes
    
    def get_candidate_children_nodes(self, completed_nodes: List[Union[str, WorkFlowNode]]) -> List[str]:
        """
        Return the next set of possible tasks to execute. If there are no loops in the graph, consider only the uncompleted children. 
        If there exists loops, also consider the previous completed tasks.

        Args:
            completed_nodes (List[Union[str, WorkFlowNode]]): A list of completed nodes.
            
        Returns:
            List[str]: List of node names that are candidates for execution.
        """
        node_names = [node if isinstance(node, str) else node.name for node in completed_nodes]
        has_loop = (len(self._loops) > 0)
        if has_loop:
            # if there exists loops, we need to check the completed nodes and their children nodes
            uncompleted_children_nodes = []
            for node_name in node_names:
                children_nodes = self.get_all_children_nodes(nodes=[node_name])
                if self.is_loop_end(node=node_name):
                    current_uncompleted_children_nodes = [] 
                    for child in children_nodes:
                        if self.is_loop_start(node=child):
                            # node_name是一个环的结束的时候，如果它的子节点是环的开始，那么无论它是否completed，都添加到下一步可执行的操作
                            current_uncompleted_children_nodes.append(child)
                        else:
                            # node_name是环的结束，但是子节点不是环的开始时，需要检查child是否已经completed，只添加未完成的任务
                            current_uncompleted_children_nodes.extend(self.filter_completed_nodes(nodes=[child]))
                else:
                    current_uncompleted_children_nodes = self.filter_completed_nodes(nodes=children_nodes)
                for child in current_uncompleted_children_nodes:
                    if child not in uncompleted_children_nodes:
                        uncompleted_children_nodes.append(child)
        else:
            # 不存在环的时候直接得到所有的子节点，并去掉其中已完成的部分
            children_nodes = self.get_all_children_nodes(nodes=node_names)
            uncompleted_children_nodes = self.filter_completed_nodes(nodes=children_nodes)

        return uncompleted_children_nodes
    
    def are_dependencies_complete(self, node_name: str) -> bool:
        """
        Check if all predecessors for a node are complete.

        Args:
            node_name (str): The name of the task/node to check.
        
        Returns:
            bool: True if all predecessors are complete, False otherwise.
        """
        has_loop = (len(self._loops) > 0)
        predecessors = self.get_node_predecessors(node=node_name)
        if has_loop and self.is_loop_start(node=node_name):
            flag = True 
            for pre in predecessors:
                if self.is_loop_end(pre):
                    pass 
                else:
                    flag &= self.get_node(pre).is_complete
        else:
            flag = all(self.get_node(pre).is_complete for pre in predecessors)
        return flag

    def filter_nodes_with_uncompleted_predecessors(self, nodes: List[Union[str, WorkFlowNode]]) -> List[str]:
        node_names = [node if isinstance(node, str) else node.name for node in nodes]
        nodes_with_completed_predecessors = [] 
        for node_name in node_names:
            if self.are_dependencies_complete(node_name=node_name):
                nodes_with_completed_predecessors.append(node_name)
        return nodes_with_completed_predecessors

    def get_next_candidate_nodes(self) -> List[str]:

        # `find_initial_nodes` classifies a node as initial purely from data readiness
        # (its required inputs are a subset of the workflow inputs). A node can satisfy that
        # while still having incoming edges — an explicit control edge, or an inferred edge
        # feeding one of its optional inputs. Those nodes must not start before their
        # predecessors, so filter the initial candidates by predecessor completion as well.
        uncomplete_initial_nodes = self.get_uncomplete_initial_nodes()
        ready_initial_nodes = self.filter_nodes_with_uncompleted_predecessors(uncomplete_initial_nodes)
        if len(ready_initial_nodes) > 0:
            return ready_initial_nodes

        # find the last completed nodes in all paths starting from initial nodes.
        completed_leaf_nodes = self.find_completed_leaf_nodes_start_from_initial_nodes()

        # obtain children nodes of last completed nodes which are uncompleted (consider previous completed tasks if there exists loops)
        # children_nodes = self.get_all_children_nodes(nodes=completed_leaf_nodes)
        # uncompleted_children_nodes = self.filter_completed_nodes(nodes=children_nodes)
        candidate_children_nodes = self.get_candidate_children_nodes(completed_nodes=completed_leaf_nodes)

        # check whether all the predecessors are completed
        # children_nodes_with_complete_predecessors = self.filter_nodes_with_uncompleted_predecessors(uncompleted_children_nodes)
        children_nodes_with_complete_predecessors = self.filter_nodes_with_uncompleted_predecessors(candidate_children_nodes)

        return children_nodes_with_complete_predecessors

    def next(self) -> List[WorkFlowNode]:
        if self.is_complete:
            return [] 
        candidate_node_names = self.get_next_candidate_nodes()
        candidate_tasks = [self.get_node(node_name=node_name) for node_name in candidate_node_names]
        return candidate_tasks

    def step(self, source_node: Union[str, WorkFlowNode], target_node: Union[str, WorkFlowNode]):

        if source_node is None:
            self.running(target_node)
            return
        
        source_node_name = source_node if isinstance(source_node, str) else source_node.name
        target_node_name = target_node if isinstance(target_node, str) else target_node.name
        source_node_status = self.get_node_status(source_node_name)
        if source_node_status != WorkFlowNodeState.COMPLETED:
            raise ValueError(f"The state of `source_node` should be WorkFlowNodeState.COMPLETED, but found {source_node_status}")
        # set the state of `target_node` to WorkFlowNodeState.RUNNING
        if self.is_loop_end(source_node_name) and self.is_loop_start(target_node_name):
            loops = self.find_loops_with_start_and_end(
                start_node=target_node_name, end_node=source_node_name
            )
            loop_nodes = set(sum(loops, []))
            for loop_node in loop_nodes:
                self.pending(node=loop_node)
        if not self.edge_exists(edge=(source_node_name, target_node_name)):
            # the execution doesn't follow an edge means re-executing a previous subtask due to errors or incomplete output
            # find a path that contains both source_node and target node and set them as "pending"
            all_paths = self.get_all_paths_from_node(start_node=target_node_name)
            for path in all_paths:
                if source_node_name in path:
                    for node_name in path:
                        self.pending(node=node_name)
        self.running(node=target_node_name)
    
    def get_node_description(self, node: Union[str, WorkFlowNode]) -> str:

        if isinstance(node, str):
            node = self.get_node(node_name=node)
        
        def format_parameters(params: List[Parameter]) -> str:
            if not params:
                return "  - None"
            # return "\n".join(f"  - {param.name} ({param.type}): {param.description}" for param in params)
            return "\n".join(f"  - {param.name} ({param.type})" for param in params)
        
        def format_agents(agent_names: List[str]) -> str:
            if not agent_names:
                return "None"
            return "\n".join(f"  - {name}" for name in agent_names)
        
        def format_action_graph(action_graph: ActionGraph) -> str:
            if action_graph is None:
                return "  - None"
            return type(action_graph).__name__
        
        desc = (
            f"Name: {node.name}\n"
            # f"Description: {node.description}\n"
            f"Inputs:\n{format_parameters(node.inputs)}\n"
            f"Outputs:\n{format_parameters(node.outputs)}\n"
            f"Agents:\n{format_agents(node.get_agents())}\n"
            f"Action Graph:\n{format_action_graph(node.action_graph)}"
        )
        return desc

    @staticmethod
    def _matplotlib_gui_available() -> bool:
        """Return whether the current Matplotlib backend can open a GUI window."""
        import matplotlib
        from matplotlib.backends.registry import backend_registry

        try:
            _backend, framework = backend_registry.resolve_backend(matplotlib.get_backend())
        except Exception:
            return False
        return framework is not None

    def _get_open_display_figure(self):
        """Return the live display figure, or None if it was closed."""
        fig = getattr(self, "_display_fig", None)
        if fig is None:
            return None
        import matplotlib.pyplot as plt
        try:
            still_open = plt.fignum_exists(fig.number)
        except Exception:
            still_open = False
        if still_open:
            return fig
        self._display_fig = None
        self._display_live = False
        return None

    def _redraw_display(self, fig) -> None:
        """Draw the current graph and node statuses onto ``fig``."""
        import matplotlib.pyplot as plt

        status_colors = {
            WorkFlowNodeState.PENDING: 'lightgray',
            WorkFlowNodeState.RUNNING: 'orange',
            WorkFlowNodeState.COMPLETED: 'green',
            WorkFlowNodeState.FAILED: 'red'
        }

        fig.clf()
        ax = fig.add_subplot(111)

        node_colors = [status_colors.get(self.get_node_status(node), 'lightgray') for node in self.graph.nodes]
        node_labels = {node: self.get_node_description(data["ref"]) for node, data in self.graph.nodes(data=True)}

        if len(self.graph.nodes) == 1:
            single_node = list(self.graph.nodes)[0]
            pos = {single_node: (0, 0)}
        else:
            pos = nx.shell_layout(self.graph)

        nx.draw(
            self.graph, pos, ax=ax, with_labels=False, node_color=node_colors, edge_color='black',
            node_size=1500, font_size=8, font_color='black', font_weight='bold'
        )

        if len(self.graph.nodes) == 1:
            for node, (x, y) in pos.items():
                ax.text(x+0.005, y, node_labels[node], ha='left', va='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.7))
        else:
            y_positions = [y for _, y in pos.values()]
            y_min, y_max = min(y_positions), max(y_positions)
            lower_third_boundary = y_min + (y_max - y_min) / 3

            text_offsets = {}
            for node, (x, y) in pos.items():
                if y < lower_third_boundary:
                    text_offsets[node] = (x-0.2, y + 0.23)
                else:
                    text_offsets[node] = (x-0.2, y - 0.23)

            for node, (x, y) in text_offsets.items():
                ax.text(x, y, node_labels[node], ha='left', va='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.7))

        edge_labels = nx.get_edge_attributes(self.graph, 'priority')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, ax=ax)

        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label=status.name, markersize=10, markerfacecolor=color)
            for status, color in status_colors.items()
        ]
        ax.legend(handles=legend_elements, title="Workflow Node Status", loc='upper left', fontsize='medium')
        ax.set_title("Workflow Graph")

    def _refresh_display(self) -> None:
        """Redraw an open non-blocking display. Never raises into workflow execution."""
        if not getattr(self, "_display_live", False):
            return
        if getattr(self, "_display_refreshing", False):
            return
        fig = self._get_open_display_figure()
        if fig is None:
            return

        self._display_refreshing = True
        try:
            self._redraw_display(fig)
            fig.canvas.draw()
            if self._matplotlib_gui_available():
                fig.canvas.flush_events()
        except Exception as exc:
            logger.warning("Failed to refresh workflow graph display: {}", exc)
            self._display_live = False
        finally:
            self._display_refreshing = False

    def close_display(self) -> None:
        """Close the window opened by ``display()``."""
        import matplotlib.pyplot as plt

        self._display_live = False
        fig = getattr(self, "_display_fig", None)
        self._display_fig = None
        if fig is None:
            return
        try:
            plt.close(fig)
        except Exception as exc:
            logger.debug("Failed to close workflow graph display: {}", exc)

    def display(self, block: bool = False):
        """
        Display the workflow graph with node and edge attributes.
        Nodes are colored based on their status.

        The window is non-blocking by default, so later steps keep running and
        the figure refreshes when a node status changes. Pass ``block=True`` to
        wait until the window is closed. Call ``close_display()`` to close a
        non-blocking window.

        Args:
            block: When True, block until the window is closed. Defaults to False.

        Returns:
            The Matplotlib figure, or None when the graph has no nodes.
        """
        import matplotlib.pyplot as plt

        if not self.graph.nodes:
            print("Graph is empty. No nodes to display.")
            return None

        fig = self._get_open_display_figure()
        if fig is None:
            fig = plt.figure(figsize=(12, 8))
            self._display_fig = fig

        self._display_live = not block
        self._display_refreshing = True
        try:
            self._redraw_display(fig)
        finally:
            self._display_refreshing = False

        if not self._matplotlib_gui_available():
            fig.canvas.draw()
            if block:
                self._display_live = False
            return fig

        if block:
            plt.show(block=True)
            self._display_fig = None
            self._display_live = False
            return fig

        plt.show(block=False)
        try:
            fig.canvas.flush_events()
        except Exception as exc:
            logger.debug("Failed to flush workflow graph display events: {}", exc)
        return fig

    def get_workflow_description(self) -> str:

        def format_param_requirement(required: bool):
            return "required" if required else "optional"
        
        def format_parameters(params: List[Parameter]) -> str:
            if not params:
                return "None"
            return "\n".join(
                f"  - {param.name} ({param.type}, {format_param_requirement(param.required)}): "
                f"{param.description}" for param in params
            )
        
        subtask_texts = [] 
        for node in self.nodes:
            text = (
                f"Task Name: {node.name}\n"
                f"Description: {node.description}\n"
                f"Inputs:\n{format_parameters(node.inputs)}\n"
                f"Outputs:\n{format_parameters(node.outputs)}"
            )
            subtask_texts.append(text)
        workflow_desc = "\n\n".join(subtask_texts)
        return workflow_desc


    def _check_workflow_inputs_outputs(self):
        """Checks if the workflow inputs and outputs can be satisfied by the nodes in the workflow graph.
        Raises error If any workflow input is not used by a node, or if any workflow output cannot be found from a node's output.
        """
        workflow_inputs_outputs = {"inputs": self.workflow_inputs_dict, "outputs": self.workflow_outputs_dict}
        workflow_input_in_nodes = {workflow_input_name: False for workflow_input_name in self.workflow_inputs_dict}
        workflow_output_in_nodes = {workflow_output_name: False for workflow_output_name in self.workflow_outputs_dict}
        in_nodes = {"inputs": workflow_input_in_nodes, "outputs": workflow_output_in_nodes}

        def _check_node(node: WorkFlowNode, inputs_or_outputs: Literal["inputs", "outputs"]) -> WorkFlowNode:

            # "input" or "output"
            input_or_output = inputs_or_outputs[:-1]
            node_inputs_or_outputs = getattr(node, inputs_or_outputs)

            for node_input_or_output in node_inputs_or_outputs:
                if node_input_or_output.name in in_nodes[inputs_or_outputs]:
                    in_nodes[inputs_or_outputs][node_input_or_output.name] = True
                    validate_param(
                        workflow_inputs_outputs[inputs_or_outputs][node_input_or_output.name],
                        node_input_or_output,
                        f"workflow {input_or_output}",
                        f"node '{node.name}' {input_or_output}",
                    )

            return node
            

        for i, node in enumerate(self.nodes):
            self.nodes[i] = _check_node(node, "inputs")
            self.nodes[i] = _check_node(node, "outputs")

        if not all(in_nodes["inputs"].values()):
            missing_inputs = [input_name for input_name, in_node in in_nodes["inputs"].items() if not in_node]
            raise ValueError(f"Not all workflow inputs are used by nodes: {missing_inputs}")

        if not all(in_nodes["outputs"].values()):
            missing_outputs = [output_name for output_name, in_node in in_nodes["outputs"].items() if not in_node]
            raise ValueError(f"Not all workflow outputs are found in nodes: {missing_outputs}")


    def _check_agents(self):
        """Checks if each node's inputs and outputs can be satisfied by the agents assigned to the node."""
        for node in self.nodes:
            node.check_agents()


    def validate_workflow_graph(self):
        """
        Validates the workflow graph by checking:
            - The workflow structure
            - Workflow inputs and outputs can be derived from nodes
            - Nodes' inputs and outputs can be derived from agents
        """
        self._validate_workflow_structure()
        self._check_workflow_inputs_outputs()
        self._check_agents()


    def get_config(self) -> dict:
        """
        Get a dictionary containing all necessary configuration to recreate this workflow graph.
        
        Returns:
            dict: A configuration dictionary that can be used to initialize a new WorkFlowGraph instance
            with the same properties as this one.
        """
        config = self.to_dict() 
        config.pop("graph", None)
        return config


    @classmethod
    def from_dict(cls, data: Dict, **kwargs) -> 'WorkFlowGraph':
        """
        Create a WorkFlowGraph instance from a dictionary.

        Agent entries follow the same BaseModule revival rules as the rest of the
        workflow data: string agents stay as references, dict agents without a
        top-level `class_name` stay as config dicts, and dict agents with a top-level
        `class_name` are converted into live `Agent` instances during construction.
        Use a top-level `class_name` only when that eager revival is intended; otherwise
        leave it out so `AgentManager.add_agents_from_workflow` can materialize the
        agent later with the desired `llm_config`/`tools`.

        Args:
            data (Dict): The serialized workflow graph.
            **kwargs: Additional keyword arguments forwarded to `BaseModule.from_dict`.
        """
        # Delegate the actual data loading (class_name dispatch, nested _process_data,
        # construction) to BaseModule.
        workflow_graph: WorkFlowGraph = super().from_dict(data, **kwargs)

        # Validate structure and node/agent parameter compatibility at load time. This is
        # the one behavior BaseModule.from_dict does not provide.
        workflow_graph.validate_workflow_graph()
        return workflow_graph


class SequentialWorkFlowGraph(WorkFlowGraph):

    """
    A linear workflow graph with a single path from start to end.

    Args:
        goal (str): The goal of the workflow.
        tasks (List[dict]): A list of tasks with their descriptions and inputs. Each task should have the following format:
            {
                "name": str,
                "description": str,
                "inputs": [{"name": str, "type": str, "required": bool, "description": str}, ...],
                "outputs": [{"name": str, "type": str, "required": bool, "description": str}, ...],
                "prompt": str, 
                "prompt_template": PromptTemplate, 
                "system_prompt" (optional): str, default is DEFAULT_SYSTEM_PROMPT,
                "output_parser" (optional): Type[ActionOutput],
                "parse_mode" (optional): str, default is "str" 
                "parse_func" (optional): Callable,
                "title_format" (optional): str ,
                "tool_names" (optional): List[str]
            }
    """

    def __init__(self, goal: str, tasks: List[dict], **kwargs):
        nodes = self._infer_nodes_from_tasks(tasks=tasks)
        edges = self._infer_edges_from_nodes(nodes=nodes)
        super().__init__(goal=goal, nodes=nodes, edges=edges, **kwargs)
    
    def _infer_nodes_from_tasks(self, tasks: List[dict]) -> List[WorkFlowNode]:
        nodes = [self._infer_node_from_task(task=task) for task in tasks]
        return nodes
    
    def _infer_node_from_task(self, task: dict) -> WorkFlowNode:

        node_name = task.get("name", None)
        if not node_name:
            raise ValueError("The `name` for the following task is required: {}".format(task))
        node_description = task.get("description", None)
        if not node_description:
            raise ValueError("The `description` for the following task is required: {}".format(task))
        agent_prompt = task.get("prompt", None)
        agent_prompt_template = task.get("prompt_template", None)
        if not agent_prompt and not agent_prompt_template:
            raise ValueError("The `prompt` or `prompt_template` for the following task is required: {}".format(task))
        
        inputs = task.get("inputs", [])
        outputs = task.get("outputs", [])
        agent_name = generate_dynamic_class_name(node_name+" Agent")
        agent_description = node_description # .replace("task", "agent")
        agent_system_prompt = task.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
        agent_output_parser = task.get("output_parser", None)
        agent_parse_mode = task.get("parse_mode", "str")
        agent_parse_func = task.get("parse_func", None)
        # `parse_title` is the legacy key name; `title_format` is what CustomizeAgent expects.
        # Accept both, preferring `title_format`, so old saved configs keep working.
        agent_title_format = task.get("title_format", task.get("parse_title", None))
        tool_names = task.get("tool_names", None)
        # tools = task.get("tools", [])
        # tool_names = []
        # if tools:
        #     for tool in tools:
        #         if isinstance(tool,Toolkit):
        #             tool_names.append(tool.name)
        #         elif isinstance(tool, Tool):
        #             tool_names.append(tool.name)
        #         else:
        #             tool_names.append(tool)

        node = WorkFlowNode.from_dict(
            {
                "name": node_name,
                "description": node_description,
                "inputs": inputs,
                "outputs": outputs,
                "agents": [
                    {
                        "name": agent_name,
                        "description": agent_description,
                        "prompt": agent_prompt,
                        "prompt_template": agent_prompt_template, 
                        "system_prompt": agent_system_prompt,
                        "inputs": inputs,
                        "outputs": outputs,
                        "output_parser": agent_output_parser,
                        "parse_mode": agent_parse_mode,
                        "parse_func": agent_parse_func,
                        "title_format": agent_title_format,
                        "tool_names": tool_names
                    }
                ],
            }
        )
        return node
    
    def get_graph_info(self, **kwargs) -> dict:
        """
        Get the information of the workflow graph.
        """
        config = {
            "class_name": self.__class__.__name__,
            "goal": self.goal, 
            "tasks": [
                {
                    "name": node.name,
                    "description": node.description,
                    "inputs": [param.to_dict(ignore=["class_name"]) for param in node.inputs],
                    "outputs": [param.to_dict(ignore=["class_name"]) for param in node.outputs],
                    "prompt": node.agents[0].get("prompt", None),
                    "prompt_template": node.agents[0].get("prompt_template", None).to_dict() if node.agents[0].get("prompt_template", None) else None,
                    "system_prompt": node.agents[0].get("system_prompt", None),
                    "parse_mode": node.agents[0].get("parse_mode", "str"), 
                    "parse_func": node.agents[0].get("parse_func", None).__name__ if node.agents[0].get("parse_func", None) else None,
                    "title_format": node.agents[0].get("title_format", None),
                    "tool_names": node.agents[0].get("tool_names", None)
                }
                for node in self.nodes
            ]
        }
        return config
    
    def save_module(self, path: str, ignore: List[str] = [], **kwargs):
        """
        Save the workflow graph to a module file.
        """
        logger.info("Saving {} to {}", self.__class__.__name__, path)
        config = self.get_graph_info()
        for ignore_key in ignore:
            config.pop(ignore_key, None)
        make_parent_folder(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
        return path
    
    def get_config(self) -> Dict:
        """
        Get a dictionary containing all necessary configuration to recreate this workflow graph.
        
        Returns:
            dict: A configuration dictionary that can be used to initialize a new SequentialWorkFlowGraph instance
            with the same properties as this one.
        """
        return self.get_graph_info()

    @classmethod
    def from_dict(cls, data: Dict, **kwargs) -> 'SequentialWorkFlowGraph':
        return cls._create_instance(data)
    

class SEWWorkFlowGraph(SequentialWorkFlowGraph):

    def __init__(self, **kwargs):
        goal = kwargs.pop("goal", SEW_WORKFLOW["goal"])
        tasks = kwargs.pop("tasks", SEW_WORKFLOW["tasks"])
        super().__init__(goal=goal, tasks=tasks, **kwargs)
