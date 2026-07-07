"""
Shared WorkFlowGraph glue for workflow-backed program adapters.

Optimizer-specific workflow adapters (e.g. ``SEWWorkFlowAdapter``,
``MapElitesWorkFlowAdapter``) all need the same plumbing around a
:class:`~evoagentx.workflow.workflow_graph.WorkFlowGraph`: extracting each agent's
editable prompt text (and its role), injecting mutated prompts back into a graph copy,
reconciling the ``llm`` / ``llm_config`` pair, and running the graph end-to-end.
:class:`WorkFlowAdapterMixin` centralizes that plumbing so each optimizer's workflow
adapter only implements its own unit registration and ``from_snapshot`` reconstruction.

Usage: mix into a ``ProgramAdapter`` subclass *before* the adapter base class so the
mixin's ``execute`` / ``async_execute`` win the MRO, and call
:meth:`WorkFlowAdapterMixin._init_workflow_runtime` at the start of ``__init__``::

    class MyWorkFlowAdapter(WorkFlowAdapterMixin, MyProgramAdapter):
        def __init__(self, graph, tools=None, llm=None, llm_config=None, max_execution_steps=5):
            self._init_workflow_runtime(graph, tools, llm, llm_config, max_execution_steps)
            ...
            super().__init__(...)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from ..workflow.workflow_graph import WorkFlowGraph, WorkFlowNode

from ..core.callbacks import suppress_logger_info
from ..core.logging import logger


class WorkFlowAdapterMixin:
    """
    Graph-execution glue shared by workflow-backed program adapters.

    Provides prompt extraction/injection over a ``WorkFlowGraph``'s agents, graph
    copying/rebuilding, ``llm``/``llm_config`` reconciliation, and ``execute`` /
    ``async_execute`` that run the graph end-to-end via a fresh ``AgentManager`` +
    ``WorkFlow`` per call.

    Concurrency: each call runs on its own copy of the graph (and fresh agents), so a
    single adapter instance is safe to execute concurrently — e.g. an ``evaluate_fn``
    that gathers ``async_execute`` over a dataset, or fans ``execute`` out across a
    thread pool. The ``llm``, ``tools``, and ``llm_config`` dependencies are still
    shared by reference across those concurrent runs (deep-copying LLM clients /
    toolkits is undesirable), so they must themselves be safe for concurrent use.
    Evaluators that bypass ``execute`` and run ``adapter.graph`` directly must copy the
    graph first; workflow execution mutates node status.
    """

    def _init_workflow_runtime(
        self,
        graph: "WorkFlowGraph",
        tools: Optional[List[Any]],
        llm: Optional[Any],
        llm_config: Optional[Any],
        max_execution_steps: int,
    ) -> None:
        """Validate the graph and bind the shared workflow runtime dependencies."""
        from ..workflow.workflow_graph import WorkFlowGraph

        if not isinstance(graph, WorkFlowGraph):
            raise TypeError(f"`graph` must be a WorkFlowGraph instance, got {type(graph).__name__}.")
        self.graph = graph
        self.tools = tools
        self.llm, self.llm_config = self._reconcile_llm(llm, llm_config)
        self.max_execution_steps = max_execution_steps

    def _reconcile_llm(self, llm: Optional[Any], llm_config: Optional[Any]) -> Tuple[Any, Any]:
        """Return a consistent ``(llm, llm_config)`` pair, deriving whichever is missing.

        Both are used when running the workflow, so they must agree. ``llm`` wins when
        both are supplied; ``llm_config`` is then taken from ``llm.config``.
        """
        from ..models.base_model import BaseLLM
        from ..models.model_utils import create_llm_instance

        if llm is None and llm_config is None:
            raise ValueError("Provide at least one of `llm` or `llm_config`.")
        if llm is not None:
            if not isinstance(llm, BaseLLM):
                raise TypeError(f"`llm` must be a BaseLLM instance, got {type(llm).__name__}.")
            if llm_config is not None and llm_config is not llm.config:
                logger.warning(
                    f"Both `llm` and `llm_config` were provided to {type(self).__name__}; "
                    "using `llm` and ignoring the passed `llm_config` (taking llm.config)."
                )
            return llm, llm.config
        return create_llm_instance(llm_config), llm_config

    # -- prompt extraction --------------------------------------------------
    @staticmethod
    def _template_instruction(template: Any) -> Optional[str]:
        """Read the optimizable ``instruction`` from a prompt template (dict or PromptTemplate)."""
        if template is None:
            return None
        instruction = template.get("instruction") if isinstance(template, dict) else getattr(template, "instruction", None)
        return instruction if isinstance(instruction, str) and instruction.strip() else None

    @staticmethod
    def _set_template_instruction(template: Any, text: str) -> None:
        """Write ``text`` back as the prompt template's ``instruction`` (dict or PromptTemplate)."""
        if isinstance(template, dict):
            template["instruction"] = text
        elif hasattr(template, "set_instruction"):
            template.set_instruction(text)
        else:
            setattr(template, "instruction", text)

    @classmethod
    def _agent_prompt_text(cls, agent: Any) -> Optional[str]:
        """The active optimizable prompt text of an agent config dict.

        Mirrors ``CustomizeAgent``: when both are present ``prompt_template`` wins, so the
        template's ``instruction`` is the optimizable text; otherwise the raw ``prompt`` is.
        Returns ``None`` for agents (e.g. bare string references) with no editable prompt.
        """
        if not isinstance(agent, dict):
            return None
        if agent.get("prompt_template") is not None:
            return cls._template_instruction(agent["prompt_template"])
        prompt = agent.get("prompt")
        return prompt if isinstance(prompt, str) and prompt.strip() else None

    @classmethod
    def _set_agent_prompt_text(cls, agent: Dict[str, Any], text: str) -> None:
        """Inject ``text`` into the agent's active prompt slot (template instruction or prompt)."""
        if agent.get("prompt_template") is not None:
            cls._set_template_instruction(agent["prompt_template"], text)
        else:
            agent["prompt"] = text

    @classmethod
    def _iter_prompt_agents(cls, node: "WorkFlowNode") -> List[Tuple[str, Dict[str, Any]]]:
        """Yield ``(agent_name, agent_dict)`` for every agent on ``node`` with editable prompt text."""
        result: List[Tuple[str, Dict[str, Any]]] = []
        for agent in node.agents or []:
            if not isinstance(agent, dict) or cls._agent_prompt_text(agent) is None:
                continue
            name = agent.get("name")
            if name:
                result.append((name, agent))
        return result

    @classmethod
    def _extract_prompts_and_roles(
        cls, graph: "WorkFlowGraph"
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Build ``{agent_name: prompt}`` and ``{agent_name: role}`` across all nodes' agents.

        A node may carry several agents, so units are keyed by (workflow-unique) agent name
        rather than node name. Each agent's role defaults to its own description, falling back
        to the node's description.
        """
        prompts: Dict[str, str] = {}
        roles: Dict[str, str] = {}
        for node in graph.nodes:
            for agent_name, agent in cls._iter_prompt_agents(node):
                if agent_name in prompts:
                    # Units are keyed by agent name; a duplicate would make reconstruction
                    # ambiguous (which agent does a mutated prompt belong to?).
                    raise ValueError(
                        f"Duplicate agent name {agent_name!r} found in the workflow graph. "
                        "Each agent must have a unique name so its prompt can be optimized "
                        "and reconstructed unambiguously."
                    )
                prompts[agent_name] = cls._agent_prompt_text(agent)
                role = agent.get("description") or node.description
                if role:
                    roles[agent_name] = role
        return prompts, roles

    def _copy_graph(self) -> "WorkFlowGraph":
        """Return an independent copy of the graph (its nodes/edges are deep-copied)."""
        from ..workflow.workflow_graph import WorkFlowGraph

        # Constructing from an existing graph deep-copies its nodes and edges while building
        # a fresh internal MultiDiGraph + lock, so the original is never mutated and we avoid
        # deep-copying the un-pickleable threading.Lock the graph holds. (cf. Evaluator.)
        return WorkFlowGraph(goal=self.graph.goal, graph=self.graph)

    def _rebuild_graph(self, prompts: Dict[str, str]) -> "WorkFlowGraph":
        """Deep-copy the base graph and inject the (possibly mutated) prompts into its agents."""
        new_graph = self._copy_graph()
        for node in new_graph.nodes:
            for agent_name, agent in self._iter_prompt_agents(node):
                if agent_name in prompts:
                    self._set_agent_prompt_text(agent, prompts[agent_name])
        new_graph.reset_graph()
        return new_graph

    # -- run the workflow ---------------------------------------------------
    def _build_workflow(self):
        """Assemble a fresh AgentManager + WorkFlow over a per-call graph copy."""
        from ..agents import AgentManager
        from ..workflow import WorkFlow

        # Run each execution on its own graph copy (not the shared self.graph), so
        # concurrent execute() / async_execute() calls on a single adapter — e.g. an
        # evaluate_fn that gathers over a dataset — never share mutable node-status
        # state. WorkFlow.execute_task mutates the graph (step()/completed()), so a
        # shared graph would let parallel runs clobber each other.
        graph = self._copy_graph()
        graph.reset_graph()
        agent_manager = AgentManager()
        agent_manager.add_agents_from_workflow(graph, llm_config=self.llm_config, tools=self.tools)
        return WorkFlow(
            graph=graph,
            agent_manager=agent_manager,
            llm=self.llm,
            max_execution_steps=self.max_execution_steps,
        )

    def execute(self, inputs: Optional[dict] = None, **kwargs) -> Any:
        # Suppress the workflow's per-step INFO logs so an evaluate_fn running many
        # executions doesn't flood the console. suppress_logger_info is contextvar-scoped
        # to this thread / asyncio-task, so it stays correct under concurrent execution.
        workflow = self._build_workflow()
        with suppress_logger_info():
            return workflow.execute(inputs=dict(inputs or {}), **kwargs)

    async def async_execute(self, inputs: Optional[dict] = None, **kwargs) -> Any:
        workflow = self._build_workflow()
        with suppress_logger_info():
            return await workflow.async_execute(inputs=dict(inputs or {}), **kwargs)


__all__ = ["WorkFlowAdapterMixin"]
