import threading
import contextvars
from tqdm import tqdm
# from time import time
from typing import Callable, Optional, Any, List, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from tqdm.asyncio import tqdm_asyncio

from ..core.logging import logger
from ..core.message import Message
from ..models.base_model import BaseLLM
from ..benchmark.benchmark import Benchmark
from ..workflow.workflow import WorkFlow
from ..workflow.action_graph import ActionGraph
from ..workflow.workflow_graph import WorkFlowGraph
from ..agents.agent_manager import AgentManager


class Evaluator:
    """
    A class for evaluating the performance of a workflow.
    """
    def __init__(
        self, 
        llm: BaseLLM,
        num_workers: int = 1, 
        agent_manager: Optional[AgentManager] = None,
        collate_func: Optional[Callable] = None, 
        output_postprocess_func: Optional[Callable] = None, 
        verbose: Optional[bool] = None, 
        **kwargs
    ):
        """
        Initialize the Evaluator.

        Args:
            llm (BaseLLM): The LLM to use for evaluation.
            num_workers (int): The number of parallel workers to use for evaluation. Default is 1. 
            agent_manager (AgentManager, optional): The agent manager used to construct the workflow. Only used when the workflow graph is a WorkFlowGraph.
            collate_func (Callable, optional): A function to collate the benchmark data. 
                It receives a single example from the benchmark and the output (which should be a dictionary) will serve as inputs  
                to the `execute` function of an WorkFlow (or ActionGraph) instance. 
                Note that the keys in the collated output should match the inputs of the workflow.
                The default is a lambda function that returns the example itself. 
            output_postprocess_func (Callable, optional): A function to postprocess the output of the workflow. 
                It receives the output of an WorkFlow instance (str) or an ActionGraph instance (dict) as input 
                and the output will be passed to the `evaluate` function of the benchmark. 
                The default is a lambda function that returns the output itself.
            verbose (bool, optional): Whether to print the evaluation progress.
        """
        self.llm = llm
        self.num_workers = num_workers
        self.agent_manager = agent_manager
        self._thread_agent_managers = {}
        self.collate_func = collate_func or (lambda x: x)
        self.output_postprocess_func = output_postprocess_func or (lambda x: x)
        self.verbose = verbose
        # {example_id: {"prediction": Any, "label": Any, "metrics": dict, "trajectory" (WorkFlowGraph only): List[Message]}}
        self._evaluation_records = {}
        self.kwargs = kwargs

    def _get_eval_data(self, benchmark: Benchmark, eval_mode: str = "test", indices: Optional[List[int]] = None, sample_k: Optional[int] = None, seed: Optional[int] = None) -> List[dict]:

        assert eval_mode in ["test", "dev", "train"], f"Invalid eval_mode: {eval_mode}. Choices: ['test', 'dev', 'train']"
        if eval_mode == "test":
            data = benchmark.get_test_data(indices=indices, sample_k=sample_k, seed=seed)
        elif eval_mode == "dev":
            data = benchmark.get_dev_data(indices=indices, sample_k=sample_k, seed=seed)
        else:
            data = benchmark.get_train_data(indices=indices, sample_k=sample_k, seed=seed)
        return data
    
    def evaluate(
        self, 
        graph: Union[WorkFlowGraph, ActionGraph],
        benchmark: Benchmark, 
        eval_mode: str = "test", 
        indices: Optional[List[int]] = None, 
        sample_k: Optional[int] = None, 
        seed: Optional[int] = None, 
        verbose: Optional[bool] = None,
        update_agents: Optional[bool] = False,
        **kwargs
    ) -> dict:
        """
        Evaluate the performance of the workflow on the benchmark.

        Args:
            graph (WorkFlowGraph or ActionGraph): The workflow to evaluate.
            benchmark (Benchmark): The benchmark to evaluate the workflow on.
            eval_mode (str): which split of the benchmark to evaluate the workflow on. Choices: ["test", "dev", "train"].
            indices (List[int], optional): The indices of the data to evaluate the workflow on.
            sample_k (int, optional): The number of data to evaluate the workflow on. If provided, a random sample of size `sample_k` will be used.
            verbose (bool, optional): Whether to print the evaluation progress. If not provided, the `self.verbose` will be used.
            update_agents (bool, optional): Whether to update the agents in the agent manager. Only used when the workflow graph is a WorkFlowGraph.
        Returns:
            dict: The average metrics of the workflow evaluation.
        """
        # clear the evaluation records
        self._evaluation_records.clear()

        # update the agents in the agent manager
        if isinstance(graph, WorkFlowGraph) and update_agents:
            if self.agent_manager is None:
                raise ValueError(f"`agent_manager` is not provided in {type(self).__name__}. Please provide an agent manager when evaluating a WorkFlowGraph.")
            self.agent_manager.update_agents_from_workflow(workflow_graph=graph, llm_config=self.llm.config, **kwargs)
        
        data = self._get_eval_data(benchmark=benchmark, eval_mode=eval_mode, indices=indices, sample_k=sample_k, seed=seed)
        results = self._evaluate_graph(graph=graph, data=data, benchmark=benchmark, verbose=verbose, **kwargs)
        return results
    
    def _execute_workflow_graph(self, graph: WorkFlowGraph, inputs: dict, return_trajectory: bool = False, **kwargs) -> Union[str, Tuple[str, List[Message]]]:
        """
        Execute the workflow graph and return the output.

        Args:
            graph (WorkFlowGraph): The workflow graph to execute
            inputs (dict): The inputs to the workflow graph
            **kwargs: Additional arguments for workflow graph execution

        Returns:
            str: The output of the workflow graph
        """
        if self.agent_manager is None:
            raise ValueError(f"`agent_manager` is not provided in {type(self).__name__}. Please provide an agent manager when evaluating a WorkFlowGraph.")
        
        # create a WorkFlow instance
        graph_copy = WorkFlowGraph(goal=graph.goal, graph=graph)
        graph_copy.reset_graph() # reset the status of all nodes to pending
        workflow = WorkFlow(llm=self.llm, graph=graph_copy, agent_manager=self.agent_manager, **kwargs)
        result = workflow.execute(inputs=inputs, extract_output=True, **kwargs)
        if result.status == "success":
            output = result.result
            if not isinstance(output, str):
                output = str(output)
        else:
            output = result.displayable_error or "Workflow Execution Failed"
        if return_trajectory:
            return output, workflow.environment.get()
        return output

    def _execute_action_graph(self, graph: ActionGraph, inputs: dict, **kwargs) -> dict:
        """
        Execute the action graph and return the output.

        Args:
            graph (ActionGraph): The action graph to execute
            inputs (dict): The inputs to the action graph
            **kwargs: Additional arguments for action graph execution

        Returns:
            dict: The output of the action graph
        """
        output: dict = graph.execute(**inputs, **kwargs)
        return output
    
    def _evaluate_single_example(self, graph: Union[WorkFlowGraph, ActionGraph], example: dict, benchmark: Benchmark, **kwargs) -> Optional[dict]:
        """
        Evaluate a single data example through the workflow and save the evaluation metrics to the evaluation records.

        Args:
            graph (WorkFlowGraph or ActionGraph): The workflow to execute
            example (dict): Single input data example
            **kwargs: Additional arguments for workflow execution

        Returns:
            Optional[dict]: Evaluation metrics for this example, None if failed
        """
        try:
            # collate the example   
            inputs: dict = self.collate_func(example)
            if not isinstance(inputs, dict):
                raise ValueError(f"The collate_func should return a dictionary. Got {type(inputs)}.")
            
            # execute the workflow or action graph
            if isinstance(graph, ActionGraph):
                output: dict = self._execute_action_graph(graph=graph, inputs=inputs, **kwargs)
            elif isinstance(graph, WorkFlowGraph):
                workflow_graph_outputs = self._execute_workflow_graph(graph=graph, inputs=inputs, return_trajectory=True, **kwargs)
                output: str = workflow_graph_outputs[0]
                trajectory: List[Message] = workflow_graph_outputs[1]
            else:
                raise ValueError(f"Invalid workflow type: {type(graph)}. Must be WorkFlowGraph or ActionGraph.")
            
            # postprocess the output
            output = self.output_postprocess_func(output)

            # get the label and evaluate the workflow
            label = benchmark.get_label(example)
            metrics = benchmark.evaluate(prediction=output, label=label)

            # save workflow output and metrics to the evaluation records 
            example_id = benchmark.get_id(example=example)
            self._evaluation_records[example_id] = {
                "prediction": output, 
                "label": label,
                "metrics": metrics
            }
            if isinstance(graph, WorkFlowGraph):
                self._evaluation_records[example_id]["trajectory"] = trajectory
        except Exception as e:
            logger.warning(f"Error evaluating example and set the metrics to None:\nExample: {example}\nError: {str(e)}")
            return None
        
        return metrics
        
    def _single_evaluate(self, graph: Union[WorkFlowGraph, ActionGraph], data: List[dict], benchmark: Benchmark, verbose: Optional[bool] = None, **kwargs) -> List[dict]:
        """
        Evaluate workflow on data using single thread.

        Args:
            graph (WorkFlowGraph or ActionGraph): The workflow to evaluate
            data (List[dict]): List of input data
            benchmark (Benchmark): The benchmark to evaluate the workflow on
            verbose (bool): Whether to show progress bar
            **kwargs: Additional arguments for workflow execution

        Returns:
            List[dict]: List of valid evaluation metrics
        """
        if not data:
            logger.warning("No data to evaluate. Return an empty list.")
            return []
        
        results = []
        if verbose:
            progress_bar = tqdm(data, desc="Evaluating workflow", total=len(data))
        for example in data:
            result = self._evaluate_single_example(graph, example, benchmark, **kwargs)
            # if result is not None:
            #     results.append(result)
            results.append(result) # can contain None values
            if verbose:
                progress_bar.update(1)
        if verbose:
            progress_bar.close()
        return results

    def _create_new_agent_manager(self) -> AgentManager:
        """Create a new agent manager with the same configuration but new locks"""
        if self.agent_manager is None:
            return None
        # Each worker gets its own copy of the agents so short-term memory
        # buffers do not leak across concurrent examples.
        copied_agents = [agent.model_copy(deep=True) for agent in self.agent_manager.agents]
        new_manager = AgentManager(agents=copied_agents, storage_handler=self.agent_manager.storage_handler)
        return new_manager

    def _get_thread_agent_manager(self) -> AgentManager:
        """Get or create thread-specific agent manager"""
        if self.agent_manager is None:
            return None
        thread_id = threading.get_ident()
        if thread_id not in self._thread_agent_managers:
            new_manager = self._create_new_agent_manager()
            self._thread_agent_managers[thread_id] = new_manager
        return self._thread_agent_managers[thread_id]

    def _evaluate_single_example_with_context(self, graph: Union[WorkFlowGraph, ActionGraph], example: dict, benchmark: Benchmark, **kwargs) -> Optional[dict]:
        """Wrapper that sets up thread-specific context before running evaluation"""
        thread_agent_manager = self._get_thread_agent_manager()
        if thread_agent_manager is None:
            return self._evaluate_single_example(graph, example, benchmark, **kwargs)
        
        # Store original agent manager
        original_agent_manager = self.agent_manager
        try:
            # Use thread-specific agent manager
            self.agent_manager = thread_agent_manager
            return self._evaluate_single_example(graph, example, benchmark, **kwargs)
        finally:
            # Restore original agent manager
            self.agent_manager = original_agent_manager

    def _parallel_evaluate(self, graph: Union[WorkFlowGraph, ActionGraph], data: List[dict], benchmark: Benchmark, verbose: Optional[bool] = None, **kwargs) -> List[dict]:
        if not data:
            logger.warning("No data to evaluate. Return an empty list.")
            return []
        
        results = [] 
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(
                    contextvars.copy_context().run,
                    self._evaluate_single_example_with_context,
                    graph, example, benchmark, **kwargs
                ): example
                for example in data
            }
            
            if verbose:
                progress_bar = tqdm(desc="Evaluating workflow", total=len(futures))
                
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)
                if verbose:
                    progress_bar.update(1)
                
        if verbose:
            progress_bar.close()
        return results

    def _calculate_average_score(self, scores: List[dict]) -> dict:
        """
        Calculate the average score from a list of scores.

        Args:
            scores (List[dict]): List of evaluation scores

        Returns:
            dict: Average metrics
        """
        if not scores:
            logger.warning("No scores found. Return an empty dictionary.")
            return {}
        num_total_items = len(scores)
        first_valid_score = None 
        for score in scores: 
            if score is not None:
                first_valid_score = score
                break
        if first_valid_score is None:
            logger.warning("No valid scores found. Return an empty dictionary.")
            return {}
        # return {k: sum(d[k] for d in scores) / len(scores) for k in scores[0]}
        return {k: sum(d[k] for d in scores if d is not None) / num_total_items for k in first_valid_score}


    def _evaluate_graph(self, graph: Union[WorkFlowGraph, ActionGraph], data: List[dict], benchmark: Benchmark, verbose: Optional[bool] = None, **kwargs) -> dict:
        """
        Evaluate the workflow on the data.

        Args:
            graph (WorkFlowGraph or ActionGraph): The workflow to evaluate
            data (List[dict]): List of input data to evaluate
            benchmark (Benchmark): The benchmark to evaluate the workflow on
            verbose (bool, optional): Whether to print the evaluation progress. If not provided, the `self.verbose` will be used.
            **kwargs: Additional arguments passed to workflow execution

        Returns:
            dict: The average metrics of the workflow evaluation
        """
        if not data:
            logger.warning("No data to evaluate. Return an empty dictionary.")
            return {}
        
        verbose = verbose if verbose is not None else self.verbose
        if self.num_workers > 1:
            results = self._parallel_evaluate(graph, data, benchmark, verbose, **kwargs)
        else:
            results = self._single_evaluate(graph, data, benchmark, verbose, **kwargs)
        
        return self._calculate_average_score(results)
    
    def get_example_evaluation_record(self, benchmark: Benchmark, example: Any) -> Optional[dict]:
        """
        Get the evaluation record for a given example.
        """
        example_id = benchmark.get_id(example=example)
        return self._evaluation_records.get(example_id, None)
    
    def get_evaluation_record_by_id(self, benchmark: Benchmark, example_id: str, eval_mode: str = "test") -> Optional[dict]:
        """
        Get the evaluation record for a given example id.
        """
        example = benchmark.get_example_by_id(example_id=example_id, mode=eval_mode)
        return self.get_example_evaluation_record(benchmark=benchmark, example=example)
    
    def get_all_evaluation_records(self) -> dict:
        """
        Get all the evaluation records.
        """
        return self._evaluation_records.copy()
    
    async def async_evaluate(
        self, 
        graph: Union[WorkFlowGraph, ActionGraph],
        benchmark: Benchmark, 
        eval_mode: str = "test", 
        indices: Optional[List[int]] = None, 
        sample_k: Optional[int] = None, 
        seed: Optional[int] = None, 
        verbose: Optional[bool] = None,
        **kwargs
    ) -> dict:
        """
        Asynchronously evaluate the performance of the workflow on the benchmark.

        Args:
            graph (WorkFlowGraph or ActionGraph): The workflow to evaluate.
            benchmark (Benchmark): The benchmark to evaluate the workflow on.
            eval_mode (str): which split of the benchmark to evaluate the workflow on. Choices: ["test", "dev", "train"].
            indices (List[int], optional): The indices of the data to evaluate the workflow on.
            sample_k (int, optional): The number of data to evaluate the workflow on. If provided, a random sample of size `sample_k` will be used.
            verbose (bool, optional): Whether to print the evaluation progress. If not provided, the `self.verbose` will be used.
        
        Returns:
            dict: The average metrics of the workflow evaluation.
        """
        # clear the evaluation records
        self._evaluation_records.clear()
        data = self._get_eval_data(benchmark=benchmark, eval_mode=eval_mode, indices=indices, sample_k=sample_k, seed=seed)
        
        if not data:
            logger.warning("No data to evaluate. Return an empty dictionary.")
            return {}
        
        verbose = verbose if verbose is not None else self.verbose
        
        # Create a semaphore to limit concurrent executions
        sem = asyncio.Semaphore(self.num_workers)

        async def process_with_semaphore(example):
            async with sem:
                try:
                    return await self._async_evaluate_single_example(
                        graph=graph, 
                        example=example, 
                        benchmark=benchmark, 
                        **kwargs
                    )
                except Exception as e:
                    logger.warning(f"Async evaluation failed for example with semaphore: {str(e)}")
                    return None
        
        # Create tasks for concurrent execution with semaphore
        tasks = [process_with_semaphore(example) for example in data]
        
        # Execute all tasks with progress bar if verbose
        if verbose:
            results = await tqdm_asyncio.gather(
                *tasks,
                desc=f"Evaluating {benchmark.name}",
                total=len(data)
            )
        else:
            results = await asyncio.gather(*tasks)
        
        return self._calculate_average_score(results)

    async def _async_evaluate_single_example(self, graph: Union[WorkFlowGraph, ActionGraph], example: dict, benchmark: Benchmark, **kwargs) -> Optional[dict]:
        """
        Asynchronously evaluate a single example. 
        """
        try:
            # collate the example   
            inputs: dict = self.collate_func(example)
            if not isinstance(inputs, dict):
                raise ValueError(f"The collate_func should return a dictionary. Got {type(inputs)}.")
            
            # execute the workflow or action graph
            if isinstance(graph, ActionGraph):
                output: dict = await self._async_execute_action_graph(graph=graph, inputs=inputs, **kwargs)
            elif isinstance(graph, WorkFlowGraph):
                workflow_graph_outputs = await self._async_execute_workflow_graph(graph=graph, inputs=inputs, return_trajectory=True, **kwargs)
                output: str = workflow_graph_outputs[0]
                trajectory: List[Message] = workflow_graph_outputs[1]
            else:
                raise ValueError(f"Invalid workflow type: {type(graph)}. Must be WorkFlowGraph or ActionGraph.")
            
            # postprocess the output
            output = self.output_postprocess_func(output)

            # get the label and evaluate the workflow
            label = benchmark.get_label(example)
            
            # Check if the benchmark has an async_evaluate method, otherwise use the synchronous one
            if hasattr(benchmark, 'async_evaluate') and callable(getattr(benchmark, 'async_evaluate')):
                metrics = await benchmark.async_evaluate(prediction=output, label=label)
            else:
                metrics = benchmark.evaluate(prediction=output, label=label)

            # save workflow output and metrics to the evaluation records 
            example_id = benchmark.get_id(example=example)
            self._evaluation_records[example_id] = {
                "prediction": output, 
                "label": label,
                "metrics": metrics
            }
            if isinstance(graph, WorkFlowGraph):
                self._evaluation_records[example_id]["trajectory"] = trajectory
        except Exception as e:
            logger.warning(f"Error evaluating example and set the metrics to None:\nExample: {example}\nError: {str(e)}")
            return None
        return metrics
    
    async def _async_execute_action_graph(self, graph: ActionGraph, inputs: dict, **kwargs) -> dict:
        """
        Asynchronously execute the action graph.
        """
        return await graph.async_execute(**inputs, **kwargs) 
    
    async def _async_execute_workflow_graph(self, graph: WorkFlowGraph, inputs: dict, return_trajectory: bool = False, **kwargs) -> Union[str, Tuple[str, List[Message]]]:
        """
        Asynchronously execute the workflow graph.
        """
        if self.agent_manager is None:
            raise ValueError("`agent_manager` is not provided. Please provide an agent manager when evaluating a WorkFlowGraph.")
        
        # create a WorkFlow instance
        graph_copy = WorkFlowGraph(goal=graph.goal, graph=graph)
        graph_copy.reset_graph() # reset the status of all nodes to pending
        
        # Make a local copy of agent_manager for thread-safety in async context.
        # Agents are deep-copied so each coroutine gets its own short-term
        # memory instead of sharing the parent manager's buffers.
        local_agent_manager = AgentManager(
            agents=[agent.model_copy(deep=True) for agent in self.agent_manager.agents],
            storage_handler=self.agent_manager.storage_handler
        )
        
        workflow = WorkFlow(
            llm=self.llm, 
            graph=graph_copy, 
            agent_manager=local_agent_manager, 
            **kwargs
        )
        
        result = await workflow.async_execute(inputs=inputs, extract_output=True, **kwargs)
        if result.status == "success":
            output = result.result
            if not isinstance(output, str):
                output = str(output)
        else:
            output = result.displayable_error or "Workflow Execution Failed"
        if return_trajectory:
            return output, workflow.environment.get()
        return output
