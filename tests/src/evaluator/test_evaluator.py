import threading
import unittest
from unittest.mock import Mock, patch
from evoagentx.evaluators.evaluator import Evaluator, _agents_with_fresh_short_term_memory
from evoagentx.benchmark.benchmark import Benchmark
from evoagentx.workflow.workflow_graph import WorkFlowGraph
from evoagentx.workflow.action_graph import ActionGraph
from evoagentx.agents.agent import Agent
from evoagentx.agents.agent_manager import AgentManager
from evoagentx.core.message import Message, MessageType
from evoagentx.models.base_model import BaseLLM

class TestEvaluator(unittest.TestCase):

    def setUp(self):
        # Create mock objects
        self.benchmark = Mock(spec=Benchmark)
        self.benchmark.get_test_data.return_value = [
            {"id": "1", "input": "test1"},
            {"id": "2", "input": "test2"}
        ]
        self.benchmark.get_id.side_effect = lambda example: example["id"]
        self.benchmark.get_label.return_value = "expected"
        self.benchmark.evaluate.return_value = {"accuracy": 1.0}

        # Mock LLM
        self.llm = Mock(spec=BaseLLM)
        
        # Mock agent manager
        self.agent_manager = Mock(spec=AgentManager)

        # Create mock graphs
        self.workflow_graph = Mock(spec=WorkFlowGraph)
        self.action_graph = Mock(spec=ActionGraph)
        self.action_graph.execute.return_value = {"output": "prediction"}

        # Create evaluator instance
        self.evaluator = Evaluator(
            llm=self.llm,
            num_workers=1,
            agent_manager=self.agent_manager
        )
    
    @patch.object(Evaluator, '_execute_workflow_graph')
    def test_single_thread_evaluation_workflow_graph(self, mock_execute):
        # Set up mock return value
        mock_execute.return_value = ("workflow_graph_prediction", ["trajectory_data"])
        
        # Test evaluation with single thread using WorkFlowGraph
        results = self.evaluator.evaluate(
            graph=self.workflow_graph,
            benchmark=self.benchmark,
            eval_mode="test"
        )

        # Verify results
        self.assertEqual(mock_execute.call_count, 2) 
        self.assertEqual(results, {"accuracy": 1.0})
        self.assertEqual(len(self.evaluator.get_all_evaluation_records()), 2)

    def test_single_thread_evaluation_action_graph(self):
        # Test evaluation with single thread using ActionGraph
        results = self.evaluator.evaluate(
            graph=self.action_graph,
            benchmark=self.benchmark,
            eval_mode="test"
        )

        # Verify results
        self.assertEqual(results, {"accuracy": 1.0})
        self.assertEqual(len(self.evaluator.get_all_evaluation_records()), 2)

    # @patch.object(Evaluator, '_execute_workflow_graph')
    # def test_multi_thread_evaluation(self, mock_execute):
    #     # Set up mock return value for workflow graph
    #     mock_execute.return_value = ("workflow_graph_prediction", ["trajectory_data"])
        
    #     # Test evaluation with multiple threads
    #     evaluator = Evaluator(
    #         llm=self.llm,
    #         num_workers=2,
    #         agent_manager=self.agent_manager,
    #         verbose=True
    #     )

    #     # Test workflow graph
    #     results_workflow = evaluator.evaluate(
    #         graph=self.workflow_graph,
    #         benchmark=self.benchmark,
    #         eval_mode="test"
    #     )
    #     self.assertEqual(results_workflow, {"accuracy": 1.0})
    #     self.assertEqual(len(evaluator.get_all_evaluation_records()), 2)
    #     self.assertEqual(mock_execute.call_count, 2)

    #     # Clear evaluation records for next test
    #     evaluator._evaluation_records.clear()

    #     # Test action graph
    #     results_action = evaluator.evaluate(
    #         graph=self.action_graph,
    #         benchmark=self.benchmark,
    #         eval_mode="test"
    #     )
    #     self.assertEqual(results_action, {"accuracy": 1.0})
    #     self.assertEqual(len(evaluator.get_all_evaluation_records()), 2)

    #     # Verify that records contain the expected data
    #     records = evaluator.get_all_evaluation_records()
    #     for record in records.values():
    #         self.assertIn("prediction", record)
    #         self.assertIn("label", record)
    #         self.assertIn("metrics", record)
    #         self.assertEqual(record["label"], "expected")
    #         self.assertEqual(record["metrics"], {"accuracy": 1.0})

    def test_evaluation_with_custom_collate(self):
        # Test evaluation with custom collate function
        def collate_func(x):
            return {"processed_" + k: v for k, v in x.items()}

        evaluator = Evaluator(
            llm=self.llm,
            num_workers=1,
            collate_func=collate_func
        )
        evaluator.evaluate(
            graph=self.action_graph,
            benchmark=self.benchmark,
            eval_mode="test"
        )

        # Get the first call's arguments
        call_args = self.action_graph.execute.call_args_list[0][1]
        self.assertTrue(all(k.startswith("processed_") for k in call_args.keys()))
    
    def test_evaluation_with_output_postprocess(self):
        # Test evaluation with output postprocess function
        def postprocess_func(x):
            return x["output"].upper()

        evaluator = Evaluator(
            llm=self.llm,
            num_workers=1,
            output_postprocess_func=postprocess_func
        )
        
        evaluator.evaluate(
            graph=self.action_graph,
            benchmark=self.benchmark,
            eval_mode="test"
        )

        records = evaluator.get_all_evaluation_records()
        for record in records.values():
            self.assertEqual(record["prediction"], "PREDICTION")
    
    def test_get_example_evaluation_record(self):
        # Test get example evaluation record
        self.evaluator.evaluate(
            graph=self.action_graph,
            benchmark=self.benchmark,
            eval_mode="test"
        )

        example = {"id": "1", "input": "test1"}
        record = self.evaluator.get_example_evaluation_record(self.benchmark, example)

        self.assertIsNotNone(record)
        self.assertEqual(record["prediction"], {"output": "prediction"})
        self.assertEqual(record["label"], "expected")
        self.assertEqual(record["metrics"], {"accuracy": 1.0})
    
    def test_invalid_eval_mode(self):
        with self.assertRaises(AssertionError):
            self.evaluator.evaluate(
                graph=self.action_graph,
                benchmark=self.benchmark,
                eval_mode="invalid"
            )

    def test_empty_data_evaluation(self):
        # Test empty data evaluation
        self.benchmark.get_test_data.return_value = []
        results = self.evaluator.evaluate(
            graph=self.action_graph,
            benchmark=self.benchmark,
            eval_mode="test"
        )
        
        self.assertEqual(results, {})
        self.assertEqual(len(self.evaluator.get_all_evaluation_records()), 0)

class TestEvaluatorConcurrentAgentMemoryIsolation(unittest.TestCase):

    def test_agents_with_fresh_short_term_memory_clones_memory_only(self):
        # Unit-tests the helper shared by _create_new_agent_manager (thread
        # path) and _async_execute_workflow_graph (async path): each clone
        # must get its own ShortTermMemory while everything else is shared.
        agent = Agent(name="worker", description="test agent", is_human=True)

        clone_one, clone_two = _agents_with_fresh_short_term_memory([agent, agent])

        self.assertIsNot(clone_one.short_term_memory, agent.short_term_memory)
        self.assertIsNot(clone_two.short_term_memory, clone_one.short_term_memory)
        self.assertEqual(clone_one.name, agent.name)
        self.assertIs(clone_one.llm, agent.llm)

    def test_create_new_agent_manager_isolates_short_term_memory_across_threads(self):
        # Reproduces the reported bug: two "independent" per-thread
        # AgentManagers created by _create_new_agent_manager must not let one
        # example's messages leak into another's short_term_memory.
        shared_agent = Agent(name="worker", description="test agent", is_human=True)
        agent_manager = AgentManager(agents=[shared_agent])
        evaluator = Evaluator(llm=Mock(spec=BaseLLM), num_workers=2, agent_manager=agent_manager)

        manager_a = evaluator._create_new_agent_manager()
        manager_b = evaluator._create_new_agent_manager()
        self.assertIsNot(manager_a.agents[0], manager_b.agents[0])

        barrier = threading.Barrier(2)
        results = {}

        def run_example(manager, example_id):
            agent = manager.agents[0]
            barrier.wait()
            for i in range(5):
                agent.short_term_memory.add_message(
                    Message(content=f"example-{example_id}-msg-{i}", msg_type=MessageType.INPUT)
                )
            results[example_id] = [m.content for m in agent.short_term_memory.get(n=5)]

        threads = [
            threading.Thread(target=run_example, args=(manager_a, "A")),
            threading.Thread(target=run_example, args=(manager_b, "B")),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertTrue(all(c.startswith("example-A-") for c in results["A"]))
        self.assertTrue(all(c.startswith("example-B-") for c in results["B"]))


if __name__ == '__main__':
    unittest.main()