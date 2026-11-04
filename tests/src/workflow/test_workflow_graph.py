import inspect
import unittest
from unittest.mock import patch

import pytest

from evoagentx.core.base_config import Parameter
from evoagentx.workflow.workflow_graph import (
    WorkFlowEdge,
    WorkFlowGraph,
    WorkFlowNode,
    WorkFlowNodeState,
)


class TestWorkFlowGraph(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        # Create nodes for testing
        self.task1 = WorkFlowNode(
            name="Task1",
            description="First task",
            inputs=[Parameter(name="input1", type="string", description="Input 1")],
            outputs=[Parameter(name="output1", type="string", description="Output 1")],
            agents=["TestAgent"],
            status=WorkFlowNodeState.PENDING
        )
        
        self.task2 = WorkFlowNode(
            name="Task2",
            description="Second task",
            inputs=[Parameter(name="output1", type="string", description="Output from Task1")],
            outputs=[Parameter(name="output2", type="string", description="Output 2")],
            agents=["TestAgent"],
            status=WorkFlowNodeState.PENDING
        )
        
        self.task3 = WorkFlowNode(
            name="Task3",
            description="Third task",
            inputs=[Parameter(name="output2", type="string", description="Output from Task2")],
            outputs=[Parameter(name="output3", type="string", description="Output from Task3")],
            agents=["TestAgent"],
            status=WorkFlowNodeState.PENDING
        )

        self.task4 = WorkFlowNode(
            name="Task4",
            description="Fourth task (join)",
            inputs=[
                Parameter(name="output2", type="string", description="Output from Task2"),
                Parameter(name="output3", type="string", description="Output from Task3")
            ],
            outputs=[Parameter(name="output4", type="string", description="Output from Task4")],
            agents=["TestAgent"],
            status=WorkFlowNodeState.PENDING
        )

        self.task5 = WorkFlowNode(
            name="Task5",
            description="Fifth task (first node in loop)",
            inputs=[
                Parameter(name="output1", type="string", description="Output from Task1"),
                Parameter(name="output6", type="string", description="Output from Task6", required=False)
            ],
            outputs=[Parameter(name="output5", type="string", description="Output from Task5")],
            agents=["TestAgent"],
            status=WorkFlowNodeState.PENDING
        )

        self.task6 = WorkFlowNode(
            name="Task6",
            description="Sixth task (second node in loop)",
            inputs=[
                Parameter(name="output5", type="string", description="Output from Task5")
            ],
            outputs=[Parameter(name="output6", type="string", description="Output from Task6")],
            agents=["TestAgent"],
            status=WorkFlowNodeState.PENDING
        )
        
        # Create a simple linear workflow: Task1 -> Task2 -> Task3
        self.linear_graph = WorkFlowGraph(
            goal="Simple Linear Workflow",
            nodes=[self.task1, self.task2, self.task3],
            edges=[
                WorkFlowEdge(source="Task1", target="Task2"),
                WorkFlowEdge(source="Task2", target="Task3")
            ]
        )
        
        # Create a fork-join workflow
        #     Task1
        #    /     
        # Task2 -- Task3
        #    \     /
        #    Task4
        self.fork_join_graph = WorkFlowGraph(
            goal="Fork-Join Workflow",
            nodes=[self.task1, self.task2, self.task3, self.task4],
            edges=[
                WorkFlowEdge(source="Task1", target="Task2"),
                WorkFlowEdge(source="Task2", target="Task3"),
                WorkFlowEdge(source="Task2", target="Task4"),
                WorkFlowEdge(source="Task3", target="Task4")
            ]
        )
        
        # Create a workflow with a cycle
        # Task1 -> Task5 -> Task6 -> Task5
        self.cycle_graph = WorkFlowGraph(
            goal="Workflow with Cycle",
            nodes=[self.task1, self.task5, self.task6]
        )
    
    def test_graph_initialization(self):
        """Test that graph is correctly initialized with nodes and edges."""
        # Check linear graph
        self.assertEqual(3, len(self.linear_graph.nodes))
        self.assertEqual(2, len(self.linear_graph.edges))
        
        # Verify node order
        self.assertEqual("Task1", self.linear_graph.nodes[0].name)
        self.assertEqual("Task2", self.linear_graph.nodes[1].name)
        self.assertEqual("Task3", self.linear_graph.nodes[2].name)
        
        # Verify edge connections
        edge_pairs = [(edge.source, edge.target) for edge in self.linear_graph.edges]
        self.assertIn(("Task1", "Task2"), edge_pairs)
        self.assertIn(("Task2", "Task3"), edge_pairs)
    
    def test_find_initial_nodes(self):
        """Test finding initial nodes in a workflow."""
        # In the linear graph, Task1 is the only initial node
        initial_nodes = self.linear_graph.find_initial_nodes()
        self.assertEqual(1, len(initial_nodes))
        self.assertEqual("Task1", initial_nodes[0])
        
        # In the fork-join graph, Task1 is the only initial node
        initial_nodes = self.fork_join_graph.find_initial_nodes()
        self.assertEqual(1, len(initial_nodes))
        self.assertEqual("Task1", initial_nodes[0])
        
        # In the cycle graph, Task1 is not an initial node because of the cycle
        # All nodes have incoming edges in a cycle
        initial_nodes = self.cycle_graph.find_initial_nodes()
        self.assertEqual(1, len(initial_nodes))
    
    def test_find_end_nodes(self):
        """Test finding end nodes in a workflow."""
        # In the linear graph, Task3 is the only end node
        end_nodes = self.linear_graph.find_end_nodes()
        self.assertEqual(1, len(end_nodes))
        self.assertEqual("Task3", end_nodes[0])
        
        # In the fork-join graph, Task4 is the only end node
        end_nodes = self.fork_join_graph.find_end_nodes()
        self.assertEqual(1, len(end_nodes))
        self.assertEqual("Task4", end_nodes[0])
        
        # In the cycle graph, there are no end nodes because of the cycle
        # All nodes have outgoing edges in a cycle
        end_nodes = self.cycle_graph.find_end_nodes()
        self.assertEqual(0, len(end_nodes))
    
    def test_next_execution(self):
        """Test the 'next' method to determine the next executable tasks."""
        # In the linear graph, initially Task1 is the only executable task
        next_tasks = self.linear_graph.next()
        self.assertEqual(1, len(next_tasks))
        self.assertEqual("Task1", next_tasks[0].name)
        
        # After completing Task1, Task2 should be the next executable task
        self.linear_graph.set_node_status("Task1", WorkFlowNodeState.COMPLETED)
        next_tasks = self.linear_graph.next()
        self.assertEqual(1, len(next_tasks))
        self.assertEqual("Task2", next_tasks[0].name)
        
        # After completing Task2, Task3 should be the next executable task
        self.linear_graph.set_node_status("Task2", WorkFlowNodeState.COMPLETED)
        next_tasks = self.linear_graph.next()
        self.assertEqual(1, len(next_tasks))
        self.assertEqual("Task3", next_tasks[0].name)
        
        # After completing Task3, there should be no more executable tasks
        self.linear_graph.set_node_status("Task3", WorkFlowNodeState.COMPLETED)
        next_tasks = self.linear_graph.next()
        self.assertEqual(0, len(next_tasks))
    
    def test_fork_join_execution(self):
        """Test execution in a fork-join workflow."""
        # Initially Task1 is the only executable task
        next_tasks = self.fork_join_graph.next()
        self.assertEqual(1, len(next_tasks))
        self.assertEqual("Task1", next_tasks[0].name)
        
        # After completing Task1, both Task2 and Task3 should be executable
        self.fork_join_graph.set_node_status("Task1", WorkFlowNodeState.COMPLETED)
        next_tasks = self.fork_join_graph.next()
        self.assertEqual(1, len(next_tasks))
        self.assertEqual("Task2", next_tasks[0].name)
        
        # After completing Task2 but not Task3, Task4 should not be executable yet
        self.fork_join_graph.set_node_status("Task2", WorkFlowNodeState.COMPLETED)
        next_tasks = self.fork_join_graph.next()
        self.assertEqual(1, len(next_tasks))
        self.assertEqual("Task3", next_tasks[0].name)
        
        # After completing Task3 as well, Task4 should be executable
        self.fork_join_graph.set_node_status("Task3", WorkFlowNodeState.COMPLETED)
        next_tasks = self.fork_join_graph.next()
        self.assertEqual(1, len(next_tasks))
        self.assertEqual("Task4", next_tasks[0].name)
        
        # After completing Task4, there should be no more executable tasks
        self.fork_join_graph.set_node_status("Task4", WorkFlowNodeState.COMPLETED)
        next_tasks = self.fork_join_graph.next()
        self.assertEqual(0, len(next_tasks))

    def test_workflow_completes_when_workflow_output_nodes_complete(self):
        """Nodes that do not produce workflow outputs should not keep the workflow open."""
        node_a = WorkFlowNode(
            name="A",
            description="initial task",
            inputs=[Parameter(name="input1", type="string", description="workflow input")],
            outputs=[
                Parameter(name="target_input", type="string", description="feeds target output"),
                Parameter(name="side_input", type="string", description="feeds side branch"),
            ],
            agents=["TestAgent"],
        )
        node_b = WorkFlowNode(
            name="B",
            description="workflow output task",
            inputs=[Parameter(name="target_input", type="string", description="from A")],
            outputs=[Parameter(name="target_output", type="string", description="workflow output")],
            agents=["TestAgent"],
        )
        node_c = WorkFlowNode(
            name="C",
            description="side branch task",
            inputs=[Parameter(name="side_input", type="string", description="from A")],
            outputs=[Parameter(name="side_output", type="string", description="not a workflow output")],
            agents=["TestAgent"],
        )
        graph = WorkFlowGraph(
            goal="Complete on workflow output",
            nodes=[node_a, node_b, node_c],
            workflow_inputs=[Parameter(name="input1", type="string", description="workflow input")],
            workflow_outputs=[Parameter(name="target_output", type="string", description="workflow output")],
        )

        graph.set_node_status("A", WorkFlowNodeState.COMPLETED)
        self.assertFalse(graph.is_complete)

        graph.set_node_status("B", WorkFlowNodeState.COMPLETED)
        self.assertTrue(graph.is_complete)
        self.assertEqual([], graph.next())
    
    def test_control_edge_not_executed_in_parallel(self):
        """An explicit control edge (A -> B with no shared data) must be respected even
        when B's required inputs are all workflow inputs and therefore B is data-initial."""
        node_a = WorkFlowNode(
            name="A",
            description="control source",
            inputs=[Parameter(name="input1", type="string", description="workflow input")],
            outputs=[Parameter(name="outputA", type="string", description="output A")],
            agents=["TestAgent"],
        )
        node_b = WorkFlowNode(
            name="B",
            description="control target",
            inputs=[Parameter(name="input1", type="string", description="workflow input")],
            outputs=[Parameter(name="outputB", type="string", description="output B")],
            agents=["TestAgent"],
        )
        graph = WorkFlowGraph(
            goal="Control Edge Workflow",
            nodes=[node_a, node_b],
            edges=[WorkFlowEdge(source="A", target="B")],
            workflow_inputs=[Parameter(name="input1", type="string", description="workflow input")],
            workflow_outputs=[
                Parameter(name="outputA", type="string", description="output A"),
                Parameter(name="outputB", type="string", description="output B"),
            ],
        )

        # Both A and B are data-initial, but only A may run first.
        self.assertEqual({"A", "B"}, set(graph.find_initial_nodes()))
        next_tasks = graph.next()
        self.assertEqual(1, len(next_tasks))
        self.assertEqual("A", next_tasks[0].name)

        graph.set_node_status("A", WorkFlowNodeState.COMPLETED)
        next_tasks = graph.next()
        self.assertEqual(1, len(next_tasks))
        self.assertEqual("B", next_tasks[0].name)

    def test_optional_input_edge_respects_dependency(self):
        """An inferred edge feeding an optional input must be respected even though the
        target's required inputs are all workflow inputs (so it is data-initial)."""
        node_a = WorkFlowNode(
            name="A",
            description="optional source",
            inputs=[Parameter(name="input1", type="string", description="workflow input")],
            outputs=[Parameter(name="outA", type="string", description="output A")],
            agents=["TestAgent"],
        )
        node_b = WorkFlowNode(
            name="B",
            description="optional target",
            inputs=[
                Parameter(name="input1", type="string", description="workflow input"),
                Parameter(name="outA", type="string", description="optional from A", required=False),
            ],
            outputs=[Parameter(name="outputB", type="string", description="output B")],
            agents=["TestAgent"],
        )
        graph = WorkFlowGraph(
            goal="Optional Input Workflow",
            nodes=[node_a, node_b],
            workflow_inputs=[Parameter(name="input1", type="string", description="workflow input")],
            workflow_outputs=[
                Parameter(name="outA", type="string", description="output A"),
                Parameter(name="outputB", type="string", description="output B"),
            ],
        )

        # The A -> B edge is inferred from the shared `outA` name (B's optional input).
        edge_pairs = [(edge.source, edge.target) for edge in graph.edges]
        self.assertIn(("A", "B"), edge_pairs)

        # B is data-initial (only required input is the workflow input) but must wait for A.
        self.assertEqual({"A", "B"}, set(graph.find_initial_nodes()))
        next_tasks = graph.next()
        self.assertEqual(1, len(next_tasks))
        self.assertEqual("A", next_tasks[0].name)

        graph.set_node_status("A", WorkFlowNodeState.COMPLETED)
        next_tasks = graph.next()
        self.assertEqual(1, len(next_tasks))
        self.assertEqual("B", next_tasks[0].name)

    def test_cycle_detection(self):
        """Test cycle detection in a workflow."""
        # The cycle graph should identify a loop
        loops = self.cycle_graph._find_all_loops()
        self.assertTrue(loops)  # Should contain at least one loop
        self.assertTrue(self.cycle_graph.is_loop_start("Task5"))
        self.assertTrue(self.cycle_graph.is_loop_end("Task6"))
    
    def test_node_status_management(self):
        """Test node status management."""
        # Check initial status
        self.assertEqual(WorkFlowNodeState.PENDING, self.linear_graph.get_node_status("Task1"))
        
        # Set status to RUNNING
        self.linear_graph.set_node_status("Task1", WorkFlowNodeState.RUNNING)
        self.assertEqual(WorkFlowNodeState.RUNNING, self.linear_graph.get_node_status("Task1"))
        self.assertTrue(self.linear_graph.running("Task1"))
        
        # Set status to COMPLETED
        self.linear_graph.set_node_status("Task1", WorkFlowNodeState.COMPLETED)
        self.assertEqual(WorkFlowNodeState.COMPLETED, self.linear_graph.get_node_status("Task1"))
        self.assertTrue(self.linear_graph.completed("Task1"))
        
        # Set status to FAILED
        self.linear_graph.set_node_status("Task1", WorkFlowNodeState.FAILED)
        self.assertEqual(WorkFlowNodeState.FAILED, self.linear_graph.get_node_status("Task1"))
        self.assertTrue(self.linear_graph.failed("Task1"))
    
    def test_graph_reset(self):
        """Test resetting the graph to initial state."""
        # Set all nodes to COMPLETED
        for node in self.linear_graph.nodes:
            self.linear_graph.set_node_status(node.name, WorkFlowNodeState.COMPLETED)
        
        # Verify all nodes are COMPLETED
        for node in self.linear_graph.nodes:
            self.assertEqual(WorkFlowNodeState.COMPLETED, node.status)
        
        # Reset the graph
        self.linear_graph.reset_graph()
        
        # Verify all nodes are reset to PENDING
        for node in self.linear_graph.nodes:
            self.assertEqual(WorkFlowNodeState.PENDING, node.status)
    
    def test_graph_dependency_checking(self):
        """Test checking dependencies between nodes."""
        # In the linear graph, Task2 depends on Task1
        self.assertFalse(self.linear_graph.are_dependencies_complete("Task2"))
        
        # After completing Task1, Task2's dependencies should be satisfied
        self.linear_graph.set_node_status("Task1", WorkFlowNodeState.COMPLETED)
        self.assertTrue(self.linear_graph.are_dependencies_complete("Task2"))
        
        # In the fork-join graph, Task4 depends on both Task2 and Task3
        self.assertFalse(self.fork_join_graph.are_dependencies_complete("Task4"))
        
        # After completing only Task2, Task4's dependencies are still not satisfied
        self.fork_join_graph.set_node_status("Task1", WorkFlowNodeState.COMPLETED)
        self.fork_join_graph.set_node_status("Task2", WorkFlowNodeState.COMPLETED)
        self.assertFalse(self.fork_join_graph.are_dependencies_complete("Task4"))
        
        # After completing Task3 as well, Task4's dependencies should be satisfied
        self.fork_join_graph.set_node_status("Task3", WorkFlowNodeState.COMPLETED)
        self.assertTrue(self.fork_join_graph.are_dependencies_complete("Task4"))

    def test_workflow_io_duplicates(self):
        """Test that workflow inputs and outputs cannot have duplicate names."""
        with pytest.raises(ValueError, match=r"Workflow inputs and outputs share the following name\(s\), which is not allowed: \['shared'\]"):
            WorkFlowGraph(
                goal="Duplicate IO",
                nodes=[self.task1],
                workflow_inputs=[Parameter(name="shared", type="string", description="desc")],
                workflow_outputs=[Parameter(name="shared", type="string", description="desc")]
            )

    def test_node_io_duplicates(self):
        """Test that node inputs and outputs cannot have internal duplicates or overlap."""
        # Duplicate input name
        node_dup_in = WorkFlowNode(
            name="DupIn",
            description="test",
            inputs=[
                Parameter(name="in1", type="string", description="desc"),
                Parameter(name="in1", type="string", description="desc")
            ],
            outputs=[Parameter(name="out1", type="string", description="desc")],
            agents=["TestAgent"]
        )
        with pytest.raises(ValueError, match="Node 'DupIn' has duplicate input name: 'in1'"):
            WorkFlowGraph(
                goal="test", 
                nodes=[node_dup_in],
                workflow_inputs=[Parameter(name="workflow_in", type="string", description="desc")],
                workflow_outputs=[Parameter(name="workflow_out", type="string", description="desc")]
            )

        # Duplicate output name
        node_dup_out = WorkFlowNode(
            name="DupOut",
            description="test",
            inputs=[Parameter(name="in1", type="string", description="desc")],
            outputs=[
                Parameter(name="out1", type="string", description="desc"),
                Parameter(name="out1", type="string", description="desc")
            ],
            agents=["TestAgent"]
        )
        with pytest.raises(ValueError, match="Node 'DupOut' has duplicate output name: 'out1'"):
            WorkFlowGraph(
                goal="test", 
                nodes=[node_dup_out],
                workflow_inputs=[Parameter(name="workflow_in", type="string", description="desc")],
                workflow_outputs=[Parameter(name="workflow_out", type="string", description="desc")]
            )

        # Overlap between inputs and outputs
        node_overlap = WorkFlowNode(
            name="Overlap",
            description="test",
            inputs=[Parameter(name="shared", type="string", description="desc")],
            outputs=[Parameter(name="shared", type="string", description="desc")],
            agents=["TestAgent"]
        )
        with pytest.raises(ValueError, match=r"Node 'Overlap' inputs and outputs share the following name\(s\), which is not allowed: \['shared'\]"):
            WorkFlowGraph(
                goal="test", 
                nodes=[node_overlap],
                workflow_inputs=[Parameter(name="workflow_in", type="string", description="desc")],
                workflow_outputs=[Parameter(name="workflow_out", type="string", description="desc")]
            )

    def test_node_output_uniqueness(self):
        """Test that each output name must be unique across all nodes."""
        node1 = WorkFlowNode(
            name="Node1",
            description="test",
            inputs=[Parameter(name="in1", type="string", description="desc")],
            outputs=[Parameter(name="out_shared", type="string", description="desc")],
            agents=["TestAgent"]
        )
        node2 = WorkFlowNode(
            name="Node2",
            description="test",
            inputs=[Parameter(name="in2", type="string", description="desc")],
            outputs=[Parameter(name="out_shared", type="string", description="desc")],
            agents=["TestAgent"]
        )
        expected_msg = (
            r"Each node output name must be unique across all nodes\. "
            r"Found conflicts:\n'out_shared' produced by \['Node1', 'Node2'\]"
        )
        with pytest.raises(ValueError, match=expected_msg):
            WorkFlowGraph(
                goal="test", 
                nodes=[node1, node2],
                workflow_inputs=[Parameter(name="workflow_in", type="string", description="desc")],
                workflow_outputs=[Parameter(name="workflow_out", type="string", description="desc")]
            )

    def test_mismatched_params_raise(self):
        """A node parameter that mismatches the workflow parameter (type/required) must raise."""
        mismatched_node = WorkFlowNode(
            name="MismatchedNode",
            description="test",
            inputs=[Parameter(name="input", type="number", description="desc", required=False)], # Should be string, required=True
            outputs=[Parameter(name="output", type="boolean", description="desc")],
            agents=["TestAgent"]
        )

        with pytest.raises(ValueError):
            WorkFlowGraph(
                goal="Test mismatch",
                nodes=[mismatched_node],
                workflow_inputs=[Parameter(name="input", type="string", description="desc", required=True)],
                workflow_outputs=[Parameter(name="output", type="string", description="desc")],
            )

    def test_json_schema_cannot_be_dropped_downstream(self):
        """A node/agent cannot omit a schema declared by the workflow output."""
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        }
        graph_dict = {
            "goal": "Schema contract",
            "nodes": [{
                "name": "SchemaNode",
                "description": "test",
                "inputs": [{"name": "input", "type": "string", "description": "desc"}],
                "outputs": [{"name": "output", "type": "object", "description": "desc"}],
                "agents": [{
                    "name": "SchemaAgent",
                    "description": "test",
                    "inputs": [{"name": "input", "type": "string", "description": "desc"}],
                    "outputs": [{"name": "output", "type": "object", "description": "desc"}],
                }],
            }],
            "workflow_inputs": [{"name": "input", "type": "string", "description": "desc"}],
            "workflow_outputs": [{
                "name": "output",
                "type": "object",
                "description": "desc",
                "json_schema": schema,
            }],
        }

        with pytest.raises(Exception, match="json_schema"):
            WorkFlowGraph.from_dict(graph_dict)

    def test_object_params_without_any_json_schema_are_allowed(self):
        """object params remain valid when no workflow/node/agent layer declares a schema."""
        graph_dict = {
            "goal": "Schema optional",
            "nodes": [{
                "name": "SchemaNode",
                "description": "test",
                "inputs": [{"name": "input", "type": "string", "description": "desc"}],
                "outputs": [{"name": "output", "type": "object", "description": "desc"}],
                "agents": [{
                    "name": "SchemaAgent",
                    "description": "test",
                    "inputs": [{"name": "input", "type": "string", "description": "desc"}],
                    "outputs": [{"name": "output", "type": "object", "description": "desc"}],
                }],
            }],
            "workflow_inputs": [{"name": "input", "type": "string", "description": "desc"}],
            "workflow_outputs": [{"name": "output", "type": "object", "description": "desc"}],
        }

        graph = WorkFlowGraph.from_dict(graph_dict)
        self.assertIsNone(graph.get_node("SchemaNode").outputs[0].json_schema)
        self.assertNotIn("json_schema", graph.get_node("SchemaNode").agents[0]["outputs"][0])

    def test_from_dict_preserves_explicit_edges(self):
        """Explicit control edges (no shared input/output) must survive a from_dict round-trip;
        they cannot be re-inferred from data flow."""

        def make_agent(name, in_name, out_name):
            return {
                "name": name,
                "description": "test",
                "inputs": [{"name": in_name, "type": "string", "description": "desc"}],
                "outputs": [{"name": out_name, "type": "string", "description": "desc"}],
                "prompt_template": {"class_name": "ChatTemplate", "instruction": "instruction"},
            }

        graph_dict = {
            "goal": "Test Explicit Edges",
            "nodes": [
                {
                    "name": "A",
                    "description": "control source",
                    "inputs": [{"name": "wf_in", "type": "string", "description": "desc"}],
                    "outputs": [{"name": "outA", "type": "string", "description": "desc"}],
                    "agents": [make_agent("AgentA", "wf_in", "outA")],
                },
                {
                    "name": "B",
                    "description": "control target",
                    "inputs": [{"name": "wf_in", "type": "string", "description": "desc"}],
                    "outputs": [{"name": "outB", "type": "string", "description": "desc"}],
                    "agents": [make_agent("AgentB", "wf_in", "outB")],
                },
            ],
            # A -> B shares no input/output, so it cannot be inferred from data flow.
            "edges": [{"source": "A", "target": "B"}],
            "workflow_inputs": [{"name": "wf_in", "type": "string", "description": "desc"}],
            "workflow_outputs": [
                {"name": "outA", "type": "string", "description": "desc"},
                {"name": "outB", "type": "string", "description": "desc"},
            ],
        }

        graph = WorkFlowGraph.from_dict(graph_dict)

        edge_pairs = {(edge.source, edge.target) for edge in graph.edges}
        self.assertIn(("A", "B"), edge_pairs)

        # The restored control edge must still gate execution: B waits for A.
        next_tasks = graph.next()
        self.assertEqual(["A"], [task.name for task in next_tasks])

    def test_explicit_edge_priority_overrides_inferred_edge_priority(self):
        """When an explicit edge matches an inferred data-flow edge, preserve the explicit metadata."""
        graph_dict = {
            "goal": "Test Explicit Edge Priority",
            "nodes": [
                {
                    "name": "A",
                    "description": "source",
                    "inputs": [{"name": "wf_in", "type": "string", "description": "desc"}],
                    "outputs": [{"name": "outA", "type": "string", "description": "desc"}],
                    "agents": ["AgentA"],
                },
                {
                    "name": "B",
                    "description": "target",
                    "inputs": [{"name": "outA", "type": "string", "description": "desc"}],
                    "outputs": [{"name": "outB", "type": "string", "description": "desc"}],
                    "agents": ["AgentB"],
                },
            ],
            # A -> B is also inferred from outA, but the explicit priority must win.
            "edges": [{"source": "A", "target": "B", "priority": 7}],
            "workflow_inputs": [{"name": "wf_in", "type": "string", "description": "desc"}],
            "workflow_outputs": [{"name": "outB", "type": "string", "description": "desc"}],
        }

        graph = WorkFlowGraph.from_dict(graph_dict)

        self.assertEqual([("A", "B", 7)], [(edge.source, edge.target, edge.priority) for edge in graph.edges])
        graph_edge_refs = [
            attrs["ref"]
            for source, target, attrs in graph.graph.edges(data=True)
            if source == "A" and target == "B"
        ]
        self.assertEqual([7], [edge.priority for edge in graph_edge_refs])

    def test_to_dict_supports_string_and_dict_agents(self):
        """get_config()/to_dict() must support string agents and convert a callable
        parse_func in a dict agent to its function name (JSON-serializable)."""
        import json

        def my_parser(x):
            return x

        node_a = WorkFlowNode(
            name="A",
            description="d",
            inputs=[Parameter(name="i", type="string", description="x")],
            outputs=[Parameter(name="o", type="string", description="x")],
            agents=["StrAgent"],
        )
        node_b = WorkFlowNode(
            name="B",
            description="d",
            inputs=[Parameter(name="o", type="string", description="x")],
            outputs=[Parameter(name="o2", type="string", description="x")],
            agents=[{"name": "DAgent", "description": "d", "parse_func": my_parser}],
        )
        graph = WorkFlowGraph(
            goal="g",
            nodes=[node_a, node_b],
            workflow_inputs=[Parameter(name="i", type="string", description="x")],
            workflow_outputs=[Parameter(name="o2", type="string", description="x")],
        )

        config = graph.get_config()

        # string agent is preserved as-is
        self.assertEqual(["StrAgent"], config["nodes"][0]["agents"])
        # callable parse_func is converted to its name
        self.assertEqual("my_parser", config["nodes"][1]["agents"][0]["parse_func"])
        # whole config is JSON-serializable
        json.dumps(config)

    def test_derived_workflow_inputs_deduplicated(self):
        """When workflow_inputs is not provided, two initial nodes sharing the same input
        name must not produce a duplicate (which the uniqueness check would reject)."""
        node_a = WorkFlowNode(
            name="A",
            description="d",
            inputs=[Parameter(name="shared_in", type="string", description="x")],
            outputs=[Parameter(name="outA", type="string", description="x")],
            agents=["TestAgent"],
        )
        node_b = WorkFlowNode(
            name="B",
            description="d",
            inputs=[Parameter(name="shared_in", type="string", description="x")],
            outputs=[Parameter(name="outB", type="string", description="x")],
            agents=["TestAgent"],
        )
        # No workflow_inputs/outputs provided -> they are derived from initial/end nodes.
        graph = WorkFlowGraph(goal="g", nodes=[node_a, node_b])

        input_names = [param.name for param in graph.workflow_inputs]
        self.assertEqual(["shared_in"], input_names)
        self.assertIn("shared_in", graph.workflow_inputs_dict)

    def test_node_input_unknown_source_raises(self):
        """A node input that is neither a workflow input nor any other node's output must fail."""
        workflow_input = Parameter(name="wf_in", type="string", description="workflow input", required=True)
        workflow_output = Parameter(name="wf_out", type="string", description="workflow output")
        
        nodeA = WorkFlowNode(
            name="NodeA",
            description="NodeA",
            inputs=[Parameter(name="wf_in", type="string", description="workflow input", required=True)],
            outputs=[Parameter(name="nodeA_out", type="string", description="node A output")]
        )

        nodeB = WorkFlowNode(
            name="NodeB",
            description="NodeB",
            inputs=[
                Parameter(name="ghost_input", type="string", description="ghost_input", required=True),
                Parameter(name="nodeA_out", type="string", description="node A output", required=True)
            ],
            outputs=[Parameter(name="wf_out", type="string", description="workflow output")]
        )

        with pytest.raises(
            ValueError,
            match="Node 'NodeB' input 'ghost_input' is not a workflow input or from another node's output.",
        ):
            WorkFlowGraph(
                goal="test",
                nodes=[nodeA, nodeB],
                workflow_inputs=[workflow_input],
                workflow_outputs=[workflow_output],
            )

    def _node_facecolors(self, fig):
        from matplotlib.collections import PathCollection

        ax = fig.axes[0]
        collection = next(coll for coll in ax.collections if isinstance(coll, PathCollection))
        return [tuple(round(float(channel), 5) for channel in row) for row in collection.get_facecolor()]

    def test_display_default_is_nonblocking(self):
        signature = inspect.signature(WorkFlowGraph.display)
        self.assertIs(signature.parameters["block"].default, False)

    def test_display_updates_colors_without_blocking(self):
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        graph = self.linear_graph
        fig = graph.display()
        try:
            self.assertIsNotNone(fig)
            self.assertTrue(graph._display_live)
            self.assertIs(graph._display_fig, fig)

            before = self._node_facecolors(fig)
            graph.running("Task1")
            after = self._node_facecolors(fig)
            task_index = list(graph.graph.nodes).index("Task1")
            self.assertNotEqual(before[task_index], after[task_index])
            self.assertGreater(after[task_index][0], after[task_index][2])

            graph.close_display()
            self.assertFalse(graph._display_live)
            self.assertIsNone(graph._display_fig)
            self.assertFalse(plt.fignum_exists(fig.number))

            graph.completed("Task1")
            self.assertIsNone(graph._display_fig)
        finally:
            plt.close("all")

    def test_display_block_waits_until_window_closes(self):
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        graph = self.linear_graph
        with patch.object(WorkFlowGraph, "_matplotlib_gui_available", return_value=True), \
             patch("matplotlib.pyplot.show") as show:
            fig = graph.display(block=True)
        try:
            show.assert_called_once_with(block=True)
            self.assertFalse(graph._display_live)
            self.assertIsNone(graph._display_fig)
        finally:
            plt.close(fig)
            plt.close("all")

    def test_display_show_uses_nonblocking_flag(self):
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        graph = self.linear_graph
        with patch.object(WorkFlowGraph, "_matplotlib_gui_available", return_value=True), \
             patch("matplotlib.pyplot.show") as show:
            fig = graph.display()
        try:
            show.assert_called_once_with(block=False)
            self.assertTrue(graph._display_live)
        finally:
            graph.close_display()
            plt.close(fig)
            plt.close("all")

    def test_display_empty_graph_does_not_open_window(self):
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        graph = self.linear_graph
        graph.graph.clear()
        open_figures = set(plt.get_fignums())
        try:
            self.assertIsNone(graph.display())
            self.assertEqual(open_figures, set(plt.get_fignums()))
        finally:
            plt.close("all")


if __name__ == "__main__":
    unittest.main()
