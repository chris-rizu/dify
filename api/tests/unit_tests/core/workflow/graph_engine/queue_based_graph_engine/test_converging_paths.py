"""
Test cases for queue-based graph engine with converging paths.

This module tests workflows where different conditional paths converge to the same end node.
"""

import time
from unittest.mock import patch

import pytest

from core.workflow.entities import Graph, GraphRuntimeState, VariablePool
from core.workflow.events import GraphRunStartedEvent, GraphRunSucceededEvent
from core.workflow.graph_engine.queue_based_graph_engine import QueueBasedGraphEngine
from core.workflow.system_variable import SystemVariable

from .conftest import MockEventQueue, MockTaskQueue


@pytest.mark.skip
class TestQueueBasedGraphEngineConvergingPaths:
    """Test cases for queue-based graph engine with paths converging to same end node."""

    def test_if_else_paths_converging_to_same_end(self, app):
        """
        Test if-else node with both paths converging to the same end node.

        Graph structure:

                        ┌─(true)──────────┐
        START -> IF-ELSE                   └──> END
                        └─(false)─> A ─────┘

        Path 1: START -> IF-ELSE -> END (direct path when condition is true)
        Path 2: START -> IF-ELSE -> A -> END (through node A when condition is false)

        Tests:
        1. True condition takes direct path to end
        2. False condition goes through intermediate node A
        3. Both paths converge at the same end node
        4. End node produces correct output regardless of path taken
        """
        # Arrange - Workflow with if-else paths converging to same end
        graph_config = {
            "nodes": [
                {
                    "id": "start",
                    "data": {
                        "type": "start",
                        "title": "Start",
                        "variables": [
                            {
                                "label": "message",
                                "max_length": 100,
                                "options": [],
                                "required": True,
                                "type": "text-input",
                                "variable": "message",
                            },
                        ],
                    },
                },
                {
                    "id": "if-else",
                    "data": {
                        "type": "if-else",
                        "title": "Check Condition",
                        "cases": [
                            {
                                "case_id": "true",
                                "logical_operator": "and",
                                "conditions": [
                                    {
                                        "comparison_operator": "contains",
                                        "variable_selector": ["sys", "query"],
                                        "value": "yes",
                                    }
                                ],
                            }
                        ],
                    },
                },
                {
                    "id": "node-A",
                    "data": {
                        "type": "code",
                        "title": "Process Node A",
                        "code": "result = message + ' (processed by Node A)'",
                        "code_language": "python3",
                        "variables": [{"variable": "message", "value_selector": ["start", "message"]}],
                        "outputs": {"result": {"type": "string"}},
                    },
                },
                {
                    "id": "end",
                    "data": {
                        "type": "end",
                        "title": "End",
                        "outputs": [{"value_selector": ["start", "message"], "variable": "output"}],
                    },
                },
            ],
            "edges": [
                # Start to if-else
                {"id": "e1", "source": "start", "target": "if-else"},
                # True path: if-else directly to end
                {"id": "e2", "source": "if-else", "sourceHandle": "true", "target": "end"},
                # False path: if-else to node-A, then to end
                {"id": "e3", "source": "if-else", "sourceHandle": "false", "target": "node-A"},
                {"id": "e4", "source": "node-A", "target": "end"},
            ],
        }

        # Test case 1: query contains "yes" (true path)
        graph = Graph.init(graph_config=graph_config)
        variable_pool = VariablePool(
            system_variables=SystemVariable(
                user_id="test_user",
                app_id="test_app",
                workflow_id="test_workflow",
                files=[],
                query="yes please",  # Contains "yes" - will take true path
            ),
            user_inputs={"message": "DIRECT_PATH_MESSAGE"},
        )

        runtime_state = GraphRuntimeState(variable_pool=variable_pool, start_at=time.perf_counter())

        # Create custom task queue and event queue
        task_queue = MockTaskQueue()
        event_queue = MockEventQueue()

        engine = QueueBasedGraphEngine(
            graph=graph,
            runtime_state=runtime_state,
            max_execution_steps=100,
            max_execution_time=30,
            worker_count=1,  # Single worker for predictable execution
            task_queue=task_queue,  # Inject custom queue
            event_queue=event_queue,  # Inject custom event queue
        )

        events = list(engine.run())

        # Assert basic event structure
        assert len(events) >= 2
        assert isinstance(events[0], GraphRunStartedEvent)
        assert isinstance(events[-1], GraphRunSucceededEvent)

        # Verify output
        final_outputs = events[-1].outputs
        assert final_outputs is not None
        assert "output" in final_outputs
        assert final_outputs["output"] == "DIRECT_PATH_MESSAGE"

        # Verify custom queue was used
        assert engine.task_queue is task_queue
        assert engine.event_queue is event_queue

        # Verify nodes executed - node-A should NOT be executed
        executed_nodes = [entry["task"].node_id for entry in task_queue.get_history]
        assert "start" in executed_nodes
        assert "if-else" in executed_nodes
        assert "end" in executed_nodes
        assert "node-A" not in executed_nodes
        assert len([n for n in executed_nodes if n != "start"]) == 2  # if-else and end

        # Verify queue is empty after completion
        assert task_queue.empty()
        assert event_queue.empty()

    @patch("core.helper.code_executor.code_executor.CodeExecutor.execute_workflow_code_template")
    def test_if_else_false_path_through_node_a(self, mock_execute_code):
        """
        Test the false path going through node A before reaching end.
        """

        # Mock the code executor to simulate node A processing
        def mock_code_execution(language, code, inputs):
            if "result = message +" in code:
                # Simulate node A processing: message + ' (processed by Node A)'
                message = inputs.get("message", "")
                return {"result": f"{message} (processed by Node A)"}
            return {"result": "default"}

        mock_execute_code.side_effect = mock_code_execution
        # Same graph structure as above
        graph_config = {
            "nodes": [
                {
                    "id": "start",
                    "data": {
                        "type": "start",
                        "title": "Start",
                        "variables": [
                            {
                                "label": "message",
                                "max_length": 100,
                                "options": [],
                                "required": True,
                                "type": "text-input",
                                "variable": "message",
                            },
                        ],
                    },
                },
                {
                    "id": "if-else",
                    "data": {
                        "type": "if-else",
                        "title": "Check Condition",
                        "cases": [
                            {
                                "case_id": "true",
                                "logical_operator": "and",
                                "conditions": [
                                    {
                                        "comparison_operator": "contains",
                                        "variable_selector": ["sys", "query"],
                                        "value": "yes",
                                    }
                                ],
                            }
                        ],
                    },
                },
                {
                    "id": "node-A",
                    "data": {
                        "type": "code",
                        "title": "Process Node A",
                        "code": "result = message + ' (processed by Node A)'",
                        "code_language": "python3",
                        "variables": [{"variable": "message", "value_selector": ["start", "message"]}],
                        "outputs": {"result": {"type": "string"}},
                    },
                },
                {
                    "id": "end",
                    "data": {
                        "type": "end",
                        "title": "End",
                        "outputs": [{"value_selector": ["node-A", "result"], "variable": "output"}],
                    },
                },
            ],
            "edges": [
                {"id": "e1", "source": "start", "target": "if-else"},
                {"id": "e2", "source": "if-else", "sourceHandle": "true", "target": "end"},
                {"id": "e3", "source": "if-else", "sourceHandle": "false", "target": "node-A"},
                {"id": "e4", "source": "node-A", "target": "end"},
            ],
        }

        # Test case 2: query does not contain "yes" (false path through node-A)
        graph = Graph.init(graph_config=graph_config)
        variable_pool = VariablePool(
            system_variables=SystemVariable(
                user_id="test_user",
                app_id="test_app",
                workflow_id="test_workflow",
                files=[],
                query="no thank you",  # Does not contain "yes" - will take false path
            ),
            user_inputs={"message": "FALSE_PATH_MESSAGE"},
        )

        runtime_state = GraphRuntimeState(variable_pool=variable_pool, start_at=time.perf_counter())

        # Create custom task queue and event queue
        task_queue = MockTaskQueue()
        event_queue = MockEventQueue()

        engine = QueueBasedGraphEngine(
            graph=graph,
            runtime_state=runtime_state,
            max_execution_steps=100,
            max_execution_time=30,
            worker_count=1,
            task_queue=task_queue,  # Inject custom queue
            event_queue=event_queue,  # Inject custom event queue
        )

        events = list(engine.run())

        # Verify output - false path should have processed_message from node A
        final_outputs = events[-1].outputs
        assert final_outputs is not None
        assert "output" in final_outputs
        assert final_outputs["output"] == "FALSE_PATH_MESSAGE (processed by Node A)"

        # Verify custom queue was used
        assert engine.task_queue is task_queue
        assert engine.event_queue is event_queue

        # Verify nodes executed - node-A SHOULD be executed
        executed_nodes = [entry["task"].node_id for entry in task_queue.get_history]
        assert "start" in executed_nodes
        assert "if-else" in executed_nodes
        assert "node-A" in executed_nodes
        assert "end" in executed_nodes
        assert len([n for n in executed_nodes if n != "start"]) == 3  # if-else, node-A, and end

        # Verify queue is empty after completion
        assert task_queue.empty()
        assert event_queue.empty()
