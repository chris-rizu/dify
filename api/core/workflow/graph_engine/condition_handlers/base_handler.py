from abc import ABC, abstractmethod

from core.workflow.entities import Graph
from core.workflow.entities.graph_init_params import GraphInitParams
from core.workflow.entities.route_node_state import RouteNodeState
from core.workflow.graph_engine.entities.run_condition import RunCondition
from core.workflow.runtime_state import GraphRuntimeState


class RunConditionHandler(ABC):
    def __init__(self, init_params: GraphInitParams, graph: Graph, condition: RunCondition):
        self.init_params = init_params
        self.graph = graph
        self.condition = condition

    @abstractmethod
    def check(self, graph_runtime_state: GraphRuntimeState, previous_route_node_state: RouteNodeState) -> bool:
        """
        Check if the condition can be executed

        :param graph_runtime_state: graph runtime state
        :param previous_route_node_state: previous route node state
        :return: bool
        """
        raise NotImplementedError
