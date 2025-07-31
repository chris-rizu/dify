from collections.abc import Sequence
from datetime import datetime
from typing import Optional

from pydantic import Field

from core.model_runtime.entities.llm_entities import LLMUsage
from core.rag.entities.citation_metadata import RetrievalSourceMetadata
from core.workflow.entities.node_entities import AgentNodeStrategyInit, NodeRunResult
from core.workflow.events.base import BaseNodeEvent, NodeEvent


class NodeRunStartedEvent(BaseNodeEvent):
    predecessor_node_id: Optional[str] = None
    """predecessor node id"""
    parallel_mode_run_id: Optional[str] = None
    """iteration node parallel mode run id"""
    agent_strategy: Optional[AgentNodeStrategyInit] = None


class NodeRunStreamChunkEvent(BaseNodeEvent):
    chunk_content: str = Field(..., description="chunk content")
    from_variable_selector: Optional[list[str]] = None
    """from variable selector"""


class NodeRunRetrieverResourceEvent(BaseNodeEvent):
    retriever_resources: Sequence[RetrievalSourceMetadata] = Field(..., description="retriever resources")
    context: str = Field(..., description="context")


class NodeRunSucceededEvent(BaseNodeEvent):
    pass


class NodeRunFailedEvent(BaseNodeEvent):
    error: str = Field(..., description="error")


class NodeRunExceptionEvent(BaseNodeEvent):
    error: str = Field(..., description="error")


class NodeInIterationFailedEvent(BaseNodeEvent):
    error: str = Field(..., description="error")


class NodeInLoopFailedEvent(BaseNodeEvent):
    error: str = Field(..., description="error")


class NodeRunRetryEvent(NodeRunStartedEvent):
    error: str = Field(..., description="error")
    retry_index: int = Field(..., description="which retry attempt is about to be performed")
    start_at: datetime = Field(..., description="retry start time")


# Events from nodes/events.py
class RunCompletedEvent(NodeEvent):
    run_result: NodeRunResult = Field(..., description="run result")


class RunStreamChunkEvent(NodeEvent):
    chunk_content: str = Field(..., description="chunk content")
    from_variable_selector: list[str] = Field(..., description="from variable selector")


class RunRetrieverResourceEvent(NodeEvent):
    retriever_resources: Sequence[RetrievalSourceMetadata] = Field(..., description="retriever resources")
    context: str = Field(..., description="context")


class ModelInvokeCompletedEvent(NodeEvent):
    """
    Model invoke completed
    """

    text: str
    usage: LLMUsage
    finish_reason: str | None = None


class RunRetryEvent(NodeEvent):
    """Node Run Retry event"""

    error: str = Field(..., description="error")
    retry_index: int = Field(..., description="Retry attempt number")
    start_at: datetime = Field(..., description="Retry start time")
