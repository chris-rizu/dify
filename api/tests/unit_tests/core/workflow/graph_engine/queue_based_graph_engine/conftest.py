import queue
import threading
import time
from typing import Optional

from core.workflow.events import GraphEngineEvent
from core.workflow.graph_engine.queue_based_graph_engine.entities import Task


class MockTaskQueue:
    """
    Mock implementation of a task queue that tracks all operations
    for testing purposes. Implements the TaskQueueProtocol[Task].
    """

    def __init__(self):
        self._queue = queue.Queue()
        self.put_history = []  # Track all put operations
        self.get_history = []  # Track all get operations
        self.put_count = 0
        self.get_count = 0
        self.task_timestamps = {}  # Track when each task was queued/processed

    def put(self, item: Task, block: bool = True, timeout: Optional[float] = None) -> None:
        """Add a task to the queue and record the operation"""
        self.put_count += 1
        self.put_history.append({"task": item, "timestamp": time.time(), "thread": threading.current_thread().name})
        self.task_timestamps[f"{item.node_id}_queued"] = time.time()
        self._queue.put(item, block=block, timeout=timeout)

    def get(self, block: bool = True, timeout: Optional[float] = None) -> Task:
        """Get a task from the queue and record the operation"""
        task = self._queue.get(block=block, timeout=timeout)
        self.get_count += 1
        self.get_history.append({"task": task, "timestamp": time.time(), "thread": threading.current_thread().name})
        self.task_timestamps[f"{task.node_id}_processed"] = time.time()
        return task

    def empty(self) -> bool:
        """Check if queue is empty"""
        return self._queue.empty()

    def qsize(self) -> int:
        """Get queue size"""
        return self._queue.qsize()

    def task_done(self) -> None:
        """Mark task as done"""
        self._queue.task_done()

    def join(self) -> None:
        """Wait for all tasks to complete"""
        self._queue.join()


class MockEventQueue:
    """
    Mock implementation of an event queue that tracks all operations
    for testing purposes. Implements the TaskQueueProtocol[GraphEngineEvent].
    """

    def __init__(self):
        self._queue = queue.Queue()
        self.put_history = []  # Track all put operations
        self.get_history = []  # Track all get operations
        self.put_count = 0
        self.get_count = 0

    def put(self, item: GraphEngineEvent, block: bool = True, timeout: Optional[float] = None) -> None:
        """Add an event to the queue and record the operation"""
        self.put_count += 1
        self.put_history.append({"event": item, "timestamp": time.time(), "thread": threading.current_thread().name})
        self._queue.put(item, block=block, timeout=timeout)

    def get(self, block: bool = True, timeout: Optional[float] = None) -> GraphEngineEvent:
        """Get an event from the queue and record the operation"""
        event = self._queue.get(block=block, timeout=timeout)
        self.get_count += 1
        self.get_history.append({"event": event, "timestamp": time.time(), "thread": threading.current_thread().name})
        return event

    def empty(self) -> bool:
        """Check if queue is empty"""
        return self._queue.empty()

    def qsize(self) -> int:
        """Get queue size"""
        return self._queue.qsize()

    def task_done(self) -> None:
        """Mark task as done"""
        self._queue.task_done()

    def join(self) -> None:
        """Wait for all tasks to complete"""
        self._queue.join()
