from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto

from modal_proto import api_pb2

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AppInfo:
    id: str
    name: str
    url: str


@dataclass(frozen=True, slots=True)
class LogsItem:
    task_id: str
    root_function_id: str
    fd: FD
    data: str
    timestamp: float


@dataclass(frozen=True, slots=True)
class StateUpdate:
    task_id: str
    root_function_id: str
    state: State
    timestamp: float


@dataclass
class TaskInfo:
    task_id: str
    root_function_id: str
    state: State
    last_update: float


class State(Enum):
    """Task states, simplified version of the TaskState enum from the Modal API."""

    PENDING = auto()
    CREATING = auto()
    ACTIVE = auto()
    COMPLETED = auto()

    @classmethod
    def from_proto(cls, proto_enum: api_pb2.TaskState.ValueType) -> State:
        """Create a State from a protobuf enum value."""
        if proto_enum in (
            api_pb2.TaskState.TASK_STATE_CREATED,
            api_pb2.TaskState.TASK_STATE_QUEUED,
            api_pb2.TaskState.TASK_STATE_PREEMPTED,
            api_pb2.TaskState.TASK_STATE_WORKER_ASSIGNED,
        ):
            return cls.PENDING
        if proto_enum in (
            api_pb2.TaskState.TASK_STATE_CREATING_CONTAINER,
            api_pb2.TaskState.TASK_STATE_LOADING_IMAGE,
            api_pb2.TaskState.TASK_STATE_LOADING_CHECKPOINT_IMAGE,
        ):
            return cls.CREATING
        if proto_enum in (
            api_pb2.TaskState.TASK_STATE_IDLE,
            api_pb2.TaskState.TASK_STATE_ACTIVE,
            api_pb2.TaskState.TASK_STATE_PREEMPTIBLE,
        ):
            return cls.ACTIVE
        if proto_enum in (api_pb2.TaskState.TASK_STATE_COMPLETED,):
            return cls.COMPLETED

        raise ValueError(f'Unknown task state: {proto_enum}')


class FD(Enum):
    """File descriptor types for logs."""

    STDOUT = 1
    """Standard output stream from within the container."""
    STDERR = 2
    """Standard error stream from within the container."""
    INFO = 3
    """Messages from the Modal infrastructure, not from user code."""

    @classmethod
    def from_proto(cls, proto_enum: api_pb2.FileDescriptor.ValueType) -> FD:
        """Create an FD from a protobuf enum value."""
        if proto_enum == api_pb2.FileDescriptor.FILE_DESCRIPTOR_STDOUT:
            return cls.STDOUT
        if proto_enum == api_pb2.FileDescriptor.FILE_DESCRIPTOR_STDERR:
            return cls.STDERR
        if proto_enum == api_pb2.FileDescriptor.FILE_DESCRIPTOR_INFO:
            return cls.INFO
        raise ValueError(f'Unknown file descriptor: {proto_enum}')
