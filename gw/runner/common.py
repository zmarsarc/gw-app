from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Optional

from pydantic import BaseModel


class WorkerStarter(ABC):

    @abstractmethod
    def start_runner(self, name: str, model_id: str):
        pass


class Keys:

    suffix: str = "gw::runner"

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def base(self) -> str:
        return f"{self._name}::{self.suffix}"

    @property
    def heartbeat(self) -> str:
        return f"{self.base}::heartbeat"

    @property
    def stream(self) -> str:
        return f"{self.base}::stream"
    
    @property
    def readgroup(self) -> str:
        return f"{self.base}::readgroup"


class Command(StrEnum):
    stop = "stop"
    task = "task"


class Message(BaseModel):
    cmd: Command
    data: bytes = bytes()
