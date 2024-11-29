from abc import ABC, abstractmethod
from datetime import datetime
from enum import StrEnum
from typing import List, Optional

from loguru import logger
from pydantic import BaseModel
from redis import ConnectionPool, Redis

from .streams import RedisStream
from .utils import generate_a_random_hex_str

RUNNER_ID_LENGTH = 4


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


class Runner:

    def __init__(self, rdb: Redis, name: str) -> None:
        self._name = name
        self._rdb = rdb

    @property
    def name(self) -> str:
        return self._name

    @property
    def redis_client(self):
        return self._rdb

    @property
    def keys(self) -> Keys:
        return Keys(self.name)

    @property
    def model_id(self) -> str:
        return self.redis_client.hget(self.keys.base, "model_id").decode()

    @property
    def ctime(self) -> datetime:
        resp = self.redis_client.hget(self.keys.base, "ctime")
        return datetime.fromisoformat(resp.decode())

    @property
    def utime(self) -> datetime:
        resp = self.redis_client.hget(self.keys.base, "utime")
        return datetime.fromisoformat(resp.decode())

    @utime.setter
    def utime(self, dt: datetime):
        self.redis_client.hset(self.keys.base, "utime", dt.isoformat())

    @property
    def is_busy(self) -> bool:
        busy = int(self.redis_client.hget(self.keys.base, "busy"))
        return busy == 1

    @is_busy.setter
    def is_busy(self, busy: bool):
        self.redis_client.hset(self.keys.base, "busy", 1 if busy else 0)

    @property
    def is_alive(self) -> bool:
        alive = int(self.redis_client.hget(self.keys.base, "is_alive"))
        return alive == 1

    @is_alive.setter
    def is_alive(self, alive: bool):
        self.redis_client.hset(self.keys.base, "is_alive", 1 if alive else 0)

    @property
    def task(self) -> Optional[str]:
        resp = self.redis_client.hget(self.keys.base, "task")
        return resp.decode() if resp is not None else None

    @task.setter
    def task(self, tid: Optional[str]):
        if tid is None:
            self.redis_client.hdel(self.keys.base, "task")
        else:
            self.redis_client.hset(self.keys.base, "task", tid)

    @property
    def heartbeat(self) -> Optional[datetime]:
        resp = self.redis_client.get(self.keys.heartbeat)
        return datetime.fromisoformat(resp.decode()) if resp is not None else None

    @property
    def stream(self) -> RedisStream:
        return RedisStream(
            self.keys.stream,
            self.keys.readgroup,
            connection_pool=self.redis_client.connection_pool,
        )

    def stop(self):
        self.stream.publish(Message(cmd=Command.stop).model_dump())

    def run_task(self, tid: str):
        self.stream.publish(
            Message(cmd=Command.task, data=tid.encode()).model_dump())

    def update_heartbeat(self, dt: datetime, ttl: float):
        self.redis_client.set(self.keys.heartbeat, dt.isoformat(), ex=ttl)

    def clean_heartbeat(self):
        self.redis_client.delete(self.keys.heartbeat)


class RunnerPool:

    def __init__(
        self,
        host="127.0.0.1",
        port=6379,
        db=0,
        rdb: Redis = None,
        connection_pool: ConnectionPool = None,
        starter: WorkerStarter = None,
    ) -> None:
        if connection_pool is not None:
            self._rdb = Redis(connection_pool=connection_pool)
        elif rdb is not None:
            self._rdb = rdb
        elif host and port and db:
            self._rdb = Redis(host=host, port=port, db=db)
        else:
            raise ValueError(
                "no valid redis connection provided for runner pool.")

        if starter is None:
            raise ValueError(
                "runner pool must have a woker starter, but actual none.")
        self._starter = starter

    def get(self, name: str) -> Optional[Runner]:
        keys = Keys(name)

        # Try find runner key in redis
        # If key in redis, runner may exists but dead.
        exists = int(self._rdb.exists(keys.base))
        if exists == 0:
            return None
        return Runner(rdb=Redis(connection_pool=self._rdb.connection_pool), name=name)

    def new(self, model_id: str, name: str = None, ctime: datetime = None) -> Runner:
        is_specify_name = name is not None
        if name is None:
            name = generate_a_random_hex_str(length=RUNNER_ID_LENGTH)
        if ctime is None:
            ctime = datetime.now()

        # Check if runner already exists.
        #
        # If runner exists but we not require a name
        # make another name and retry.
        # Else if given a specified name, it can't create new runner.
        runner = self.get(name)
        if runner is not None:
            if is_specify_name:
                raise KeyError(f"runner named {name} already exists")
            while runner is not None:
                name = generate_a_random_hex_str(length=RUNNER_ID_LENGTH)
                runner = self.get(name)

        # Name ok, make a new runner.
        runner = Runner(rdb=Redis(connection_pool=self._rdb.connection_pool),
                        name=name)

        # Write runner metadata.
        self._rdb.hset(
            runner.keys.base,
            mapping={
                "name": name,
                "model_id": model_id,
                "ctime": ctime.isoformat(),
                "utime": ctime.isoformat(),
                "busy": 0,
                "is_alive": 0,
            },
        )
        logger.debug(f"runner data, name [{name}], model id [{model_id}]")

        # Create runner stream and readgroup for command message.
        self._rdb.xgroup_create(
            runner.keys.stream, runner.keys.readgroup, mkstream=True
        )

        # Start runner worker.
        self._starter.start_runner(name=name, model_id=model_id)
        return runner

    def delete(self, name: str):
        r = self.get(name)
        if r is not None:
            r.stop()
        keys = Keys(name)
        self._rdb.delete(keys.base, keys.heartbeat, keys.stream)

    def count(self) -> int:
        return len(self._get_all_runner_keys())

    def runners(self) -> List[Runner]:
        result = []

        runner_keys = self._get_all_runner_keys()
        for k in runner_keys:
            r = Runner(
                rdb=Redis(connection_pool=self._rdb.connection_pool),
                name=k.decode().removesuffix(f"::{Keys.suffix}"),
            )
            result.append(r)

        return result

    def clean_dead_runners(self):
        for r in self.runners():
            if r.is_alive and r.heartbeat is not None:
                continue
            logger.debug(f"runner [{r.name}] is dead, clean.")
            self.delete(r.name)

    def _get_all_runner_keys(self):
        return self._rdb.keys(f"*::{Keys.suffix}")
