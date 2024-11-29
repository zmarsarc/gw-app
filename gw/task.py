from typing import Optional
from .utils import generate_a_random_hex_str
from redis import ConnectionPool, Redis

DEFAULT_TASK_TTL_S = 24 * 60 * 60
TASK_ID_LENGTH = 16

def make_task_key(tid: str) -> str:
    return f"{tid}::task::gw"


class Task:

    def __init__(self, tid: str, rdb: Redis = None, conpool: ConnectionPool = None):
        self._tid = tid

        if rdb:
            self._rdb = rdb
        elif conpool:
            self._rdb = Redis(connection_pool=conpool)
        else:
            raise ValueError("task need a valid redis client.")

    def __del__(self):
        self._rdb.close()

    @property
    def task_id(self) -> str:
        return self._tid

    @property
    def model_id(self) -> str:
        return self._rdb.hget(make_task_key(self.task_id), "model_id").decode()

    @property
    def post_process(self) -> str:
        return self._rdb.hget(make_task_key(self.task_id), "post_process").decode()

    @property
    def image_url(self) -> str:
        return self._rdb.hget(make_task_key(self.task_id), "image_url").decode()

    @property
    def callback(self) -> str:
        return self._rdb.hget(make_task_key(self.task_id), "callback").decode()

    @property
    def result(self) -> Optional[bytes]:
        # TODO: move key to a public lib.
        key = f"{self.task_id}::result::task::gw"
        return self._rdb.get(key)

    @result.setter
    def result(self, data: bytes):
        # TODO: move key to a public lib, make ex equal to the task key.
        key = f"{self.task_id}::result::task::gw"
        self._rdb.set(key, data, ex=DEFAULT_TASK_TTL_S)


class TaskPool:

    def __init__(self, rdb: Redis = None, connection_pool: ConnectionPool = None, ttl: int = DEFAULT_TASK_TTL_S):
        if rdb is not None:
            self._rdb = rdb
        elif connection_pool is not None:
            self._rdb = Redis(connection_pool=connection_pool)
        else:
            raise ValueError("task pool must have a valid redis client.")

        self._ttl = ttl

    def new(self, model_id: str, image_url: str, post_process: str, callback: str, task_id: str = None) -> Task:
        if task_id is None:
            task_id = generate_a_random_hex_str(TASK_ID_LENGTH)
        self._rdb.hset(make_task_key(task_id), mapping={
            "task_id": task_id,
            "model_id": model_id,
            "image_url": image_url,
            "post_process": post_process,
            "callback": callback
        })
        self._rdb.expire(make_task_key(task_id), self._ttl)
        return Task(tid=task_id, conpool=self._rdb.connection_pool)

    def get(self, task_id: str) -> Optional[Task]:
        exists = int(self._rdb.exists(make_task_key(task_id)))
        if exists != 1:
            return None
        return Task(task_id, conpool=self._rdb.connection_pool)

    def delete(self, task_id: str):
        self._rdb.delete(make_task_key(task_id))
