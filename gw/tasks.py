from typing import Optional

import redis

from .redis_keys import RedisKeys
from .settings import get_app_settings
from .utils import generate_a_random_hex_str

_settings = get_app_settings()


class Task(redis.Redis):

    def __init__(self, tid: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tid = tid

    @property
    def task_id(self) -> str:
        return self._tid

    @property
    def model_id(self) -> str:
        return self.hget(RedisKeys.task(self.task_id), "model_id").decode()

    @property
    def post_process(self) -> str:
        return self.hget(RedisKeys.task(self.task_id), "post_process").decode()

    @property
    def image_url(self) -> str:
        return self.hget(RedisKeys.task(self.task_id), "image_url").decode()

    @property
    def callback(self) -> str:
        return self.hget(RedisKeys.task(self.task_id), "callback").decode()

    @property
    def inference_result(self) -> Optional[str]:
        res: bytes = self.get(RedisKeys.inference_result(self.task_id))
        return res.decode() if res is not None else res

    @property
    def postprocess_result(self) -> Optional[str]:
        res: bytes = self.get(RedisKeys.inference_result(self.task_id))
        return res.decode() if res is not None else res

    @property
    def ttl(self) -> int:
        return int(super().ttl(RedisKeys.task(self.task_id)))

    @inference_result.setter
    def inference_result(self, data: str):
        self.set(RedisKeys.inference_result(self.task_id), data, ex=self.ttl)

    @postprocess_result.setter
    def postprocess_result(self, data: str):
        self.set(RedisKeys.postprocess_result(self.task_id), data, ex=self.ttl)


class TaskPool(redis.Redis):

    TASK_ID_LENGTH = 16

    def __init__(self, task_ttl: int = _settings.task_lifetime_s, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._task_ttl = task_ttl

    def new(self, model_id: str, image_url: str, post_process: str, callback: str, task_id: str = None) -> Task:
        if task_id is None:
            task_id = generate_a_random_hex_str(self.TASK_ID_LENGTH)
        self.hset(RedisKeys.task(task_id), mapping={
            "task_id": task_id,
            "model_id": model_id,
            "image_url": image_url,
            "post_process": post_process,
            "callback": callback
        })
        self.expire(RedisKeys.task(task_id), self._task_ttl)
        return Task(tid=task_id, connection_pool=self.connection_pool)

    def get(self, task_id: str) -> Optional[Task]:
        exists = int(self.exists(RedisKeys.task(task_id)))
        if exists != 1:
            return None
        return Task(task_id, connection_pool=self.connection_pool)

    def delete(self, task_id: str):
        super().delete(RedisKeys.task(task_id))
