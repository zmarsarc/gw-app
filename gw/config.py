from functools import lru_cache
from typing import Set

from pydantic_settings import BaseSettings


class Config(BaseSettings):
    redis_host: str = "127.0.0.1"
    redis_port: int = 6379
    image_prefix: str = "gw::image"
    image_lifetime_s: int = 30
    allow_format: Set[str] = {"png", "jpg", "jpeg"}
    task_stream_name: str = "inference-task"


@lru_cache
def get_global_config():
    return Config()
