from functools import lru_cache
from typing import Set

from pydantic_settings import BaseSettings, SettingsConfigDict


class RedisKeySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="redis_key")

    stream_task_create: str = "gw::task::create"
    stream_task_finish: str = "gw::task::finish"

    readgroup_task_finish_notifier: str = "gw::task::finish::notifier"


class Config(BaseSettings):
    redis_host: str = "127.0.0.1"
    redis_port: int = 6379
    image_prefix: str = "gw::image"
    image_lifetime_s: int = 30
    allow_format: Set[str] = {"png", "jpg", "jpeg"}

    redis_keys: RedisKeySettings = RedisKeySettings()


@lru_cache
def get_global_config():
    return Config()
