from functools import lru_cache
from typing import Set

from pydantic_settings import BaseSettings, SettingsConfigDict


class RedisSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="redis_")

    host: str = "127.0.0.1"
    port: int = 6379

    image_prefix: str = "gw::image"
    task_prefix: str = "gw::task"

    s_task_create: str = "gw::task::create"
    s_task_finish: str = "gw::task::finish"

    rg_task_finish: str = "gw::task::finish::notifier"
    rg_task_create: str = "gw::task::create::dispatcher"


class AppSettings(BaseSettings):

    redis: RedisSettings = RedisSettings()

    allow_format: Set[str] = {"png", "jpg", "jpeg"}

    pending_message_claim_time_ms: int = 30 * 1000

    image_lifetime_s: int = 24 * 60 * 60
    task_lifetime_s: int = 24 * 60 * 60

    runner_slot_num: int = 10


@lru_cache
def get_app_settings():
    return AppSettings()
