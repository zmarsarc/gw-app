from functools import lru_cache
from typing import Set

from pydantic_settings import BaseSettings, SettingsConfigDict


class RedisSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="redis_")

    host: str = "127.0.0.1"
    port: int = 6379

    image_prefix: str = "gw::image"

    s_task_create: str = "gw::task::create"
    s_task_finish: str = "gw::task::finish"

    rg_task_finish: str = "gw::task::finish::notifier"
    rg_task_create: str = "gw::task::create::dispatcher"


class AppSettings(BaseSettings):

    redis: RedisSettings = RedisSettings()

    image_lifetime_s: int = 30
    allow_format: Set[str] = {"png", "jpg", "jpeg"}

    pending_message_claim_time_ms: int = 30 * 1000


@lru_cache
def get_app_settings():
    return AppSettings()
