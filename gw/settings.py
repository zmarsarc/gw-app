from functools import lru_cache
from typing import Set

from pydantic_settings import BaseSettings, SettingsConfigDict


class Keys(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REID_KEY")

    max_runner_num: str = "runner_num::dispatcher::gw"


class AppSettings(BaseSettings):

    redis_host: str = "127.0.0.1"
    redis_port: int = 6379
    redis_db: int = 0

    log_level: str = "DEBUG"

    allow_format: Set[str] = {"png", "jpg", "jpeg"}

    pending_message_claim_time_ms: int = 30 * 1000

    image_lifetime_s: int = 24 * 60 * 60
    task_lifetime_s: int = 24 * 60 * 60

    runner_slot_num: int = 10
    runner_heartbeat_ttl_s: int = 60
    runner_heartbeat_update_period_s: int = 30


@lru_cache
def get_app_settings():
    return AppSettings()

keys = Keys()